import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import re
import os
import copy
from collections import Counter
import subprocess
import json
import requests
import shutil
import random
from io import StringIO
from pathlib import Path
from urllib.parse import quote
from typing import Tuple, Dict, List
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from glycowork.motif.graph import glycan_to_nxGraph, glycan_to_graph
from glycowork.motif.annotate import link_find
from glycowork.motif.processing import canonicalize_iupac
import mdtraj as md

# MAN indicates either alpha and beta bonds, instead of just alpha.. this is a problem
# GalNAc is recorded as "GLC" which is wrong: need for a checker function that counts the number of atoms - Glc = 21 (<25), GalNAc = 28 (>25)
map_dict = {'NDG':'GlcNAc(a','NAG':'GlcNAc(b','MAN':'Man(a', 'BMA':'Man(b', 'AFL':'Fuc(a',
              'FUC':'Fuc(a', 'FUL':'Fuc(b', 'FCA':'dFuc(a', 'FCB':'dFuc(b', '0FA':'Fuc(a', 'GYE':'dFucf(b',
              'GAL':'Gal(b', 'GLA':'Gal(a', 'GIV':'lGal(b', 'GXL':'lGal(a', 'GZL':'Galf(b',
              'GLC':'Glc(a', '0WB':'ManNAc(b', 'ZAD':'Ara(b', '0aU':'Ara(b', '2aU':'Ara(b', '3aU':'Ara(b', '0aD':'Ara(a', '2aD':'Ara(a', '3aD':'Ara(a',
              'IDR':'IdoA(a', 'RAM':'Rha(a', 'RHM':'Rha(b', 'RM4':'Rha(b', 'XXR':'dRha(a',
              '0AU':'Ara(b', '2AU':'Ara(b', '3AU':'Ara(b', '0AD':'Ara(a', '2AD':'Ara(a', '3AD':'Ara(a',
              'A2G':'GalNAc(a', 'NGA': 'GalNAc(b', 'YYQ':'lGlcNAc(a', 'XYP':'Xyl(b', 'XYS':'Xyl(a',
              'XYZ':'Xylf(b', '1CU': 'Fru(b',  '0CU': 'Fru(b', '1CD': 'Fru(a', 'LXC':'lXyl(b', 'HSY':'lXyl(a', 'SIA':'Neu5Ac(a', 'SLB':'Neu5Ac(b',
              'NGC':'Neu5Gc(a', 'NGE':'Neu5Gc(b', 'BDP':'GlcA(b', 'GCU':'GlcA(a','VYS':'GlcNS(a', '0YS':'GlcNS(a', '4YS':'GlcNS(a', '6YS':'GlcNS(a', 'UYS':'GlcNS(a', 'QYS':'GlcNS(a', 'GCS':'GlcN(b', 
              'PA1':'GlcN(a', 'ROH':' ', 'BGC':'Glc(b', '0OA':'GalA(a', '4OA':'GalA(a', 'BCA':'2-4-diacetimido-2-4-6-trideoxyhexose(a',
              "NAG6SO3":"GlcNAc6S(b", "NDG6SO3":"GlcNAc6S(a", "GLC4SO3":"GalNAc4S(b", "NGA4SO3":"GalNAc4S(b", 'A2G4SO3':'GalNAc4S(a', "IDR2SO3":"IdoA2S(a", 
              "BDP3SO3":"GlcA3S(b", "BDP2SO3":"GlcA2S(b", "GCU2SO3":"GlcA2S(a", "SIA9ACX":"Neu5Ac9Ac(b", "MAN3MEX":"Man3Me(a", 
              "SIA9MEX":"Neu5Ac9Me(a", "NGC9MEX":"Neu5Gc9Me(a", "BDP4MEX":"GlcA4Me(b", "GAL6SO3":"Gal6S(b", "NDG3SO3":"GlcNAc3S6S(a",
              "NAG6PCX":"GlcNAc6Pc(b", "UYS6SO3":"GlcNS6S(a", 'VYS3SO3':'GlcNS3S6S(a',  'VYS6SO3':'GlcNS3S6S(a', "QYS3SO3":"GlcNS3S6S(a", "QYS6SO3":"GlcNS3S6S(a", "4YS6SO3":"GlcNS6S(a", "6YS6SO3":"GlcNS6S(a"}

PACKAGE_ROOT = Path(__file__).parent.parent
global_path = PACKAGE_ROOT / 'glycans_pdb/'
this_dir = Path(__file__).parent
json_path = this_dir / "20250205_GLYCOSHAPE.json"
with open(json_path) as f:
    glycoshape_mirror = json.load(f)


def get_glycoshape_IUPAC(fresh=False) :
    #get the list of available glycans on glycoshape
    if fresh:
        return json.loads(subprocess.run('curl -X GET https://glycoshape.org/api/available_glycans', shell=True,capture_output=True,text=True).stdout)['glycan_list']
    else:
        return set(entry["iupac"] for entry in glycoshape_mirror.values())


def download_from_glycoshape(my_path, IUPAC):
    # Download pdb files given an IUPAC sequence that exists in the glycoshape database
    if ')' not in IUPAC:
       print('This IUPAC corresponds to a single monosaccharide: ignored')
       return False
    if IUPAC[-1]==']':
       print('This IUPAC is not formatted properly: ignored')
       return False
    outpath = f'{my_path}/{IUPAC}'
    IUPAC_name = quote(IUPAC)
    os.makedirs(outpath, exist_ok=True)
    for linktype in ['alpha', 'beta']:
        for i in range(0, 500):
            output = f'_{linktype}_{i}.pdb'
            url = f'https://glycoshape.org/database/{IUPAC_name}/PDB_format_ATOM/cluster{i}_{linktype}.PDB.pdb'
            if "404 Not Found" in subprocess.run(f'curl "{url}"', shell=True, capture_output=True, text=True).stdout:
                break
            subprocess.run(f'curl -o {output} "{url}"', shell=True)
            new_name = f'{IUPAC}{output}'
            os.rename(output, new_name)
            shutil.move(new_name, outpath)


def extract_3D_coordinates(pdb_file):
    """
    Extract 3D coordinates from a PDB file and return them as a DataFrame.

    Parameters:
    - pdb_file (str): Path to the PDB file.

    Returns:
    - pd.DataFrame: DataFrame containing the extracted coordinates.
    """
    # Define column names for the DataFrame
    columns = ['record_name', 'atom_number', 'atom_name', 'monosaccharide', 'chain_id', 'residue_number',
               'x', 'y', 'z', 'occupancy', 'temperature_factor', 'element']
    # Open the PDB file for reading
    with open(pdb_file) as pdb_f:
        lines = [line for line in pdb_f if line.startswith('ATOM')]
    # Read the relevant lines into a DataFrame using fixed-width format
    return pd.read_fwf(StringIO(''.join(lines)), names=columns, colspecs=[(0, 6), (6, 11), (12, 16), (17, 20), (20, 22), (22, 26),
                                                         (30, 38), (38, 46), (46, 54), (54, 60), (60, 66), (76, 78)])


def make_atom_contact_table(coord_df, threshold = 10, mode = 'exclusive') :
    ### Create a contact table of atoms of a given PDB file processed into a dataframe by extract_3D_coordinates()
    # coord_df : a dataframe as returned by extract_3D_coordinates()
    # threshold : maximal distance to be considered. Otherwise set to threshold + 1
    # mode : 'exclusive' to avoid intra-residue distance calculation/representation, or 'inclusive' to include intra-residue values
    mono_nomenclature = 'IUPAC' if 'IUPAC' in coord_df else 'monosaccharide'
    coords = coord_df[['x', 'y', 'z']].values
    diff = coords[:, np.newaxis, :] - coords
    distances = np.abs(diff).sum(axis=2)
    labels = [f"{num}_{mono}_{atom}_{anum}" for num, mono, atom, anum in 
             zip(coord_df['residue_number'], coord_df[mono_nomenclature], 
                 coord_df['atom_name'], coord_df['atom_number'])]
    if mode == 'exclusive':
        # Mask intra-residue distances
        mask = coord_df['residue_number'].values[:, np.newaxis] != coord_df['residue_number'].values
        distances = np.where(mask, np.where(distances <= threshold, distances, threshold + 1), 0)
    else:
        distances = np.where(distances <= threshold, distances, threshold + 1)
    return pd.DataFrame(distances, index=labels, columns=labels)


def make_monosaccharide_contact_table(coord_df, threshold = 10, mode = 'binary') :
    # threshold : maximal distance to be considered. Otherwise set to threshold + 1
    # mode : can be either binary (return a table with 0 or 1 based on threshold), distance (return a table with the distance or threshold+1 based on threshold), or both
    mono_nomenclature = 'IUPAC' if 'IUPAC' in coord_df.columns else 'monosaccharide'
    residues = sorted(coord_df['residue_number'].unique())
    n_residues = len(residues)
    binary_matrix = np.ones((n_residues, n_residues))
    dist_matrix = np.full((n_residues, n_residues), threshold + 1)
    labels = [f"{i}_{coord_df[coord_df['residue_number']==i][mono_nomenclature].iloc[0]}" 
             for i in residues]
    for i, res1 in enumerate(residues):
        coords1 = coord_df[coord_df['residue_number']==res1][['x','y','z']].values
        for j, res2 in enumerate(residues):
            coords2 = coord_df[coord_df['residue_number']==res2][['x','y','z']].values
            # Compute all pairwise distances
            diffs = coords1[:, np.newaxis, :] - coords2
            distances = np.abs(diffs).sum(axis=2)
            min_dist = np.min(distances)
            if min_dist <= threshold:
                binary_matrix[i,j] = 0
                dist_matrix[i,j] = min_dist
    if mode == 'binary':
        return pd.DataFrame(binary_matrix, index=labels, columns=labels)
    if mode == 'distance':
        return pd.DataFrame(dist_matrix, index=labels, columns=labels)
    return [pd.DataFrame(binary_matrix, index=labels, columns=labels),
            pd.DataFrame(dist_matrix, index=labels, columns=labels)]


def focus_table_on_residue(table, residue) :
    ### Take a monosaccharide Contact Table and focus it to keep only one residue type (ie MAN)
    mask = table.columns.str.contains(residue, regex=False)
    return table.loc[mask, mask]


def get_contact_tables(glycan, stereo, level="monosaccharide", my_path=None):
    dfs, _ = annotation_pipeline(glycan, pdb_file=my_path, threshold=3.5, stereo=stereo)
    if level == "monosaccharide":
        return [make_monosaccharide_contact_table(df, mode='distance', threshold=200) for df in dfs]
    else:
        return [make_atom_contact_table(df, mode='distance', threshold=200) for df in dfs]


def inter_structure_variability_table(glycan, stereo='alpha', mode='standard', my_path=None, fresh=False):
    ### Creates a table as make_atom_contact_table() or the monosaccharide equivalent, 
    ### but values represent the stability of monosaccharides/atoms across different PDB of the same molecule.
    ### Includes weighted scores calculation based on cluster frequencies only if in "weighted" mode.
    # my_path : path to the folder containing all PDB folders
    # glycan : glycan in IUPAC sequence
    # stereo : 'alpha' or 'beta' to work with alpha- or beta-linked glycans
    # mode : can be 'standard' (compute the sum of the absolute distances to the mean), 
    #        'amplify' (uses the power 2 of the sum which decreases noise and increases outliers importance),
    #        or 'weighted' (computes weighted deviations using cluster frequencies).
    if isinstance(glycan, str):
        dfs = get_contact_tables(glycan, stereo, my_path=my_path)
    elif isinstance(glycan, list):
        dfs = glycan
    columns = dfs[0].columns
    values_array = np.array([df.values for df in dfs])
    mean_values = np.mean(values_array, axis=0)
    deviations = np.abs(values_array - mean_values)
    if mode == 'weighted':
        weights = np.array(get_all_clusters_frequency(fresh=fresh)[glycan]) / 100
        result = np.average(deviations, weights=weights, axis=0)
    elif mode == 'amplify':
        result = np.sum(deviations, axis=0) ** 2
    else:  # standard mode
        result = np.sum(deviations, axis=0)
    return pd.DataFrame(result, columns=columns, index=columns)


def make_correlation_matrix(glycan, stereo='alpha', my_path=None):
    ### Compute a Pearson correlation matrix
    # my_path : path to the folder containing all PDB folders
    # glycan : glycan in IUPAC sequence
    # stereo : 'alpha' or 'beta' to work with alpha- or beta-linked glycans
    if isinstance(glycan, str):
        dfs = get_contact_tables(glycan, stereo, my_path=my_path)
    elif isinstance(glycan, list):
        dfs = glycan
    # Create an empty correlation matrix
    corr_sum = np.zeros((len(dfs[0]), len(dfs[0])))
    # Calculate the correlation matrix based on the distances
    for df in dfs:
        corr_sum += np.corrcoef(df.values, rowvar=False)
    return pd.DataFrame(corr_sum/len(dfs), columns=df.columns, index=df.columns)


def inter_structure_frequency_table(glycan, stereo='alpha', threshold = 5, my_path=None):
    ### Creates a table as make_atom_contact_table() or the monosaccharide equivalent but values represent the frequency of monosaccharides/atoms pairs that crossed a threshold distance across different PDB of the same molecule
    # my_path : path to the folder containing all PDB folders
    # glycan : glycan in IUPAC sequence
    # stereo : 'alpha' or 'beta' to work with alpha- or beta-linked glycans
    # threshold : maximal distance a pair can show to be counted as a contact
    if isinstance(glycan, str):
        dfs = get_contact_tables(glycan, stereo, my_path=my_path)
    elif isinstance(glycan, list):
        dfs = glycan
    # Apply thresholding and create a new list of transformed DataFrames
    binary_arrays = [df.values < threshold for df in dfs]
    # Sum up the transformed DataFrames to create the final DataFrame
    return pd.DataFrame(sum(binary_arrays), columns=dfs[0].columns, index=dfs[0].columns)


def extract_binary_interactions_from_PDB(coordinates_df):
    """
    Extract binary interactions between C1-2 atoms and oxygen atoms from a DataFrame.
    Parameters:
    - coordinates_df (pd.DataFrame): DataFrame obtained using extract_3D_coordinates.
    Returns:
    - pd.DataFrame: DataFrame with columns 'Atom', 'Column', and 'Value' representing interactions.
    """
    carbon_mask = coordinates_df['atom_name'].isin(['C1', 'C2'])
    oxygen_mask = coordinates_df['element'] == 'O'
    carbons = coordinates_df[carbon_mask]
    oxygens = coordinates_df[oxygen_mask]
    c_coords = carbons[['x', 'y', 'z']].values
    o_coords = oxygens[['x', 'y', 'z']].values
    c_residues = carbons['residue_number'].values
    o_residues = oxygens['residue_number'].values
    c_labels = [f"{r}_{m}_{a}" for r, m, a in 
                zip(carbons['residue_number'], carbons['monosaccharide'], 
                    carbons['atom_name'])]
    o_labels = [f"{r}_{m}_{a}" for r, m, a in 
                zip(oxygens['residue_number'], oxygens['monosaccharide'], 
                    oxygens['atom_name'])]
    interactions = []
    for i, (c_coord, c_res, c_label) in enumerate(zip(c_coords, c_residues, c_labels)):
        # Compute all distances
        distances = np.abs(o_coords - c_coord).sum(axis=1)
        mask = (o_residues != c_res)
        if mask.any():
            min_idx = distances[mask].argmin()
            min_idx = np.where(mask)[0][min_idx]
            interactions.append({
                'Atom': c_label,
                'Column': o_labels[min_idx],
                'Value': distances[min_idx]
            })
    return pd.DataFrame(interactions)


def create_mapping_dict_and_interactions(df, valid_fragments, n_glycan) :
  #df is an interaction dataframe as returned by extract_binary_interactions_from_PDB()
  # valid_fragments : obtained from glycowork to ensure that we only append valid monolinks into mapping dict
  # n_glycan : True or False, indicates if the first mannose should be corrected or not
  special_cases = {
            'Man(a1-4)', '-R', 'GlcNAc(a1-1)', 'GlcNAc(b1-1)', 'GalNAc(a1-1)', 
            'GalNAc(b1-1)', 'Glc(a1-1)', 'Glc(b1-1)', 'Rha(a1-1)', 'Rha(b1-1)', 
            'Neu5Ac(a2-1)', 'Neu5Ac(b2-1)', 'Man(a1-1)', 'Man(b1-1)', 'Gal(a1-1)', 
            'Gal(b1-1)', 'Fuc(a1-1)', 'Fuc(b1-1)', 'Xyl(a1-1)', 'Xyl(b1-1)', 
            'GlcA(a1-1)', 'GlcA(b1-1)', 'GlcNS(a1-1)', 'GlcNS(b1-1)', 'GlcNAc6S(a1-1)', 
            'GlcNAc6S(b1-1)', 'GlcNS6S(a1-1)', 'GlcNS6S(b1-1)', 'GlcNS3S6S(a1-1)', 
            'GlcNS3S6S(b1-1)', '2-4-diacetimido-2-4-6-trideoxyhexose(a1-1)', 
            'GlcA2S(a1-1)', 'GlcA2S(b1-1)', 'Ara(a1-1)', 'Ara(b1-1)', 'Fru(a1-1)', 
            'Fru(b1-1)', 'ManNAc(a1-1)', 'ManNAc(b1-1)'
        }
  mapping_dict = {'1_ROH': '-R'}
  interaction_dict, interaction_dict2 = {}, {}
  wrong_mannose, individual_entities = [], []
  for _, row in df.iterrows():
        first_mono = row['Atom']
        second_mono = row['Column']
        mono = first_mono.rsplit('_', 1)[0]
        second_mono_base = second_mono.rsplit('_', 1)[0]
        first_val = ''.join(re.findall(r'\d+', first_mono.split('_')[-1]))
        last_val = ''.join(re.findall(r'\d+', second_mono.split('_')[-1]))
        individual_entities.extend([mono, second_mono_base])
        individual_entities = list(dict.fromkeys(individual_entities))
        # Handle special MAN case for n_glycan
        if mono.split('_')[1] + f'({first_val}-{last_val})' == "MAN(1-4)" and n_glycan:
            wrong_mannose.append(mono)
        for m in [mono, second_mono_base]:
            if m in wrong_mannose:
                m = f"{m.split('_')[0]}_BMA"
        mapped_to_check = f"{map_dict[mono.split('_')[1]]}{first_val}-{last_val})"
        if (mapped_to_check in valid_fragments) or (mapped_to_check in special_cases):
            mapped_to_use = 'Man(b1-4)' if mapped_to_check == 'Man(a1-4)' else mapped_to_check
            mapping_dict[mono] = mapped_to_use
            mono_key = f"{mono.split('_')[0]}_({mapped_to_use.split('(')[1]}"
            if mono in interaction_dict:
                if second_mono_base not in interaction_dict[mono]:
                    interaction_dict[mono].append(second_mono_base)
                    interaction_dict2[mono] = [mono_key]
                    interaction_dict2[mono_key] = [second_mono_base]
            else:
                interaction_dict[mono] = [second_mono_base]
                interaction_dict2[mono] = [mono_key]
                if mono_key in interaction_dict2:
                    interaction_dict2[mono_key].append(second_mono_base)
                else:
                    interaction_dict2[mono_key] = [second_mono_base]
  return mapping_dict, interaction_dict2


def extract_binary_glycontact_interactions(interaction_dict):
  # transform the interactions detected in the PDB file into IUPAC binary interactions for further comparison to glycowork
  # interaction_dict formatted as: {'12_GAL': ['12_(b1-4)'], '12_(b1-4)': ['11_NAG'], '13_AFL': ['13_(a1-3)'], '13_(a1-3)': ['11_NAG']}
  result = []
  for k, v in interaction_dict.items():
      new_k = k.split('_')[1].replace('(', '').replace(')', '') if '(' in k else map_dict[k.split('_')[1]].split('(')[0]
      new_v = v[0].split('_')[1].replace('(', '').replace(')', '') if '(' in v[0] else map_dict[v[0].split('_')[1]].split('(')[0]
      result.append((new_k, new_v))
  return result


def extract_binary_glycowork_interactions(graph_output):
    """
    Extracts a list of binary interactions from the output of glycan_to_graph function.

    Parameters:
    - graph_output (tuple): The output tuple from glycan_to_graph function.

    Returns:
    - list of binary interactions as pairs of labels.
    """
    mask_dic, adj_matrix = graph_output
    n = len(mask_dic)
    return [(mask_dic[k], mask_dic[j]) for k in range(n) 
            for j in range(k + 1, n) if adj_matrix[k, j] == 1]


def glycowork_vs_glycontact_interactions(glycowork_interactions, glycontact_interactions) :
  # Take two sets of binary interactions to compare them and return any difference other than GlcNAc-a1-1 and a1-1-R (only considered by glycontact)
  ignore_pairs = {
        ('GlcNAc', 'a1-1'), ('a1-1', ' '), ('a2-1', ' '), ('b2-1', ' '),
        ('GlcNAc', 'b1-1'), ('b1-1', ' '), ('GalNAc', 'a1-1'), ('GalNAc', 'b1-1'),
        ('Glc', 'a1-1'), ('Glc', 'b1-1'), ('Rha', 'b1-1'), ('Rha', 'a1-1'),
        ('Neu5Ac', 'b2-1'), ('Neu5Ac', 'a2-1'), ('Man', 'b1-1'), ('Man', 'a1-1'),
        ('Gal', 'b1-1'), ('Gal', 'a1-1'), ('Fuc', 'b1-1'), ('Fuc', 'a1-1'),
        ('Xyl', 'b1-1'), ('Xyl', 'a1-1'), ('GlcA', 'a1-1'), ('GlcA', 'b1-1'),
        ('GlcNS', 'a1-1'), ('GlcNS', 'b1-1'), ('GlcNAc6S', 'a1-1'),
        ('GlcNAc6S', 'b1-1'), ('GlcNS6S', 'a1-1'), ('GlcNS6S', 'b1-1'),
        ('GlcNS3S6S', 'a1-1'), ('GlcNS3S6S', 'b1-1'),
        ('2-4-diacetimido-2-4-6-trideoxyhexose', 'a1-1'), ('GlcA2S', 'a1-1'),
        ('GlcA2S', 'b1-1'), ('Ara', 'a1-1'), ('Ara', 'b1-1'), ('Fru', 'a1-1'),
        ('Fru', 'b1-1'), ('ManNAc', 'a1-1'), ('ManNAc', 'b1-1')
    }
  differences = set(glycontact_interactions) ^ set(glycowork_interactions)
  filtered_differences = [pair for pair in differences if pair not in ignore_pairs]
  return (not filtered_differences and len(glycontact_interactions) > len(glycowork_interactions))


def check_reconstructed_interactions(interaction_dict) :
  # Use the interaction_dict to build a NetworkX network and check that it contains a single component (meaning that the glycan has been correctly reconstructed from the PDB file)
  # Create a directed graph
  G = nx.Graph()
  # Add nodes and edges from dictionary interactions
  G.add_edges_from((node, neighbor) 
                     for node, neighbors in interaction_dict.items() 
                     for neighbor in neighbors)
  return nx.is_connected(G)


def annotate_pdb_data(pdb_dataframe, mapping_dict) :
  m_dict = copy.deepcopy(mapping_dict)
  for m, v in m_dict.items():
        if "BMA" in m:
            mapping_dict[f"{m.split('_')[0]}_MAN"] = v #restore the corrected mannose into a wrong one for annotation
  pdb_dataframe['lookup_key'] = pdb_dataframe['residue_number'].astype(str) + '_' + pdb_dataframe['monosaccharide']
  # Map values using the dictionary, falling back to original monosaccharide
  pdb_dataframe['IUPAC'] = pdb_dataframe['lookup_key'].map(mapping_dict).fillna(pdb_dataframe['monosaccharide'])
  # Drop temporary column
  pdb_dataframe.drop('lookup_key', axis=1, inplace=True)
  return pdb_dataframe


def correct_dataframe(df):
  #Correct an annotated dataframe, transforming unexpected GLC into GalNAc based on the number of C atom they contain
  ### WARNING: this is a modified version of the function, assuming that it is always GalNAc(b which is wrong
  c_counts = df[df['element'] == 'C'].groupby('residue_number').size()
  high_carbon_residues = c_counts[c_counts >= 7].index
  # Create masks for GLC and BGC replacements
  glc_mask = (df['monosaccharide'] == 'GLC') & df['residue_number'].isin(high_carbon_residues)
  bgc_mask = (df['monosaccharide'] == 'BGC') & df['residue_number'].isin(high_carbon_residues)
  # Apply replacements
  df.loc[glc_mask, 'monosaccharide'] = 'NGA'
  df.loc[bgc_mask, 'monosaccharide'] = 'A2G'
  return df


def get_annotation(glycan, pdb_file, threshold=3.5):
  MODIFIED_MONO = {
        "GlcNAc6S", "GalNAc4S", "IdoA2S", "GlcA3S", "GlcA2S", "Neu5Ac9Ac", 
        "Man3Me", "Neu5Ac9Me", "Neu5Gc9Me", "GlcA4Me", "Gal6S", "GlcNAc6Pc",
        "GlcNS6S", "GlcNS3S6S"
    }
  NON_MONO = {'SO3', 'ACX', 'MEX', 'PCX'}
  CUSTOM_PDB = {
        "NAG6SO3": "GlcNAc6S", "NDG6SO3": "GlcNAc6S", "NDG3SO3": "GlcNAc3S6S",
        "NGA4SO3": "GalNAc4S", "IDR2SO3": "IdoA2S", "BDP3SO3": "GlcA3S",
        "BDP2SO3": "GlcA2S", "SIA9ACX": "Neu5Ac9Ac", "MAN3MEX": "Man3Me",
        "SIA9MEX": "Neu5Ac9Me", "NGC9MEX": "Neu5Gc9Me", "BDP4MEX": "GlcA4Me",
        "GAL6SO3": "Gal6S", "NAG6PCX": "GlcNAc6Pc", "UYS6SO3": "GlcNS6S",
        "4YS6SO3": "GlcNS6S", "6YS6SO3": "GlcNS6S", "GCU2SO3": "GlcA2S",
        'VYS3SO3': 'GlcNS3S6S', 'VYS6SO3': 'GlcNS3S6S',
        "QYS3SO3": "GlcNS3S6S", "QYS6SO3": "GlcNS3S6S"
    }
  n_glycan = 'Man(b1-4)GlcNAc(b1-4)' in glycan
  df = correct_dataframe(extract_3D_coordinates(pdb_file))
  if any(mm in glycan for mm in MODIFIED_MONO):
        # Process modified glycans
        to_modify_dict = {}
        dist_table = make_atom_contact_table(df)
        # Get residue mapping
        resdict = df.groupby('residue_number')['monosaccharide'].first().to_dict()
        # Process non-monosaccharide elements
        for key, val in resdict.items():
            if val in NON_MONO:
                element = f"{key}_{val}"
                contact_table = dist_table.filter(regex=element)
                # Filter contact table
                mask = ~contact_table.index.str.contains('|'.join(contact_table.columns))
                filtered_table = contact_table.loc[mask]
                filtered_table = filtered_table[~filtered_table.index.str.split('_').str[2].str.contains('H')]
                # Find closest partner
                if not filtered_table.empty:
                    partners = filtered_table[filtered_table != 0].stack().idxmin()
                    sugar_partner = partners[0]
                    sugar_resnum, sugar, atom, _ = sugar_partner.split("_")
                    link_pos = re.findall(r'\d+', atom)[0]
                    modified_mono = sugar + link_pos + val
                    if modified_mono in CUSTOM_PDB:
                        to_modify_dict[int(sugar_resnum)] = modified_mono
                        to_modify_dict[key] = [modified_mono, sugar_resnum]
        # Apply modifications to dataframe
        for residue_num, modification in to_modify_dict.items():
            if isinstance(modification, str):
                df.loc[df['residue_number'] == residue_num, 'monosaccharide'] = modification
            else:
                monosaccharide, new_residue = modification
                mask = df['residue_number'] == residue_num
                df.loc[mask, 'monosaccharide'] = monosaccharide
                df.loc[mask, 'residue_number'] = int(new_residue)
        df = df.sort_values('residue_number')
  # Extract and validate linkages
  valid_fragments = {x.split(')')[0] + ')' for x in link_find(glycan)}
  res = extract_binary_interactions_from_PDB(df)
  if isinstance(threshold, float) or isinstance(threshold, int):
      res = res[res.Value < threshold].reset_index(drop=True)
  else:
      for thresh in threshold:
          res = res[res.Value < thresh].reset_index(drop=True)
          if len(res) > 0:
              break
  mapping_dict, interaction_dict = create_mapping_dict_and_interactions(res, valid_fragments, n_glycan)
  # Validate against glycowork
  glycowork_interactions = extract_binary_glycowork_interactions(glycan_to_graph(glycan))
  glycontact_interactions = extract_binary_glycontact_interactions(interaction_dict)
  if (glycowork_vs_glycontact_interactions(glycowork_interactions, glycontact_interactions) and 
        check_reconstructed_interactions(interaction_dict)):
        return annotate_pdb_data(df, mapping_dict), interaction_dict
  return pd.DataFrame(), {}


def annotation_pipeline(glycan, pdb_file = None, threshold=3.5, stereo = "alpha") :
  ### Huge function combining all smaller ones required to annotate a PDB file into IUPAC nomenclature, ensuring that the conversion is correct
  ### It allows also to determine if PDB to IUPAC conversion at the monosaccharide level works fine
  if pdb_file is None:
      pdb_file = os.listdir(global_path / glycan)
      pdb_file = [global_path / glycan / pdb for pdb in pdb_file if stereo in pdb]
  if isinstance(pdb_file, str):
      pdb_file = [pdb_file]
  dfs, int_dicts = zip(*[get_annotation(glycan, pdb, threshold=threshold) for pdb in pdb_file])
  return dfs, int_dicts


def get_example_pdb(glycan, stereo=''):
    pdb_file = os.listdir(global_path / glycan)
    return random.choice([global_path / glycan / pdb for pdb in pdb_file if stereo in pdb])


def monosaccharide_preference_structure(df, monosaccharide, threshold, mode='default'):
  #return the preferred partner of a given monosaccharide, except those closer than the threshold (which will be considered as covalent linkages)
  #df must be a monosaccharide distance table correctly reannotated
  #mode can be 'default' (check individual monosaccharides in glycan), 'monolink' (check monosaccharide-linkages in glycan), 'monosaccharide' (check monosaccharide types)
  # should the observed frequencies be normalized based on the occurence of each monosaccharide? Indeed, if GlcNAc is often close to Man, is it by choice, or because it is surrounded by so many Man that it has no other choice?
  entities = df.columns.tolist()
  preferred_partners = {}
  if '(' not in monosaccharide:
        mono_mask = [e.split('_')[1].split('(')[0] == monosaccharide for e in entities]
  else:
        mono_mask = [e.split('_')[1] == monosaccharide for e in entities]
  target_entities = [e for e, m in zip(entities, mono_mask) if m]
  for entity in target_entities:
        distances = df[entity]
        valid_distances = distances[(distances != 0) & (distances >= threshold)]
        if not valid_distances.empty:
            closest_partner = valid_distances.idxmin()
            preferred_partners[entity] = closest_partner
  if mode == 'default':
        return preferred_partners
  elif mode == 'monolink':
        return {k: v.split('_')[1] for k, v in preferred_partners.items()}
  else:  # mode == 'monosaccharide'
        return {k: v.split('_')[1].split('(')[0] for k, v in preferred_partners.items()}


def multi_glycan_monosaccharide_preference_structure(glycan, stereo, monosaccharide, threshold=3.5, mode='default'):
  ### with multiple dicts accross multiple structures
  # suffix : 'alpha' or 'beta'
  # glycan_sequence : IUPAC
  mono_tables = get_contact_tables(glycan, stereo)
  dict_list = [monosaccharide_preference_structure(dist, monosaccharide, 
                                                              threshold, mode) for dist in mono_tables]
  all_values = [v for d in dict_list for v in d.values()]
  if not all_values:
        return
  value_counts = Counter(all_values)
  plt.bar(value_counts.keys(), value_counts.values())
  plt.xlabel('Values')
  plt.ylabel('Frequency')
  plt.title(f'Frequency for {monosaccharide} above {threshold} across structures')
  plt.tight_layout()
  plt.show()


def get_all_clusters_frequency(fresh=False):
  ### Extract all glycan cluster frequencies from glycoshape and returns a dict
  data = {}
  if fresh:
      response = requests.get("https://glycoshape.org/database/GLYCOSHAPE.json")
      if response.status_code == 200:
        data = response.json()
  else:
      data = glycoshape_mirror
  return {value["iupac"]: [value["clusters"][key] for key in value["clusters"]]
                for key, value in data.items()}


def glycan_cluster_pattern(threshold = 70, mute = False, fresh=False) :
    ### Parse all clusters of all glycans on glycoshape. 
    ### Returns glycans with one major cluster AND glycans with many minor clusters
    ### Classification is performed based on a proportion threshold (default = 70)
    # threshold: proportion that the main cluster must have to be considered as a major cluster
    # if mute is set to True, then the prints are ignored
    # If the proportion of the main cluster is lower, the current glycan is assumed to be represented by multiple structural clusters
    all_frequencies = get_all_clusters_frequency(fresh=fresh)
    major_clusters, minor_clusters = [], []
    for glycan, freqs in all_frequencies.items():
        try:
            if float(freqs[0]) >= threshold:
                major_clusters.append(glycan)
            else:
                minor_clusters.append(glycan)
        except (IndexError, ValueError):
            continue
    if not mute:
        print(f"Number of glycans with one major cluster: {len(major_clusters)}")
        print(f"Number of glycans without a major cluster: {len(minor_clusters)}")
    return major_clusters, minor_clusters


def get_sasa_table(glycan, stereo = 'alpha', my_path=None, fresh=False) :
    if my_path is None:
        pdb_files = os.listdir(global_path / glycan)
        pdb_files = sorted(global_path / glycan / pdb for pdb in pdb_files if stereo in pdb)
    else:
        pdb_files = sorted(str(p) for p in Path(f"{my_path}{glycan}").glob(f"*{stereo}*"))
    weights = np.array(get_all_clusters_frequency(fresh=fresh)[glycan]) / 100
    df, _ = get_annotation(glycan, pdb_files[0], threshold=3.5)
    residue_modifications = df.set_index('residue_number')['IUPAC'].to_dict()
    # Process each PDB file
    sasa_values = {}
    for pdb_file in pdb_files:
        structure = md.load(pdb_file)
        sasa = md.shrake_rupley(structure, mode='atom')
        # Group SASA by residue
        mono_sasa, modification_to_parent = {}, {}
        # First pass: identify modification groups and their parent residues
        for res in structure.topology.residues:
            if res.name == 'SO3':
                # The parent is typically the residue before it
                parent_resSeq = res.resSeq - 1
                modification_to_parent[res.resSeq] = parent_resSeq
        # Second pass: calculate SASA value
        for atom in structure.topology.atoms:
            res = atom.residue
            res_seq = res.resSeq
            # If this is a modification residue, get its parent's resSeq
            if res.name == 'SO3' and res_seq in modification_to_parent:
                parent_resSeq = modification_to_parent[res_seq]
                if parent_resSeq not in mono_sasa:
                    mono_sasa[parent_resSeq] = {
                        'resName': residue_modifications.get(parent_resSeq, 'NAG'),  # Use IUPAC name if available
                        'sasa': 0
                    }
                mono_sasa[parent_resSeq]['sasa'] += sasa[0][atom.index]
                continue
            if res.name == 'SO3':
                continue
            if res_seq not in mono_sasa:
                mono_sasa[res_seq] = {'resName': residue_modifications.get(res_seq, res.name), 'sasa': 0}
            mono_sasa[res_seq]['sasa'] += sasa[0][atom.index]  # Add SASA contribution
        sasa_values[pdb_file] = mono_sasa
    # Calculate statistics
    first_pdb = sasa_values[pdb_files[0]]
    stats = {resSeq: {
        'resName': first_pdb[resSeq]['resName'],
        'values': [sasa_values[pdb][resSeq]['sasa'] for pdb in pdb_files]
    } for resSeq in first_pdb}
    # Create DataFrame
    df_data = {
        'Monosaccharide_id': [], 'Monosaccharide': [],
        'Mean SASA': [], 'Median SASA': [],
        'Weighted SASA': [], 'Standard Deviation': [],
        'Coefficient of Variation': []
    }
    for resSeq, data in stats.items():
        values = np.array(data['values'])
        mean = np.mean(values)
        df_data['Monosaccharide_id'].append(resSeq)
        df_data['Monosaccharide'].append(data['resName'])
        df_data['Mean SASA'].append(mean)
        df_data['Median SASA'].append(np.median(values))
        df_data['Weighted SASA'].append(np.average(values, weights=weights))
        std = np.std(values)
        df_data['Standard Deviation'].append(std)
        df_data['Coefficient of Variation'].append(std / mean if mean != 0 else 0)
    return pd.DataFrame(df_data)


def convert_glycan_to_class(glycan):
    """
    Converts every monosaccharide(linkage) and single monosaccharide into X, XNAc,XA, XN, dX, Sia, Pen in a glycan string.
    
    Parameters:
    - glycan (str): A string representing the glycan in IUPAC format.
    
    Returns:
    - str: The modified glycan string with each monosaccharide replaced by 'X'.
    """
    MONO_CLASSES = {
        'Hex': ['Glc', 'Gal', 'Man', 'Ins', 'Galf', 'Hex'],
        'dHex': ['Fuc', 'Qui', 'Rha', 'dHex'],
        'HexA': ['GlcA', 'ManA', 'GalA', 'IdoA', 'HexA'],
        'HexN': ['GlcN', 'ManN', 'GalN', 'HexN'],
        'HexNAc': ['GlcNAc', 'GalNAc', 'ManNAc', 'HexNAc'],
        'Pen': ['Ara', 'Xyl', 'Rib', 'Lyx', 'Pen'],
        'Sia': ['Neu5Ac', 'Neu5Gc', 'Kdn', 'Sia']
    }
    MONO_MAP = {mono: class_name for class_name, monos in MONO_CLASSES.items() 
                for mono in monos}
    CLASS_NAMES = {'Hex': 'X', 'dHex': 'dX', 'HexA': 'XA', 'HexN': 'XN', 
                  'HexNAc': 'XNAc', 'Pen': 'Pen', 'Sia': 'Sia'}
    glycan = stemify_glycan(glycan)
    result = []
    for part in glycan.replace('[', ' [ ').replace(']', ' ] ').split(')'):
        mono = part.split('(')[0].strip()
        if mono in ['[', ']']:
            result.append(mono)
        else:
            mono_class = MONO_MAP.get(mono)
            result.append(CLASS_NAMES.get(mono_class, 'Unk') if mono_class else 'Unk')
    return ''.join(result)


def group_by_silhouette(glycan_list, mode = 'X'):
    """
    Take a list of glycans and return a dataframe where they are annotated and sorted by their silhouette.
    Glycans with the same silhouette share the same branching/topology (example: Neu5Ac(a2-3)Gal(b1-3)[Neu5Ac(a2-6)]GalNAc 
    and Man(b1-4)GlcNAc(b1-4)[Fuc(a1-6)]GlcNAc share the same silhouette: XX[X]X)
    
    Parameters:
    - glycan_list (list): A list of glycans in IUPAC format.
    - mode (string): either 'X' or 'class' to convert monosaccharide with Xs or more detailed symbols (X, XNAc, Sia...)
    
    Returns:
    - Dataframe: The annotated dataframe.
    """
    silhouettes = {}
    topo_groups = {}
    for glycan in glycan_list:
        if mode == 'X':
            pattern = re.sub(r'[A-Za-z0-9]+(?:\([^\)]+\))?', 'X', glycan)
        else:
            pattern = convert_glycan_to_class(glycan)  
        if pattern not in topo_groups:
            topo_groups[pattern] = len(topo_groups)
        silhouettes[glycan] = {
            'silhouette': pattern,
            'topological_group': topo_groups[pattern]
        }
    df = pd.DataFrame.from_dict(silhouettes, orient='index')
    df.index.name = 'glycan'
    df.reset_index(inplace=True)
    return df.sort_values('topological_group')


def global_monosaccharide_instability(variability_table, mode='sum'):
    # plot monolink variability for all clusters of a given glycan
    # possible formats: png, pdf
    # mode: sum, mean
    if mode == 'sum':
        residue_stability = variability_table.sum()
    else:  # mode == 'mean'
        residue_stability = variability_table.mean()
    return sorted(residue_stability.items(), key=lambda x: x[1])


def compute_merge_SASA_flexibility(glycan, flex_mode, global_flex_mode='mean', stereo='alpha', my_path=None) :
    # flex_mode : standard, amplify, weighted
    # global_flex_mode : sum, mean
    try:
        sasa = get_sasa_table(glycan, stereo, my_path=my_path)
    except:
        sasa = pd.DataFrame()
        print('SASA failed, continuing with empty table')
    try:
        flex = inter_structure_variability_table(glycan, stereo, mode=flex_mode, my_path=None)
        mean_flex = global_monosaccharide_instability(flex, mode=global_flex_mode)
        flex_col = f'{flex_mode}_{global_flex_mode}_flexibility'
        flex_df = pd.DataFrame(mean_flex, columns=['Monosaccharide_id_Monosaccharide', flex_col])
        flex_df['Monosaccharide_id'] = flex_df['Monosaccharide_id_Monosaccharide'].str.split('_').str[0].astype(int)
    except:
        if sasa.empty:
            return pd.DataFrame(columns=[
                'Monosaccharide_id', 'Monosaccharide', 'Mean SASA', 'Median SASA',
                'Weighted SASA', 'Standard Deviation', 'Coefficient of Variation',
                f'{flex_mode}_{global_flex_mode}_flexibility'
            ])
        flex_df = pd.DataFrame(columns=['Monosaccharide_id', f'{flex_mode}_{global_flex_mode}_flexibility'])
        print('Flex calculation failed')
    if sasa.empty:
        return flex_df
    return pd.merge(sasa, flex_df[['Monosaccharide_id', f'{flex_mode}_{global_flex_mode}_flexibility']], 
                   on='Monosaccharide_id', how='left')


def map_data_to_graph(computed_df, interaction_dict, ring_conf_df=None, torsion_df=None) :
    # map the interaction dict to SASA/Flex values computed to produce a graph with node-level information
    # Create edges from simplified interaction dict
    edges = {(int(k.split('_')[0]), int(v.split('_')[0])) 
            for k, values in interaction_dict.items() 
            for v in values 
            if k.split('_')[0] != v.split('_')[0]}
    G = nx.Graph()
    G.add_edges_from(edges)
    # Create a mapping of monosaccharide_id to ring conformation data if available
    ring_conf_map = {}
    if ring_conf_df is not None:
        for _, row in ring_conf_df.iterrows():
            ring_conf_map[row['residue']] = {
                'Q': row['Q'],
                'theta': row['theta'],
                'conformation': row['conformation']
            }
    # Create a mapping for torsion angles
    torsion_map = {}
    if torsion_df is not None:
        for _, row in torsion_df.iterrows():
            # Extract the residue numbers from the linkage string
            res_nums = [match.group() for match in re.finditer(r'\d+', row['linkage'])]  # Gets ["5", "3"] from "5_FUC-3_GAL"
            edge_key = tuple(sorted([int(res_nums[0]), int(res_nums[1])]))
            torsion_map[edge_key] = {
                'phi_angle': row['phi'],
                'psi_angle': row['psi'],
            }
    # Add node attributes
    for _, row in computed_df.iterrows():
        node_id = row['Monosaccharide_id']
        attrs = {}
        # Add monosaccharide info
        attrs['Monosaccharide'] = row.get('Monosaccharide', node_id)
        # Add SASA scores if available
        for col in ['Mean SASA', 'Median SASA', 'Weighted SASA']:
            if col in row:
                attrs[col] = row[col]
        # Add flexibility if available
        if 'weighted_mean_flexibility' in row:
            attrs['weighted_mean_flexibility'] = row['weighted_mean_flexibility']
        # Add ring conformation data if available
        if ring_conf_map and node_id in ring_conf_map:
            attrs.update(ring_conf_map[node_id])
        G.add_node(node_id, **attrs)
    # Add torsion angles as edge attributes
    for edge_key, torsion_data in torsion_map.items():
        if edge_key in G.edges():
            nx.set_edge_attributes(G, {edge_key: torsion_data})
    return G


def remove_and_concatenate_labels(graph):
    nodes_to_remove = []  # List to store nodes that need to be removed
    # Iterate through nodes in sorted order to ensure proper handling
    for node in sorted(graph.nodes):
        if node % 2 == 1:  # Odd index
            neighbors = list(graph.neighbors(node))
            if len(neighbors) > 1:  # Only connect neighbors if there's more than one
                for i in range(len(neighbors)):
                    for j in range(i + 1, len(neighbors)):
                        graph.add_edge(neighbors[i], neighbors[j])  # Add edge between neighbors
            predecessor = node - 1  # Get predecessor index
            if predecessor in graph.nodes:  # Ensure the predecessor exists
                # Concatenate string_labels
                predecessor_label = graph.nodes[predecessor].get("string_labels", "")
                current_label = graph.nodes[node].get("string_labels", "")
                graph.nodes[predecessor]["string_labels"] = predecessor_label + '('+current_label+')'
            nodes_to_remove.append(node)  # Mark node for removal
    # Remove the odd-indexed nodes after processing
    graph.remove_nodes_from(nodes_to_remove)


def trim_gcontact(G_contact) :
    # Remove node 1 which corresponds to -R, absent from G_work
    if 1 in G_contact:
        neighbors = list(G_contact.neighbors(1))  # Get the neighbors of node 1
        if len(neighbors) > 1:  # If node 1 has more than one neighbor
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    G_contact.add_edge(neighbors[i], neighbors[j])  # Add edge between neighbors
        G_contact.remove_node(1)  # Remove node 1


# Function to perform attribute-aware isomorphism check
def compare_graphs_with_attributes(G_contact, G_work):
    # Define a custom node matcher
    def node_match(node_attrs1, node_attrs2):
        # Ensure 'string_labels' in G is part of 'Monosaccharide' in G2
        return (
            'string_labels' in node_attrs1
            and 'Monosaccharide' in node_attrs2
            and node_attrs1['string_labels'] in node_attrs2['Monosaccharide']
        )
    # Create an isomorphism matcher with the custom node matcher
    matcher = nx.isomorphism.GraphMatcher(G_work, G_contact, node_match=node_match)
    mapping_dict = {} # format= gcontact_index: gwork_index
    if matcher.is_isomorphic():  # Check if the graphs are isomorphic
        # Extract the mapping of nodes
        mapping = matcher.mapping
        for node_g, node_g2 in mapping.items():
            mapping_dict[node_g2] = node_g
    else:
        print("The graphs are not isomorphic with the given attribute constraints.")
    return(mapping_dict)


def create_glycontact_annotated_graph(glycan: str, mapping_dict, g_contact) -> nx.Graph:
    """Create a glyco-contact annotated graph with flexibility attributes."""
    glycowork_graph = glycan_to_nxGraph(glycan)
    node_attributes = {node: g_contact.nodes[node]
                           for node in g_contact.nodes}
    # Map attributes to the glycowork graph nodes
    flex_attribute_mapping = {
        mapping_dict[gcontact_node]: attributes
        for gcontact_node, attributes in node_attributes.items()
        if gcontact_node in mapping_dict
    }
    # Assign the mapped attributes to the glycowork graph
    nx.set_node_attributes(glycowork_graph, flex_attribute_mapping)
    # Map torsion angles to linkage nodes
    edge_attributes = nx.get_edge_attributes(g_contact, 'phi_angle')
    for (u, v), phi in edge_attributes.items():
        # Find the linkage node between these monosaccharides in glycowork_graph
        u_mapped = mapping_dict[u]
        v_mapped = mapping_dict[v]
        # Find the node that represents the linkage between u_mapped and v_mapped
        common_neighbors = set(glycowork_graph.neighbors(u_mapped)) & set(glycowork_graph.neighbors(v_mapped))
        for linkage_node in common_neighbors:
            glycowork_graph.nodes[linkage_node].update({
                'phi_angle': g_contact[u][v]['phi_angle'],
                'psi_angle': g_contact[u][v]['psi_angle']
            })
    return glycowork_graph


def get_structure_graph(glycan, stereo='alpha'):
    merged = compute_merge_SASA_flexibility(glycan, 'weighted', stereo=stereo)
    res, datadict = get_annotation(glycan, get_example_pdb(glycan, stereo=stereo), threshold=3.5)
    ring_conf = get_ring_conformations(res)
    torsion_angles = get_glycosidic_torsions(res, datadict)
    G_contact = map_data_to_graph(merged, datadict, ring_conf_df=ring_conf, torsion_df=torsion_angles)
    G_work = glycan_to_nxGraph(canonicalize_iupac(glycan))
    remove_and_concatenate_labels(G_work)
    trim_gcontact(G_contact)
    m_dict = compare_graphs_with_attributes(G_contact, G_work)
    return create_glycontact_annotated_graph(glycan, mapping_dict=m_dict, g_contact=G_contact)


def check_graph_content(G) : 
    # Print the nodes and their attributes
    print("Graph Nodes and Their Attributes:")
    for node, attrs in G.nodes(data=True):
        print(f"Node {node}: {attrs}")
    # Print the edges
    print("\nGraph Edges:")
    for edge in G.edges():
        print(edge)


def get_score_list(datatable, glycan, column):
    #try to extract score in the same order as glycan string to ensure GlycoDraw will plot them correctly
    # datatable is either a SASA table, a flex table, or a merged table
    scores = datatable[column].to_list()[::-1]
    monos = datatable['Monosaccharide'].to_list()[::-1]
    glycan_monos = [m + (')' if '(' in m else '') 
                    for m in glycan.replace('[','').replace(']','').split(')')[:-1]]
    new_scores = []
    i = 0
    while i < len(glycan_monos):
        if i >= len(monos):
            break
        if glycan_monos[i] == monos[i]:
            new_scores.append(scores[i])
            i += 1
        elif i + 1 < len(monos) and glycan_monos[i] == monos[i + 1]:
            new_scores.extend([scores[i + 1], scores[i]])
            i += 2
        else:
            i += 1
    # Add remaining scores
    new_scores.extend(scores[i:i+2])
    return new_scores if len(new_scores) == len(scores) else scores


def extract_glycan_coords(pdb_filepath, residue_ids=None):
    """
    Extract main chain coordinates of glycan residues from PDB file.
    
    Args:
        pdb_file: Path to PDB file
        residue_ids: Optional list of residue numbers to extract. If None, extracts all main chain atoms.
            
    Returns:
        Tuple of (coordinates array, atom labels)
    """
    df = extract_3D_coordinates(pdb_filepath)
    if residue_ids:
        df = df[df['residue_number'].isin(residue_ids)]
    # Get common atoms present in most glycans
    common_atoms = ['C1', 'C2', 'C3', 'C4', 'C5', 'O5']
    coords, atom_labels = [], []
    for _, row in df.iterrows():
        if row['atom_name'] in common_atoms:
            coords.append([row['x'], row['y'], row['z']])
            atom_labels.append(f"{row['residue_number']}_{row['monosaccharide']}_{row['atom_name']}")
    return np.array(coords), atom_labels


def align_point_sets(mobile_coords, ref_coords):
    """
    Find optimal rigid transformation to align two point sets.
    Uses Kabsch algorithm after finding best point correspondences.
    
    Args:
        mobile_coords: Nx3 array of coordinates to transform
        ref_coords: Mx3 array of reference coordinates
        
    Returns:
        Tuple of (transformed coordinates, RMSD)
    """
    def get_rotation_matrix(angles):
        """Create 3D rotation matrix from angles."""
        cx, cy, cz = np.cos(angles)
        sx, sy, sz = np.sin(angles)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        return Rx @ Ry @ Rz

    def objective(params):
        """Objective function to minimize."""
        angles = params[:3]
        translation = params[3:]
        # Apply rotation and translation
        R = get_rotation_matrix(angles)
        transformed = (mobile_coords @ R) + translation
        # Calculate distances between all points
        distances = cdist(transformed, ref_coords)
        # Use sum of minimum distances as score
        return np.min(distances, axis=1).sum()

    # Initial guess
    initial_guess = np.zeros(6)  # 3 rotation angles + 3 translation components
    # Optimize alignment
    result = minimize(objective, initial_guess, method='Nelder-Mead')
    # Get final transformation
    final_angles = result.x[:3]
    final_translation = result.x[3:]
    R = get_rotation_matrix(final_angles)
    transformed_coords = (mobile_coords @ R) + final_translation
    # Calculate final RMSD
    distances = cdist(transformed_coords, ref_coords)
    min_distances = np.min(distances, axis=1)
    rmsd = np.sqrt(np.mean(min_distances ** 2))
    return transformed_coords, rmsd


def superimpose_glycans(ref_pdb, mobile_pdb, ref_residues=None, mobile_residues=None):
    """
    Superimpose two glycan structures and calculate RMSD.
    
    Args:
        ref_pdb: Reference PDB file
        mobile_pdb: Mobile PDB file to superimpose
        ref_residues: Optional list of residue numbers for reference glycan
        mobile_residues: Optional list of residue numbers for mobile glycan
        
    Returns:
        Dict containing:
            - ref_coords: Original coordinates of reference
            - transformed_coords: Aligned mobile coordinates
            - rmsd: Root mean square deviation
            - ref_labels: Atom labels from reference structure
            - mobile_labels: Atom labels from mobile structure
    """
    # Extract coordinates
    ref_coords, ref_labels = extract_glycan_coords(ref_pdb, ref_residues)
    mobile_coords, mobile_labels = extract_glycan_coords(mobile_pdb, mobile_residues)
    transformed_coords, rmsd = align_point_sets(mobile_coords, ref_coords)
    return {
        'ref_coords': ref_coords,
        'transformed_coords': transformed_coords,
        'rmsd': rmsd,
        'ref_labels': ref_labels,
        'mobile_labels': mobile_labels
    }


def calculate_torsion_angle(coords: List[List[float]]) -> float:
    """Calculate torsion angle from 4 xyz coordinates using vector algebra.
    Args:
        coords: List of 4 [x,y,z] coordinates
    Returns:
        float: Torsion angle in degrees
    """
    p = [np.array(p, dtype=float) for p in coords]
    v = [p[1] - p[0], p[2] - p[1], p[3] - p[2]]
    n1, n2 = np.cross(v[0], v[1]), np.cross(v[1], v[2])
    n1 /= np.linalg.norm(n1)
    n2 /= np.linalg.norm(n2)
    return np.degrees(np.arctan2(
        np.dot(np.cross(n1, v[1]/np.linalg.norm(v[1])), n2),
        np.dot(n1, n2)
    ))


def get_glycosidic_torsions(df: pd.DataFrame, interaction_dict: Dict[str, List[str]]) -> pd.DataFrame:
    """Calculate phi/psi torsion angles for all glycosidic linkages in structure.
    Args:
        df: DataFrame with PDB atomic coordinates
        interaction_dict: Dictionary of glycosidic linkages
    Returns:
        DataFrame: Phi/psi angles for each linkage
    """
    results = []
    for donor_key, linkage_info in interaction_dict.items():
        if not any('_(' in link for link in linkage_info):
            continue
        linkage_str = linkage_info[0]
        match = re.match(r'\d+_\(([\w])(\d+)-(\d+)\)', linkage_str)
        if not match:
            continue
        aform, pos = match.group(1), int(match.group(3))
        donor_res = int(donor_key.split('_')[0])
        acceptor_id = interaction_dict[linkage_str][0]
        acceptor_res = int(acceptor_id.split('_')[0])
        if df[df['residue_number'] == acceptor_res]['monosaccharide'].iloc[0] == 'ROH':
            continue
        try:
            donor = df[df['residue_number'] == donor_res]
            acceptor = df[df['residue_number'] == acceptor_res]
            # Special handling for sialic acid
            if 'SIA' in donor_key:
                o5_name = 'O6'  # In sialic acid, O5 is actually O6
                o_pos = 'O1A'   # Sialic acid uses O1A for glycosidic bond
            else:
                o5_name = 'O5'
                o_pos = f'O{pos}'
            coords_phi = [
                donor[donor['atom_name'] == o5_name].iloc[0][['x', 'y', 'z']].values.astype(float),
                donor[donor['atom_name'] == 'C1'].iloc[0][['x', 'y', 'z']].values.astype(float),
                donor[donor['atom_name'] == o_pos].iloc[0][['x', 'y', 'z']].values.astype(float),
                acceptor[acceptor['atom_name'] == f'C{pos}'].iloc[0][['x', 'y', 'z']].values.astype(float)
            ]
            next_c = pos + 1 if pos < 6 else pos - 1
            coords_psi = [coords_phi[1], coords_phi[2], coords_phi[3],
                acceptor[acceptor['atom_name'] == f'C{next_c}'].iloc[0][['x', 'y', 'z']].values.astype(float)]
            results.append({
                'linkage': f"{donor_key}-{acceptor_id}",
                'phi': round(calculate_torsion_angle(coords_phi), 2),
                'psi': round(calculate_torsion_angle(coords_psi), 2),
                'anomeric_form': aform,
                'position': pos
            })
        except (IndexError, KeyError) as e:
            print(f"Warning: Skipping {donor_key}-{acceptor_id}: {str(e)}")
            continue
    return pd.DataFrame(results)


def calculate_ring_pucker(df: pd.DataFrame, residue_number: int) -> Dict:
    """Calculate ring puckering parameters for a monosaccharide using the Cremer-Pople method.
    Args:
        df: DataFrame with PDB coordinates
        residue_number: Residue number to analyze
    Returns:
        Dictionary with puckering parameters
    """
    residue = df[df['residue_number'] == residue_number]
    mono_type = residue['monosaccharide'].iloc[0]
    is_l_sugar = mono_type in {'FUC', 'RAM', 'ARA'}
    # Get ring atoms based on monosaccharide type
    is_sialic = 'SIA' in mono_type
    if is_sialic:  # 9-atom sialic acid rings
        ring_atoms = ['C2', 'C3', 'C4', 'C5', 'C6', 'O6']
    else:  # Standard 6-membered pyranose rings
        ring_atoms = ['C1', 'C2', 'C3', 'C4', 'C5', 'O5']
    # Extract coordinates of ring atoms
    coords = []
    for atom in ring_atoms:
        atom_data = residue[residue['atom_name'] == atom]
        if atom_data.empty:
            raise ValueError(f"Missing ring atom {atom} in residue {residue_number}")
        coords.append(atom_data[['x', 'y', 'z']].values[0].astype(float))
    coords = np.array(coords)
    # Calculate geometrical center
    center = np.mean(coords, axis=0)
    n = len(ring_atoms)
    # Define normal vector to mean plane
    z_vector = np.zeros(3)
    for j in range(n):
        k = (j + 1) % n
        z_vector += np.cross(coords[j] - center, coords[k] - center)
    z_vector /= np.linalg.norm(z_vector)
    # Project atoms onto mean plane
    y_vector = coords[0] - center
    y_vector -= np.dot(y_vector, z_vector) * z_vector
    y_vector /= np.linalg.norm(y_vector)
    x_vector = np.cross(y_vector, z_vector)
    # Calculate puckering coordinates
    zj = np.array([np.dot(coord - center, z_vector) for coord in coords])
    # Calculate puckering amplitudes
    qm = np.zeros(n//2)
    phi = np.zeros(n//2)
    for m in range(n//2):
        qm_sin = 0
        qm_cos = 0
        for j in range(n):
            angle = 2 * np.pi * (m + 1) * j / n
            qm_sin += zj[j] * np.sin(angle)
            qm_cos += zj[j] * np.cos(angle)
        qm[m] = np.sqrt(qm_sin**2 + qm_cos**2) * (2/n)
        phi[m] = np.degrees(np.arctan2(qm_sin, qm_cos)) % 360
    # Total puckering amplitude
    Q = np.sqrt(np.sum(qm**2))
    # Phase angle θ
    if is_sialic:
        # For sialic acid rings (using second largest amplitude)
        theta = np.degrees(np.arccos(qm[2] / Q))
        # Adjust the phase angle calculation for the larger ring
        phi = [np.degrees(np.arctan2(
            np.sum([zj[j] * np.sin(2 * np.pi * (m + 1) * j / n) for j in range(n)]),
            np.sum([zj[j] * np.cos(2 * np.pi * (m + 1) * j / n) for j in range(n)])
        )) % 360 for m in range(n//2)]
    else:
        # For 6-membered rings
        q2 = qm[1]  # Second puckering coordinate
        q3 = qm[2]  # Third puckering coordinate
        theta = np.degrees(np.arctan2(q2, q3))
    # Determine conformation
    conformation = "Unknown"
    if is_sialic:
        # More detailed classification for sialic acids
        # Sialic acids typically prefer a 2C5 chair conformation
        if theta < 30:
            conformation = "2C5"  # Most common in nature
        elif theta > 150:
            conformation = "5C2"  # Less common inverted chair
        elif theta < 90:
            # Add more specific boat type based on phi
            phi_main = phi[2]  # Use different phi for 9-membered ring
            if 330 <= phi_main or phi_main < 30:
                conformation = "B2,5"
            elif 150 <= phi_main < 210:
                conformation = "B3,O6"
        else:
            conformation = "S3,5"  # Most common skew form
    else:
        if theta < 45:
            conformation = "4C1" if not is_l_sugar else "1C4"
        elif theta > 135:
            conformation = "1C4" if not is_l_sugar else "4C1"
        else:
            # Check for boat/skew-boat
            boat_types = {
                0: "B1,4", 60: "B2,5", 120: "B3,6",
                180: "B1,4", 240: "B2,5", 300: "B3,6"
            }
            skew_types = {
                30: "1S3", 90: "2S6", 150: "3S1",
                210: "4S2", 270: "5S3", 330: "6S4"
            }
            phi_main = phi[1]  # Main pseudorotational angle
            # Find closest reference angle
            if abs(phi_main % 60) < 30:
                # Boat conformation
                closest_angle = round(phi_main / 60) * 60
                conformation = boat_types.get(closest_angle % 360, "")
            else:
                # Skew-boat conformation
                closest_angle = round((phi_main - 30) / 60) * 60 + 30
                conformation = skew_types.get(closest_angle % 360, "")
    return {
        'residue': residue_number,
        'monosaccharide': mono_type,
        'Q': round(Q, 3),
        'theta': round(theta, 2),
        'phi': [round(p, 2) for p in phi],
        'conformation': conformation
    }


def get_ring_conformations(df: pd.DataFrame, exclude_types: List[str] = ['ROH']) -> pd.DataFrame:
    """Analyze ring conformations for all residues in structure.
    Args:
        df: DataFrame with PDB coordinates
        exclude_types: List of residue types to exclude
    Returns:
        DataFrame with ring parameters for each residue
    """
    if len(df) < 1:
        return pd.DataFrame(columns=['residue', 'monosaccharide', 'Q', 'theta', 'phi', 'conformation'])
    results = []
    residues = df.groupby('residue_number')['monosaccharide'].first()
    for res_num, mono_type in residues.items():
        if mono_type in exclude_types:
            continue
        try:
            pucker = calculate_ring_pucker(df, res_num)
            results.append(pucker)
        except ValueError as e:
            print(f"Warning: {str(e)}")
            continue   
    return pd.DataFrame(results)
