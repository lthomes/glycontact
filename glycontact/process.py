import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import re
import os
from collections import Counter
import subprocess
import json
import requests
import json
import shutil
from io import StringIO
from pathlib import Path
from urllib.parse import quote
from glycowork.motif.annotate import *
from glycowork.motif.graph import *
from glycowork.motif.tokenization import *
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


def get_glycoshape_IUPAC() :
    #get the list of available glycans on glycoshape
    return json.loads(subprocess.run('curl -X GET https://glycoshape.org/api/available_glycans',
        shell=True,capture_output=True,text=True).stdout)['glycan_list']


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
        lines = [line for line in pdb_f if 'ATOM   ' in line and 'REMARK' not in line]
    # Read the relevant lines into a DataFrame using fixed-width format
    return pd.read_fwf(StringIO(''.join(lines)), names=columns, colspecs=[(0, 6), (6, 11), (12, 16), (17, 20), (21, 22), (22, 26),
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


def get_contact_tables(my_path, glycan, link_type):
    pdbs = [f"{my_path}{glycan}/{pdb}" for pdb in os.listdir(f"{my_path}{glycan}") 
            if link_type in pdb]
    return [make_monosaccharide_contact_table(
        annotation_pipeline(f, glycan, threshold=3.5)[0],
        #[2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 2.45, 2.55, 2.65, 2.75, 2.85, 2.95, 3, 2.2, 2.25, 2.3, 2.35, 3.5])[0],
        mode='distance', threshold=200) for f in sorted(pdbs)]


def inter_structure_variability_table(my_path, glycan, link_type, mode='standard'):
    ### Creates a table as make_atom_contact_table() or the monosaccharide equivalent, 
    ### but values represent the stability of monosaccharides/atoms across different PDB of the same molecule.
    ### Includes weighted scores calculation based on cluster frequencies only if in "weighted" mode.
    # my_path : path to the folder containing all PDB folders
    # glycan : glycan in IUPAC sequence
    # link_type : 'alpha' or 'beta' to work with alpha- or beta-linked glycans
    # mode : can be 'standard' (compute the sum of the absolute distances to the mean), 
    #        'amplify' (uses the power 2 of the sum which decreases noise and increases outliers importance),
    #        or 'weighted' (computes weighted deviations using cluster frequencies).
    dfs = get_contact_tables(my_path, glycan, link_type)
    columns = dfs[0].columns
    values_array = np.array([df.values for df in dfs])
    mean_values = np.mean(values_array, axis=0)
    deviations = np.abs(values_array - mean_values)
    if mode == 'weighted':
        weights = np.array(get_all_clusters_frequency()[glycan]) / 100
        result = np.average(deviations, weights=weights, axis=0)
    elif mode == 'amplify':
        result = np.sum(deviations, axis=0) ** 2
    else:  # standard mode
        result = np.sum(deviations, axis=0)
    return pd.DataFrame(result, columns=columns, index=columns)


def make_correlation_matrix(my_path, glycan, link_type):
    ### Compute a Pearson correlation matrix
    # my_path : path to the folder containing all PDB folders
    # glycan : glycan in IUPAC sequence
    # link_type : 'alpha' or 'beta' to work with alpha- or beta-linked glycans
    dfs = get_contact_tables(my_path, glycan, link_type)
    # Create an empty correlation matrix
    corr_sum = np.zeros((len(dfs[0]), len(dfs[0])))
    # Calculate the correlation matrix based on the distances
    for df in dfs:
        corr_sum += np.corrcoef(df.values, rowvar=False)
    return pd.DataFrame(corr_sum/len(dfs), columns=df.columns, index=df.columns)


def inter_structure_frequency_table(my_path, glycan, link_type, threshold = 5):
    ### Creates a table as make_atom_contact_table() or the monosaccharide equivalent but values represent the frequency of monosaccharides/atoms pairs that crossed a threshold distance across different PDB of the same molecule
    # my_path : path to the folder containing all PDB folders
    # glycan : glycan in IUPAC sequence
    # link_type : 'alpha' or 'beta' to work with alpha- or beta-linked glycans
    # threshold : maximal distance a pair can show to be counted as a contact
    dfs = get_contact_tables(my_path, glycan, link_type)
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
        valid_mapped = mapped_to_check in valid_fragments
        is_special_case = mapped_to_check in {
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
        if valid_mapped or is_special_case:
            if mapped_to_check == 'Man(a1-4)':
                mapping_dict[mono] = 'Man(b1-4)'
            else:
                mapping_dict[mono] = mapped_to_check
            mono_key = f"{mono.split('_')[0]}_({map_dict[mono.split('_')[1]].split('(')[1]}{first_val}-{last_val})"
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


def annotation_pipeline(pdb_file, glycan,threshold=[2.2,2.4,2.5,2.6,2.7,2.8,2.9,2.25,2.45,2.55,2.65,2.75,2.85,2.95,3]) :
  ### Huge function combining all smaller ones required to annotate a PDB file into IUPAC nomenclature, ensuring that the conversion is correct
  ### It allows also to determine if PDB to IUPAC conversion at the monosaccharide level works fine
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


def monosaccharide_preference_structure(df,monosaccharide,threshold, mode='default'):
  #return the preferred partner of a given monosaccharide, except those closer than the threshold (which will be considered as covalent linkages)
  #df must be a monosaccharide distance table correctly reanotated
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


def show_monosaccharide_preference_structure(df, monosaccharide, threshold, mode='default'):
  #df must be a monosaccharide distance table correctly reanotated
  #mode can be 'default' (check individual monosaccharides in glycan), 'monolink' (check monosaccharide-linkages in glycan), 'monosaccharide' (check monosaccharide types)
  res_dict = monosaccharide_preference_structure(df, monosaccharide, threshold, mode)
  value_counts = Counter(res_dict.values())
  # Plotting the histogram
  plt.bar(value_counts.keys(), value_counts.values())
  plt.xlabel('Values')
  plt.ylabel('Frequency')
  plt.title(f'Frequency of Encountered Values for {monosaccharide} above {threshold}')
  plt.tight_layout()
  plt.show()


def multi_glycan_monosaccharide_preference_structure(prefix,suffix,glycan_sequence,monosaccharide,threshold, mode='default'):
  ### with multiple dicts accross multiple structures
  # prefix : directory (ex: "PDB_format_ATOM2")
  # suffix : 'alpha' or 'beta'
  # glycan_sequence : IUPAC
  pdb_pattern = f"{prefix}/{glycan_sequence}_{suffix}_*.pdb"
  dict_list = []
  for pdb_file in Path(prefix).glob(f"{glycan_sequence}_{suffix}_*.pdb"):
        try:
            annotated_df = annotation_pipeline(str(pdb_file), glycan_sequence)
            if not annotated_df.empty:
                dist_table = make_monosaccharide_contact_table(annotated_df, mode='distance')
                data_dict = monosaccharide_preference_structure(dist_table, monosaccharide, 
                                                              threshold, mode)
                dict_list.append(data_dict)
        except Exception as e:
            continue
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


def get_all_clusters_frequency():
  ### Extract all glycan cluster frequencies from glycoshape and returns a dict
  response = requests.get("https://glycoshape.org/database/GLYCOSHAPE.json")
  if response.status_code == 200:
        data = response.json()
        return {value["iupac"]: [value["clusters"][key] for key in value["clusters"]]
                for key, value in data.items()}
  return {}


def glycan_cluster_pattern(threshold = 70, mute = False) :
    ### Parse all clusters of all glycans on glycoshape. 
    ### Returns glycans with one major cluster AND glycans with many minor clusters
    ### Classification is performed based on a proportion threshold (default = 70)
    # threshold: proportion that the main cluster must have to be considered as a major cluster
    # if mute is set to True, then the prints are ignored
    # If the proportion of the main cluster is lower, the current glycan is assumed to be represented by multiple structural clusters
    all_frequencies = get_all_clusters_frequency()
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


def get_sasa_table(my_path, glycan, mode = 'alpha') :
    #mode determines if we are analysing alpha- or beta-linked glycans
    pattern = 'alpha' if mode == 'alpha' else 'beta'
    pdb_files = sorted(str(p) for p in Path(f"{my_path}{glycan}").glob(f"*{pattern}*"))
    sasa_values = {}
    cluster_frequencies = get_all_clusters_frequency()[glycan]
    weights = np.array([n / 100 for n in cluster_frequencies])
    # Process each PDB file
    for pdb_file in pdb_files:
        structure = md.load(pdb_file)
        sasa = md.shrake_rupley(structure, mode='atom')
        # Group SASA by residue
        mono_sasa = {}
        for atom in structure.topology.atoms:
            res = atom.residue
            if res.resSeq not in mono_sasa:
                mono_sasa[res.resSeq] = {'resName': res.name, 'sasa': 0}
            mono_sasa[res.resSeq]['sasa'] += sasa[0][atom.index]
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
        'Mean Score': [], 'Median Score': [],
        'Weighted Score': [], 'Standard Deviation': [],
        'Coefficient of Variation': []
    }
    for resSeq, data in stats.items():
        values = np.array(data['values'])
        mean = np.mean(values)
        df_data['Monosaccharide_id'].append(resSeq)
        df_data['Monosaccharide'].append(data['resName'])
        df_data['Mean Score'].append(mean)
        df_data['Median Score'].append(np.median(values))
        df_data['Weighted Score'].append(np.average(values, weights=weights))
        std = np.std(values)
        df_data['Standard Deviation'].append(std)
        df_data['Coefficient of Variation'].append(std / mean if mean != 0 else 0)
    table = pd.DataFrame(df_data)
    # Update monosaccharide names using mapping
    df = annotation_pipeline(pdb_files[0], glycan)
    if not df.empty:
        mapping_dict = df.set_index('residue_number')['IUPAC'].to_dict()
        table['Monosaccharide'] = table['Monosaccharide_id'].map(mapping_dict)
    return table


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


def global_monosaccharide_unstability(variability_table, mode='sum'):
    # plot monolink variability for all clusters of a given glycan
    # possible formats: png, pdf
    # mode: sum, mean
    if mode == 'sum':
        residue_stability = variability_table.sum()
    else:  # mode == 'mean'
        residue_stability = variability_table.mean()
    return sorted(residue_stability.items(), key=lambda x: x[1])


def compute_merge_SASA_flexibility(mypath, glycan, flex_mode, global_flex_mode) :
    # flex_mode : standard, amplify, weighted
    # global_flex_mode : sum, mean
    try:
        sasa = get_sasa_table(mypath, glycan, 'beta')
    except:
        sasa = pd.DataFrame()
        print('SASA failed, continuing with empty table')
    try:
        flex = inter_structure_variability_table(mypath, glycan, 'beta', mode=flex_mode)
        mean_flex = global_monosaccharide_unstability(flex, mode=global_flex_mode)
        flex_col = f'{flex_mode}_{global_flex_mode}_flexibility'
        flex_df = pd.DataFrame(mean_flex, columns=['Monosaccharide_id_Monosaccharide', flex_col])
        flex_df['Monosaccharide_id'] = flex_df['Monosaccharide_id_Monosaccharide'].str.split('_').str[0].astype(int)
    except:
        if sasa.empty:
            return pd.DataFrame(columns=[
                'Monosaccharide_id', 'Monosaccharide', 'Mean Score', 'Median Score',
                'Weighted Score', 'Standard Deviation', 'Coefficient of Variation',
                f'{flex_mode}_{global_flex_mode}_flexibility'
            ])
        flex_df = pd.DataFrame(columns=['Monosaccharide_id', f'{flex_mode}_{global_flex_mode}_flexibility'])
        print('Flex calculation failed')
    if sasa.empty:
        return flex_df
    return pd.merge(sasa, flex_df[['Monosaccharide_id', f'{flex_mode}_{global_flex_mode}_flexibility']], 
                   on='Monosaccharide_id', how='left')


def map_data_to_graph(computed_df, interaction_dict) :
    # map the interaction dict to SASA/Flex values computed to produce a graph with node-level information
    # Create edges from simplified interaction dict
    edges = {(int(k.split('_')[0]), int(v.split('_')[0])) 
            for k, values in interaction_dict.items() 
            for v in values 
            if k.split('_')[0] != v.split('_')[0]}
    G = nx.Graph()
    G.add_edges_from(edges)
    # Add node attributes
    for _, row in computed_df.iterrows():
        node_id = row['Monosaccharide_id']
        attrs = {}
        # Add monosaccharide info
        attrs['Monosaccharide'] = row.get('Monosaccharide', node_id)
        # Add SASA scores if available
        for col in ['Mean Score', 'Median Score', 'Weighted Score']:
            if col in row:
                attrs[col] = row[col]
        # Add flexibility if available
        if 'weighted_mean_flexibility' in row:
            attrs['weighted_mean_flexibility'] = row['weighted_mean_flexibility']
        G.add_node(node_id, **attrs)
    return G


def check_graph_content(G) : 
    # Print the nodes and their attributes
    print("Graph Nodes and Their Attributes:")
    for node, attrs in G.nodes(data=True):
        print(f"Node {node}: {attrs}")
    # Print the edges
    print("\nGraph Edges:")
    for edge in G.edges():
        print(edge)


def get_score_list(datatable, my_glycans_path, glycan, mode, column):
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
