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
import pickle
from random import Random
from io import StringIO
from tqdm import tqdm
from pathlib import Path
from urllib.parse import quote
from typing import Tuple, Dict, List
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from multiprocessing import Pool
from glycowork.motif.graph import glycan_to_nxGraph, glycan_to_graph
from glycowork.motif.annotate import link_find
from glycowork.motif.processing import canonicalize_iupac, rescue_glycans, min_process_glycans
from glycowork.motif.tokenization import stemify_glycan
import mdtraj as md

# MAN indicates either alpha and beta bonds, instead of just alpha.. this is a problem
# GalNAc is recorded as "GLC" which is wrong: need for a checker function that counts the number of atoms - Glc = 21 (<25), GalNAc = 28 (>25)
map_dict = {'NDG':'GlcNAc(a','NAG':'GlcNAc(b','MAN':'Man(a', 'BMA':'Man(b', 'AFL':'Fuc(a',
              'FUC':'Fuc(a', 'FUL':'Fuc(b', 'FCA':'dFuc(a', 'FCB':'dFuc(b', '0FA':'D-Fuc(a', 'GYE':'dFucf(b',
              'GAL':'Gal(b', 'GLA':'Gal(a', 'GIV':'lGal(b', 'GXL':'lGal(a', 'GZL':'Galf(b', '2kA': 'L-Gul(a', '0mA': 'L-Man(a',
              'GLC':'Glc(a', '0WB':'ManNAc(b', 'ZAD':'Ara(b', '0aU':'Ara(b', '2aU':'Ara(b', '3aU':'Ara(b', '0aD':'Ara(a', '2aD':'Ara(a', '3aD':'Ara(a',
              'IDR':'IdoA(a', 'RAM':'Rha(a', 'RHM':'Rha(b', 'RM4':'Rha(b', 'XXR':'dRha(a', '0aU': 'Araf(b', '2aU': 'Araf(b', '3aU': 'Araf(b', 'ZaU': 'Araf(a',
              '0AU':'Ara(b', '2AU':'Ara(b', '3AU':'Ara(b', '0AD':'Ara(a', '2AD':'Ara(a', '3AD':'Ara(a', '3HA': 'D-Rha(a', 'ARB': 'D-Ara(b',
              'A2G':'GalNAc(a', 'NGA': 'GalNAc(b', 'YYQ':'lGlcNAc(a', 'XYP':'Xyl(b', 'XYS':'Xyl(a', 'WOA': 'GalA(b', '3OA': 'GalA(a', 'TOA': 'GalA(a',
              'XYZ':'Xylf(b', '1CU': 'Fru(b',  '0CU': 'Fru(b', '4CD': 'Fru(a', '1CD': 'Fru(a', 'LXC':'lXyl(b', 'HSY':'lXyl(a', 'SIA':'Neu5Ac(a', 'SLB':'Neu5Ac(b',
              'NGC':'Neu5Gc(a', 'NGE':'Neu5Gc(b', 'BDP':'GlcA(b', 'GCU':'GlcA(a','VYS':'GlcNS(a', '0YS':'GlcNS(a', '4YS':'GlcNS(a', '6YS':'GlcNS(a', 'UYS':'GlcNS(a', 'QYS':'GlcNS(a', 'GCS':'GlcN(b',
              'PA1':'GlcN(a', 'ROH':' ', 'BGC':'Glc(b', '0OA':'GalA(a', '4OA':'GalA(a', 'BCA':'2-4-diacetimido-2-4-6-trideoxyhexose(a',
              "NAG6SO3":"GlcNAc6S(b", "NDG6SO3":"GlcNAc6S(a", "GLC4SO3":"GalNAc4S(b", "NGA4SO3":"GalNAc4S(b", 'A2G4SO3':'GalNAc4S(a', "IDR2SO3":"IdoA2S(a",
              "BDP3SO3":"GlcA3S(b", "BDP2SO3":"GlcA2S(b", "GCU2SO3":"GlcA2S(a", "SIA9ACX":"Neu5Ac9Ac(a", "MAN3MEX":"Man3Me(a",
              "SIA9MEX":"Neu5Ac9Me(a", "NGC9MEX":"Neu5Gc9Me(a", "BDP4MEX":"GlcA4Me(b", "GAL6SO3":"Gal6S(b", "NDG3SO3":"GlcNAc3S6S(a", "TOA2SO3": "GalA2S(a",
              "NAG6PCX":"GlcNAc6PCho(b", "UYS6SO3":"GlcNS6S(a", 'VYS3SO3':'GlcNS3S6S(a',  'VYS6SO3':'GlcNS3S6S(a', "QYS3SO3":"GlcNS3S6S(a", "QYS6SO3":"GlcNS3S6S(a", "4YS6SO3":"GlcNS6S(a", "6YS6SO3":"GlcNS6S(a",
              "FUC2MEX3MEX4MEX": "Fuc2Me3Me4Me(a", "QYS3SO36SO3": "GlcNAc3S6S(a", "VYS3SO36SO3": "GlcNS3S6S(a", "NDG3SO36SO3": "GlcNS3S6S(a", "RAM2MEX3MEX": "Rha2Me3Me(a"}
NON_MONO = {'SO3', 'ACX', 'MEX', 'PCX'}

PACKAGE_ROOT = Path(__file__).parent.parent
global_path = PACKAGE_ROOT / 'glycans_pdb/'
this_dir = Path(__file__).parent
json_path = this_dir / "20250205_GLYCOSHAPE.json"
with open(json_path) as f:
    glycoshape_mirror = json.load(f)
with open(this_dir / "glycan_graphs.pkl", "rb") as file:
    structure_graphs = pickle.load(file)


def get_glycoshape_IUPAC(fresh=False):
  """Retrieves a list of available glycans from GlycoShape database.
  Args:
  fresh (bool): If True, fetches data directly from GlycoShape API.
                   If False, uses cached data from the local mirror.
  Returns:
      set: Set of IUPAC-formatted glycan sequences available in the database.
  """
  if fresh:
    return json.loads(subprocess.run('curl -X GET https://glycoshape.org/api/available_glycans', shell=True, capture_output=True,text=True).stdout)['glycan_list']
  else:
    return set(entry["iupac"] for entry in glycoshape_mirror.values())


def download_from_glycoshape(IUPAC):
  """Downloads PDB files for a given IUPAC sequence from the GlycoShape database.
  Args:
      IUPAC (str): IUPAC-formatted glycan sequence to download.
  Returns:
      bool: False if IUPAC is improperly formatted, None otherwise.
  """
  if IUPAC[-1]==']':
    print('This IUPAC is not formatted properly: ignored')
    return False
  IUPAC_clean = canonicalize_iupac(IUPAC)
  outpath = global_path / IUPAC_clean
  IUPAC_name = quote(IUPAC)
  os.makedirs(outpath, exist_ok=True)
  max_cluster = None
  for linktype in ['alpha', 'beta']:
    for i in range(0, 200):
      if max_cluster is not None and i > max_cluster:
        break
      output = f'_{linktype}_{i}.pdb'
      url = f'https://glycoshape.org/database/{IUPAC_name}/PDB_format_ATOM/cluster{i}_{linktype}.PDB.pdb'
      response = subprocess.run(f'curl "{url}"', shell=True, capture_output=True, text=True).stdout
      if "404 Not Found" in response:
        if max_cluster is None:
          max_cluster = i - 1
        break
      # Only save if it's not a 404
      subprocess.run(f'curl -o {output} "{url}"', shell=True)
      new_name = f'cluster{i}_{linktype}.PDB.pdb'
      os.rename(output, new_name)
      shutil.move(new_name, outpath)


def extract_3D_coordinates(pdb_file):
  """Extracts 3D coordinates from a PDB file and returns them as a DataFrame.
  Args:
      pdb_file (str): Path to the PDB file.
  Returns:
      pd.DataFrame: DataFrame containing extracted atom coordinates with columns for
                   atom information, coordinates, and properties.
  """
  permitted = set(map_dict.keys()) | NON_MONO
  # Define column names for the DataFrame
  columns = ['record_name', 'atom_number', 'atom_name', 'monosaccharide', 'chain_id', 'residue_number',
           'x', 'y', 'z', 'occupancy', 'temperature_factor', 'element']
  has_protein, has_hetatm = False, False
  with open(pdb_file, 'r') as pdb_f:
    lines = pdb_f.readlines()
  for line in lines:
    if line.startswith(('SEQRES', 'DBREF')):  # Protein sequence indicators
      has_protein = True
    if line.startswith('HETATM'):
      has_hetatm = True
    if has_protein and has_hetatm:
      break
  # Open the PDB file for reading
  relevant_lines = [line for line in lines if line.startswith('HETATM')] if has_protein else [line for line in lines if line.startswith('ATOM')]
  # Read the relevant lines into a DataFrame using fixed-width format
  out = pd.read_fwf(StringIO(''.join(relevant_lines)), names=columns, colspecs=[(0, 6), (6, 11), (12, 16), (17, 20), (20, 22), (22, 26),
                                                     (30, 38), (38, 46), (46, 54), (54, 60), (60, 66), (76, 78)])
  return out[out.monosaccharide.isin(permitted)].reset_index(drop=True)


def make_atom_contact_table(coord_df, threshold = 10, mode = 'exclusive') :
  """Creates a contact table showing distances between atoms in a PDB structure.
  Args:
      coord_df (pd.DataFrame): Dataframe of coordinates from extract_3D_coordinates.
      threshold (float): Maximum distance to consider, longer distances set to threshold+1.
      mode (str): 'exclusive' to exclude intra-residue distances, 'inclusive' to include them.
  Returns:
      pd.DataFrame: Matrix of distances between atoms.
  """
  mono_nomenclature = 'IUPAC' if 'IUPAC' in coord_df else 'monosaccharide'
  coords = coord_df[['x', 'y', 'z']].values
  diff = coords[:, np.newaxis, :] - coords
  distances = np.abs(diff).sum(axis=2)
  labels = [f"{num}_{mono}_{atom}_{anum}" for num, mono, atom, anum in
         zip(coord_df['residue_number'], coord_df[mono_nomenclature], coord_df['atom_name'], coord_df['atom_number'])]
  if mode == 'exclusive':
    # Mask intra-residue distances
    mask = coord_df['residue_number'].values[:, np.newaxis] != coord_df['residue_number'].values
    distances = np.where(mask, np.where(distances <= threshold, distances, threshold + 1), 0)
  else:
    distances = np.where(distances <= threshold, distances, threshold + 1)
  return pd.DataFrame(distances, index=labels, columns=labels)


def make_monosaccharide_contact_table(coord_df, threshold = 10, mode = 'binary') :
  """Creates a contact table at the monosaccharide level rather than atom level.
  Args:
      coord_df (pd.DataFrame): Dataframe of coordinates from extract_3D_coordinates.
      threshold (float): Maximum distance to consider.
      mode (str): 'binary' for binary contact matrix, 'distance' for distance values,
                 'both' to return both matrices.
  Returns:
      pd.DataFrame or list: Contact table(s) between monosaccharides.
  """
  mono_nomenclature = 'IUPAC' if 'IUPAC' in coord_df.columns else 'monosaccharide'
  residues = sorted(coord_df['residue_number'].unique())
  n_residues = len(residues)
  binary_matrix = np.ones((n_residues, n_residues))
  dist_matrix = np.full((n_residues, n_residues), threshold + 1)
  labels = [f"{i}_{coord_df[coord_df['residue_number']==i][mono_nomenclature].iloc[0]}" for i in residues]
  coords_by_residue = {res: coord_df[coord_df['residue_number']==res][['x','y','z']].values for res in residues}
  for i, res1 in enumerate(residues):
    coords1 = coords_by_residue[res1]
    for j, res2 in enumerate(residues[i:], i):
      coords2 = coords_by_residue[res2]
      # Compute all pairwise distances
      diffs = coords1[:, np.newaxis, :] - coords2
      distances = np.abs(diffs).sum(axis=2)
      min_dist = np.min(distances)
      if min_dist <= threshold:
        binary_matrix[i, j] = binary_matrix[j, i] = 0
        dist_matrix[i, j] = dist_matrix[j, i] = min_dist
  if mode == 'binary':
    return pd.DataFrame(binary_matrix, index=labels, columns=labels)
  if mode == 'distance':
    return pd.DataFrame(dist_matrix, index=labels, columns=labels)
  return [pd.DataFrame(binary_matrix, index=labels, columns=labels),
        pd.DataFrame(dist_matrix, index=labels, columns=labels)]


def focus_table_on_residue(table, residue) :
  """Filters a monosaccharide contact table to keep only one residue type.
  Args:
      table (pd.DataFrame): Monosaccharide contact table.
      residue (str): Residue type to focus on (e.g., 'MAN').
  Returns:
      pd.DataFrame: Filtered contact table.
  """
  mask = table.columns.str.contains(residue, regex=False)
  return table.loc[mask, mask]


def get_contact_tables(glycan, stereo=None, level="monosaccharide", my_path=None):
  """Gets contact tables for a given glycan across all its PDB structures.
  Args:
      glycan (str): IUPAC glycan sequence.
      stereo (str, optional): 'alpha' or 'beta' to select stereochemistry.
      level (str): 'monosaccharide' or 'atom' to determine detail level.
      my_path (str, optional): Custom path to PDB folders.
  Returns:
      list: List of contact tables for each PDB structure.
  """
  if stereo is None:
    stereo = 'beta' if any(glycan.endswith(mono) for mono in {'GlcNAc', 'Glc', 'Xyl'}) else 'alpha'
  dfs, _ = annotation_pipeline(glycan, pdb_file=my_path, threshold=3.5, stereo=stereo)
  if level == "monosaccharide":
    return [make_monosaccharide_contact_table(df, mode='distance', threshold=200) for df in dfs if len(df) > 0]
  else:
    return [make_atom_contact_table(df, mode='distance', threshold=200) for df in dfs if len(df) > 0]


@rescue_glycans
def inter_structure_variability_table(glycan, stereo=None, mode='standard', my_path=None, fresh=False):
  """Creates a table showing stability of atom/monosaccharide positions across different PDB structures of the same glycan.
  Args:
      glycan (str or list): Glycan in IUPAC sequence or list of contact tables.
      stereo (str, optional): 'alpha' or 'beta' to select stereochemistry.
      mode (str): 'standard', 'amplify', or 'weighted' for different calculation methods.
      my_path (str, optional): Custom path to PDB folders.
      fresh (bool): If True, fetches fresh cluster frequencies.
  Returns:
      pd.DataFrame: Variability table showing how much positions vary across structures.
  """
  if isinstance(glycan, str):
    dfs = get_contact_tables(glycan, stereo, my_path=my_path)
  elif isinstance(glycan, list):
    dfs = glycan
  if len(dfs) < 1:
    return pd.DataFrame()
  if stereo is None and isinstance(glycan, str):
    stereo = 'beta' if any(glycan.endswith(mono) for mono in {'GlcNAc', 'Glc', 'Xyl'}) else 'alpha'
  columns = dfs[0].columns
  values_array = np.array([df.values for df in dfs])
  mean_values = np.mean(values_array, axis=0)
  deviations = np.abs(values_array - mean_values)
  if mode == 'weighted':
    weights = np.array(get_all_clusters_frequency(fresh=fresh)[glycan]) / 100
    weights = [1.0]*len(dfs) if len(weights) != len(dfs) else weights
    result = np.average(deviations, weights=weights, axis=0)
  elif mode == 'amplify':
    result = np.sum(deviations, axis=0) ** 2
  else:  # standard mode
    result = np.sum(deviations, axis=0)
  return pd.DataFrame(result, columns=columns, index=columns)


@rescue_glycans
def make_correlation_matrix(glycan, stereo=None, my_path=None):
  """Computes a Pearson correlation matrix between residue positions across structures.
  Args:
      glycan (str or list): Glycan in IUPAC sequence or list of contact tables.
      stereo (str, optional): 'alpha' or 'beta' to select stereochemistry.
      my_path (str, optional): Custom path to PDB folders.
  Returns:
      pd.DataFrame: Correlation matrix showing relationships between residue positions.
  """
  if isinstance(glycan, str):
    dfs = get_contact_tables(glycan, stereo, my_path=my_path)
  elif isinstance(glycan, list):
    dfs = glycan
  if stereo is None and isinstance(glycan, str):
    stereo = 'beta' if any(glycan.endswith(mono) for mono in {'GlcNAc', 'Glc', 'Xyl'}) else 'alpha'
  # Create an empty correlation matrix
  corr_sum = np.zeros((len(dfs[0]), len(dfs[0])))
  # Calculate the correlation matrix based on the distances
  for df in dfs:
    corr_sum += np.corrcoef(df.values, rowvar=False)
  return pd.DataFrame(corr_sum/len(dfs), columns=df.columns, index=df.columns)


@rescue_glycans
def inter_structure_frequency_table(glycan, stereo=None, threshold = 5, my_path=None):
  """Creates a table showing frequency of contacts between residues across structures.
  Args:
      glycan (str or list): Glycan in IUPAC sequence or list of contact tables.
      stereo (str, optional): 'alpha' or 'beta' to select stereochemistry.
      threshold (float): Maximum distance for determining a contact.
      my_path (str, optional): Custom path to PDB folders.
  Returns:
      pd.DataFrame: Table of contact frequencies across structures.
  """
  if isinstance(glycan, str):
    dfs = get_contact_tables(glycan, stereo, my_path=my_path)
  elif isinstance(glycan, list):
    dfs = glycan
  if stereo is None and isinstance(glycan, str):
    stereo = 'beta' if any(glycan.endswith(mono) for mono in {'GlcNAc', 'Glc', 'Xyl'}) else 'alpha'
  # Apply thresholding and create a new list of transformed DataFrames
  binary_arrays = [df.values < threshold for df in dfs]
  # Sum up the transformed DataFrames to create the final DataFrame
  return pd.DataFrame(sum(binary_arrays), columns=dfs[0].columns, index=dfs[0].columns)


def extract_binary_interactions_from_PDB(coordinates_df):
  """Extracts binary interactions between C1/C2 atoms and oxygen atoms from coordinates.
  Args:
      coordinates_df (pd.DataFrame): Coordinate dataframe from extract_3D_coordinates.
  Returns:
      pd.DataFrame or list of pd.DataFrame: DataFrame with columns 'Atom', 'Column', and 'Value' 
      showing interactions. Returns a list of DataFrames if multiple chains are present.
  """
  # Check if multiple chains exist
  unique_chains = coordinates_df['chain_id'].unique()
  if len(unique_chains) > 1:
    results = []
    for chain in unique_chains:
      chain_df = coordinates_df[coordinates_df['chain_id'] == chain]
      chain_result = process_interactions(chain_df)
      if not chain_result.empty:
        results.append(chain_result)
    return results
  else:
    return process_interactions(coordinates_df)


def process_interactions(coordinates_df):
  """Extracts binary interactions between C1/C2 atoms and oxygen atoms from coordinates.
  Args:
      coordinates_df (pd.DataFrame): Coordinate dataframe from extract_3D_coordinates.
  Returns:
      pd.DataFrame: DataFrame with columns 'Atom', 'Column', and 'Value' showing interactions.
  """
  # First check if we only have one monosaccharide
  unique_residues = coordinates_df['residue_number'].nunique()
  c2_pattern = 'NGC|SIA|NGE|4CD|0CU|1CU|1CD'
  carbon_mask = (((~coordinates_df['monosaccharide'].str.contains(c2_pattern, na=False)) & (coordinates_df['atom_name'] == 'C1')) |
                 ((coordinates_df['monosaccharide'].str.contains(c2_pattern, na=False)) & (coordinates_df['atom_name'] == 'C2')))
  oxygen_mask = coordinates_df['atom_name'].isin({'O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O8', 'O9'})
  carbons = coordinates_df[carbon_mask]
  oxygens = coordinates_df[oxygen_mask]
  c_coords = carbons[['x', 'y', 'z']].values
  o_coords = oxygens[['x', 'y', 'z']].values
  c_residues = carbons['residue_number'].values
  o_residues = oxygens['residue_number'].values
  c_labels = [f"{r}_{m}_{a}" for r, m, a in zip(carbons['residue_number'], carbons['monosaccharide'], carbons['atom_name'])]
  o_labels = [f"{r}_{m}_{a}" for r, m, a in zip(oxygens['residue_number'], oxygens['monosaccharide'], oxygens['atom_name'])]
  interactions = []
  if unique_residues == 2:
    roh_oxygens = oxygens[oxygens['monosaccharide'] == 'ROH']
    if not roh_oxygens.empty:
      roh_coord = roh_oxygens[['x', 'y', 'z']].values[0]
      # Find C1 of the monosaccharide and O1 of ROH
      for i, c_label in enumerate(c_labels):
        if carbons.iloc[i]['monosaccharide'] != 'ROH' and carbons.iloc[i]['atom_name'] == 'C1':
          distance = np.abs(roh_coord - c_coords[i]).sum()
          interactions.append({
            'Atom': c_label,
            'Column': f"{roh_oxygens['residue_number'].iloc[0]}_ROH_O1",
            'Value': distance
            })
  else:
    for i, c_label in enumerate(c_labels):
      mask = (o_residues != c_residues[i])
      if np.any(mask):
        relevant_o_coords = o_coords[mask]
        distances = np.abs(relevant_o_coords - c_coords[i]).sum(axis=1)
        if len(distances) > 0:
          min_idx = np.argmin(distances)
          min_idx = np.where(mask)[0][min_idx]
          interactions.append({
            'Atom': c_label,
            'Column': o_labels[min_idx],
            'Value': distances[min_idx]
            })
  df =  pd.DataFrame(interactions)
  if len(df) > 0:
    # Extract source and target monosaccharides
    df['source_mono'] = df['Atom'].str.split('_').str[:2].str.join('_')
    df['target_mono'] = df['Column'].str.split('_').str[:2].str.join('_')
    # Group by monosaccharide pairs and keep minimum distance
    df = df.loc[df.groupby(['source_mono', 'target_mono'])['Value'].idxmin()]
  return df[['Atom', 'Column', 'Value']].reset_index(drop=True) if len(df) > 0 else df


def create_mapping_dict_and_interactions(df, valid_fragments, n_glycan, furanose_end, d_end, is_protein_complex):
  """Creates mapping dictionaries for converting PDB residue names to IUPAC notation.
  Args:
      df (pd.DataFrame): Interaction dataframe from extract_binary_interactions_from_PDB.
      valid_fragments (set): Valid monosaccharide link fragments from glycowork.
      n_glycan (bool): If True, applies N-glycan-specific corrections.
      furanose_end (bool): If True, considers furanose forms for terminal residues.
      d_end (bool): If True, considers D-form for terminal residues.
      is_protein_complex (bool): If True, assumes glycan comes from protein-glycan PDB
  Returns:
      tuple: (mapping_dict, interaction_dict) for PDB to IUPAC conversion.
  """
  special_cases = {
            'Man(a1-4)', '-R', 'GlcNAc(a1-1)', 'GlcNAc(b1-1)', 'GalNAc(a1-1)',
            'GalNAc(b1-1)', 'Glc(a1-1)', 'Glc(b1-1)', 'Rha(a1-1)', 'Rha(b1-1)',
            'Neu5Ac(a2-1)', 'Neu5Ac(b2-1)', 'Man(a1-1)', 'Man(b1-1)', 'Gal(a1-1)',
            'Gal(b1-1)', 'Fuc(a1-1)', 'Fuc(b1-1)', 'Xyl(a1-1)', 'Xyl(b1-1)', 'L-Gul(a1-1)',  'L-Gul(b1-1)',
            'GlcA(a1-1)', 'GlcA(b1-1)', 'GlcNS(a1-1)', 'GlcNS(b1-1)', 'GlcNAc6S(a1-1)',
            'GlcNAc6S(b1-1)', 'GlcNS6S(a1-1)', 'GlcNS6S(b1-1)', 'GlcNS3S6S(a1-1)',
            'GlcNS3S6S(b1-1)', '2-4-diacetimido-2-4-6-trideoxyhexose(a1-1)', 'D-Araf(a1-1)', 'D-Araf(b1-1)',
            'GlcA2S(a1-1)', 'GlcA2S(b1-1)', 'Ara(a1-1)', 'Ara(b1-1)', 'Araf(a1-1)', 'Araf(b1-1)', 'Fru(a2-1)',
            'Fru(b2-1)', 'Fruf(a2-1)', 'Fruf(b2-1)', 'ManNAc(a1-1)', 'ManNAc(b1-1)'
        }

  def d_conversion(mono, trigger, i = 1):
    if mono.startswith(trigger):
      d_version = f"D-{mapped_to_check}"
      if d_version in valid_fragments or (i == 0 and d_end):
        return d_version
    return mono

  mapping_dict = {'1_ROH': '-R'}
  interaction_dict, interaction_dict2 = {}, {}
  wrong_mannose, individual_entities = [], []
  furanose_map = {'Fru': 'Fruf', 'Gal': 'Galf', 'Ara': 'Araf', 'D-Ara': 'D-Araf'}
  for i, row in df.iterrows():
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
    mapped_to_check = d_conversion(mapped_to_check, 'Ara', i=i)
    mono_type = mapped_to_check.split('(')[0]
    if i == 0 and is_protein_complex:
     mapped_to_check2 = f"{map_dict[second_mono_base.split('_')[1]]}1-1)"
     mapped_to_check2 = f"{map_dict[second_mono_base.split('_')[1]].split('(')[0]}"
     if mapped_to_check2 in valid_fragments:
       mapping_dict[second_mono_base] = f"{map_dict[second_mono_base.split('_')[1]]}1-1)"
    if (mapped_to_check not in valid_fragments and (mapped_to_check not in special_cases or furanose_end) and mono_type in furanose_map):
      mapped_to_check = furanose_map[mono_type] + mapped_to_check[len(mono_type):]
      mapped_to_check = d_conversion(mapped_to_check, 'Araf')
    if (mapped_to_check in valid_fragments) or (mapped_to_check in special_cases):
      mapped_to_use =  'Man(b1-4)' if (mapped_to_check == 'Man(a1-4)' and n_glycan) else mapped_to_check
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


def extract_binary_glycontact_interactions(interaction_dict, mapping_dict):
  """Transforms PDB-based interactions into IUPAC binary interactions.
  Args:
      interaction_dict (dict): Dict of interactions from create_mapping_dict_and_interactions.
      mapping_dict (dict): Mapping dict from create_mapping_dict_and_interactions.
  Returns:
      list: List of binary interaction tuples in IUPAC format.
  """
  result = []
  for k, v in interaction_dict.items():
    new_k = k.split('_')[1].replace('(', '').replace(')', '') if '(' in k else mapping_dict[k].split('(')[0]
    new_v = v[0].split('_')[1].replace('(', '').replace(')', '') if '(' in v[0] else mapping_dict[v[0]].split('(')[0]
    result.append((new_k, new_v))
  return result


def extract_binary_glycowork_interactions(graph_output):
  """Extracts binary interactions from glycowork graph output.
  Args:
      graph_output (tuple): Output from glycan_to_graph function.
  Returns:
      list: List of binary interaction pairs.
  """
  mask_dic, adj_matrix = graph_output
  n = len(mask_dic)
  return [(mask_dic[k], mask_dic[j]) for k in range(n) for j in range(k + 1, n) if adj_matrix[k, j] == 1]


def glycowork_vs_glycontact_interactions(glycowork_interactions, glycontact_interactions) :
  """Compares binary interactions from glycowork and glycontact for validation.
  Args:
      glycowork_interactions (list): Interactions from glycowork.
      glycontact_interactions (list): Interactions from glycontact.
  Returns:
      bool: True if interactions are consistent (excluding special cases).
  """
  ignore_pairs = {
        ('GlcNAc', 'a1-1'), ('a1-1', '-R'), ('a2-1', '-R'), ('b2-1', '-R'),
        ('GlcNAc', 'b1-1'), ('b1-1', '-R'), ('GalNAc', 'a1-1'), ('GalNAc', 'b1-1'),
        ('Glc', 'a1-1'), ('Glc', 'b1-1'), ('Rha', 'b1-1'), ('Rha', 'a1-1'),
        ('Neu5Ac', 'b2-1'), ('Neu5Ac', 'a2-1'), ('Man', 'b1-1'), ('Man', 'a1-1'),
        ('Gal', 'b1-1'), ('Gal', 'a1-1'), ('Fuc', 'b1-1'), ('Fuc', 'a1-1'),
        ('Xyl', 'b1-1'), ('Xyl', 'a1-1'), ('GlcA', 'a1-1'), ('GlcA', 'b1-1'),
        ('GlcNS', 'a1-1'), ('GlcNS', 'b1-1'), ('GlcNAc6S', 'a1-1'),
        ('GlcNAc6S', 'b1-1'), ('GlcNS6S', 'a1-1'), ('GlcNS6S', 'b1-1'),
        ('GlcNS3S6S', 'a1-1'), ('GlcNS3S6S', 'b1-1'), ('L-Gul', 'a1-1'), ('L-Gul', 'b1-1'),
        ('2-4-diacetimido-2-4-6-trideoxyhexose', 'a1-1'), ('GlcA2S', 'a1-1'), ('D-Araf', 'a1-1'), ('D-Araf', 'b1-1'),
        ('GlcA2S', 'b1-1'), ('Ara', 'a1-1'), ('Ara', 'b1-1'), ('Araf', 'a1-1'), ('Araf', 'b1-1'), ('Fru', 'a2-1'),
        ('Fru', 'b2-1'), ('ManNAc', 'a1-1'), ('ManNAc', 'b1-1'), ('Fruf', 'a2-1'), ('Fruf', 'b2-1')
    }
  differences = set(glycontact_interactions) ^ set(glycowork_interactions)
  filtered_differences = [pair for pair in differences if pair not in ignore_pairs]
  return (not filtered_differences and len(glycontact_interactions) >= len(glycowork_interactions))


def check_reconstructed_interactions(interaction_dict) :
  """Verifies if the reconstructed glycan is connected as a single component.
  Args:
      interaction_dict (dict): Dictionary of interactions.
  Returns:
      bool: True if glycan is correctly reconstructed as a single connected component.
  """
  G = nx.Graph()
  # Add nodes and edges from dictionary interactions
  G.add_edges_from((node, neighbor) for node, neighbors in interaction_dict.items() for neighbor in neighbors)
  return nx.is_connected(G)


def annotate_pdb_data(pdb_dataframe, mapping_dict) :
  """Annotates PDB data with IUPAC nomenclature using the mapping dictionary.
  Args:
      pdb_dataframe (pd.DataFrame): DataFrame with PDB coordinates.
      mapping_dict (dict): Mapping from PDB to IUPAC nomenclature.
  Returns:
      pd.DataFrame: Annotated dataframe with IUPAC column.
  """
  m_dict = copy.deepcopy(mapping_dict)
  pdb_dataframe = pdb_dataframe.copy()
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
  """Corrects monosaccharide assignments in the dataframe based on atom counts.
  Args:
      df (pd.DataFrame): Annotated dataframe from annotate_pdb_data.
  Returns:
      pd.DataFrame: Corrected dataframe with fixed monosaccharide assignments.
  """
  c_counts = df[df['element'] == 'C'].groupby('residue_number').size()
  high_carbon_residues = c_counts[c_counts >= 7].index
  # Create masks for GLC and BGC replacements
  glc_mask = (df['monosaccharide'] == 'GLC') & df['residue_number'].isin(high_carbon_residues)
  bgc_mask = (df['monosaccharide'] == 'BGC') & df['residue_number'].isin(high_carbon_residues)
  # Apply replacements
  df.loc[glc_mask, 'monosaccharide'] = 'NGA'
  df.loc[bgc_mask, 'monosaccharide'] = 'A2G'
  return df


def process_interactions_result(res, threshold, valid_fragments, n_glycan, furanose_end, d_end, is_protein_complex, glycan, df):
  """Process a single interaction result and return the annotation if valid."""
  if isinstance(threshold, float) or isinstance(threshold, int):
    res = res[res.Value < threshold].reset_index(drop=True)
  else:
    for thresh in sorted(threshold):
      res = res[res.Value < thresh].reset_index(drop=True)
      if len(res) > 0:
        break
  mapping_dict, interaction_dict = create_mapping_dict_and_interactions(res, valid_fragments,
                                                                      n_glycan, furanose_end, d_end, is_protein_complex)
  # Validate against glycowork
  glycowork_interactions = extract_binary_glycowork_interactions(glycan_to_graph(glycan))
  glycontact_interactions = extract_binary_glycontact_interactions(interaction_dict, mapping_dict)
  glycontact_interactions = [(x + 'f' if any(f'{x}f(' in s for s in valid_fragments) and not any(f'{x}(' in s for s in valid_fragments) else x,
                            y + 'f' if any(f'{y}f(' in s for s in valid_fragments) and not any(f'{y}(' in s for s in valid_fragments) else y)
                           for x, y in glycontact_interactions]
  if (glycowork_vs_glycontact_interactions(glycowork_interactions, glycontact_interactions) and
      check_reconstructed_interactions(interaction_dict)):
    return annotate_pdb_data(df, mapping_dict), interaction_dict
  return pd.DataFrame(), {}


def get_annotation(glycan, pdb_file, threshold=3.5):
  """Annotates a PDB file with IUPAC nomenclature for a given glycan.
  Args:
      glycan (str): IUPAC glycan sequence.
      pdb_file (str): Path to PDB file.
      threshold (float or list): Distance threshold for interactions.
  Returns:
      tuple: (annotated_dataframe, interaction_dict) or (empty_dataframe, {}) if validation fails.
  """
  MODIFIED_MONO = {
        "GlcNAc6S", "GalNAc4S", "IdoA2S", "GlcA3S", "GlcA2S", "Neu5Ac9Ac",
        "Man3Me", "Neu5Ac9Me", "Neu5Gc9Me", "GlcA4Me", "Gal6S", "GlcNAc6PCho",
        "GlcNS6S", "GlcNS3S6S", "Fuc2Me3Me4Me", "Rha2Me3Me", "GalA2S"
    }
  CUSTOM_PDB = {
        "NAG6SO3": "GlcNAc6S", "NDG6SO3": "GlcNAc6S", "NDG3SO3": "GlcNAc3S6S",
        "NGA4SO3": "GalNAc4S", "IDR2SO3": "IdoA2S", "BDP3SO3": "GlcA3S", "TOA2SO3": "GalA2S",
        "BDP2SO3": "GlcA2S", "SIA9ACX": "Neu5Ac9Ac", "MAN3MEX": "Man3Me",
        "SIA9MEX": "Neu5Ac9Me", "NGC9MEX": "Neu5Gc9Me", "BDP4MEX": "GlcA4Me",
        "GAL6SO3": "Gal6S", "NAG6PCX": "GlcNAc6PCho", "UYS6SO3": "GlcNS6S",
        "4YS6SO3": "GlcNS6S", "6YS6SO3": "GlcNS6S", "GCU2SO3": "GlcA2S",
        'VYS3SO3': 'GlcNS3S6S', 'VYS6SO3': 'GlcNS3S6S', 'FUC2MEX': 'Fuc2Me', 'FUC3MEX': 'Fuc3Me', 'FUC4MEX': 'Fuc4Me',
        "QYS3SO3": "GlcNS3S6S", "QYS6SO3": "GlcNS3S6S", "RAM2MEX": "Rha2Me", "RAM3MEX": "Rha3Me"
    }
  n_glycan = 'Man(b1-4)GlcNAc(b1-4)' in glycan or 'Man(b1-4)[Fuc(a1-3)]GlcNAc' in glycan
  furanose_end = glycan.endswith('f')
  d_end = glycan[glycan.rfind('-')-1] == "D"
  df = correct_dataframe(extract_3D_coordinates(pdb_file))
  is_protein_complex = df['record_name'].iloc[0] == 'HETATM' 
  # Handle multiple instances of a single monosaccharide in protein complexes
  if is_protein_complex and ')' not in glycan:
    results = []
    for res_num in df['residue_number'].unique():
      instance_df = df[df['residue_number'] == res_num].copy()
      mono_type = instance_df['monosaccharide'].iloc[0]
      # Create instance-specific mapping dictionary using the global map_dict
      instance_mapping = {f"{res_num}_{mono_type}": map_dict.get(mono_type, mono_type)}
      results.append((annotate_pdb_data(instance_df, instance_mapping), {}))
    return results
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
    grouped_mods = {}
    for residue_num, modification in to_modify_dict.items():
      target_residue = modification[1] if isinstance(modification, list) else residue_num
      if target_residue not in grouped_mods:
        grouped_mods[target_residue] = []
      grouped_mods[target_residue].append(modification)
    for target_residue, mods in grouped_mods.items():
      if len(mods) > 1:
        # Multiple modifications for same residue - combine them
        base = mods[0][0] if isinstance(mods[0], list) else mods[0]  # Get base from first mod
        base = base[:3]  # Extract just the base part (e.g., 'FUC')
        mod_parts = [mod[0][3:] if isinstance(mod, list) else mod[3:] for mod in mods]  # Get just the modification part (e.g., '2MEX')
        combined_mod = base + ''.join(sorted(mod_parts))  # Combine with sorted modifications
        df.loc[df['residue_number'] == int(target_residue), 'monosaccharide'] = combined_mod
      else:
        # Single modification - use original logic
        mod = mods[0]
        if isinstance(mod, str):
          df.loc[df['residue_number'] == target_residue, 'monosaccharide'] = mod
        else:
          monosaccharide, new_residue = mod
          mask = df['residue_number'] == target_residue
          df.loc[mask, 'monosaccharide'] = monosaccharide
          df.loc[mask, 'residue_number'] = int(new_residue)
    df = df.sort_values('residue_number')
  # Extract and validate linkages
  valid_fragments = {x.split(')')[0] + ')' for x in link_find(glycan)} | ({min_process_glycans([glycan])[0][-1]} if is_protein_complex else set())
  res = extract_binary_interactions_from_PDB(df)
  # Handle case where extract_binary_interactions_from_PDB returns a list of DataFrames (multiple chains)
  if isinstance(res, list):
    chain_ids = df.chain_id.unique()
    expected_residue_count = glycan.count('(') + 1
    # Try each chain's result and use the first one that successfully validates
    for i, chain_res in enumerate(res):
      if not chain_res.empty:
        max_residue = max(
          max([int(atom.split('_')[0]) for atom in chain_res['Atom']]),
          max([int(col.split('_')[0]) for col in chain_res['Column']])
        )
        if max_residue != expected_residue_count:
          continue
      result = process_interactions_result(chain_res, threshold, valid_fragments, 
                                         n_glycan, furanose_end, d_end, is_protein_complex, glycan, df[df.chain_id==chain_ids[i]])
      if result[0] is not None:  # If validation succeeded
        return result
    # If no chain validates successfully
    return pd.DataFrame(), {}
  else:
    # Original single-chain behavior
    return process_interactions_result(res, threshold, valid_fragments, 
                                     n_glycan, furanose_end, d_end, is_protein_complex, glycan, df)


@rescue_glycans
def annotation_pipeline(glycan, pdb_file = None, threshold=3.5, stereo = None) :
  """Combines all annotation steps to convert PDB files to IUPAC annotations.
  Args:
      glycan (str): IUPAC glycan sequence.
      pdb_file (str or list, optional): Path(s) to PDB file(s).
      threshold (float): Distance threshold for interactions.
      stereo (str, optional): 'alpha' or 'beta' stereochemistry.
  Returns:
      tuple: (dataframes_list, interaction_dicts_list) for all processed PDBs.
  """
  if stereo is None:
    stereo = 'beta' if any(glycan.endswith(mono) for mono in {'GlcNAc', 'Glc', 'Xyl'}) else 'alpha'
  if pdb_file is None:
    pdb_file = os.listdir(global_path / glycan)
    pdb_file = [global_path / glycan / pdb for pdb in pdb_file if stereo in pdb]
  if isinstance(pdb_file, str):
    pdb_file = [pdb_file]
  dfs, int_dicts = zip(*[get_annotation(glycan, pdb, threshold=threshold) for pdb in pdb_file])
  return dfs, int_dicts


def get_example_pdb(glycan, stereo=None, rng=None):
  """Gets a random example PDB file for a given glycan.
  Args:
      glycan (str): IUPAC glycan sequence.
      stereo (str, optional): 'alpha' or 'beta' stereochemistry.
      rng (Random, optional): Random number generator instance.
  Returns:
      Path: Path to a randomly selected PDB file.
  """
  if rng is None:
    rng = Random(42)
  if stereo is None:
    stereo = 'beta' if any(glycan.endswith(mono) for mono in {'GlcNAc', 'Glc', 'Xyl'}) else 'alpha'
  pdb_file = os.listdir(global_path / glycan)
  return global_path / glycan / rng.choice([pdb for pdb in pdb_file if stereo in pdb])


def monosaccharide_preference_structure(df, monosaccharide, threshold, mode='default'):
  """Finds preferred partners for a given monosaccharide.
  Args:
      df (pd.DataFrame): Monosaccharide distance table.
      monosaccharide (str): Target monosaccharide type.
      threshold (float): Minimum distance to exclude covalent bonds.
      mode (str): 'default', 'monolink', or 'monosaccharide' for different reporting formats.
  Returns:
      dict: Dictionary of preferred partners for the target monosaccharide.
  """
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


@rescue_glycans
def multi_glycan_monosaccharide_preference_structure(glycan, monosaccharide, stereo=None, threshold=3.5, mode='default'):
  """Visualizes monosaccharide partner preferences across multiple structures.
  Args:
      glycan (str): IUPAC glycan sequence.
      monosaccharide (str): Target monosaccharide type.
      stereo (str, optional): 'alpha' or 'beta' stereochemistry.
      threshold (float): Minimum distance to exclude covalent bonds.
      mode (str): 'default', 'monolink', or 'monosaccharide' for different reporting formats.
  Returns:
      None: Displays a bar plot of partner frequencies.
  """
  if stereo is None:
    stereo = 'beta' if any(glycan.endswith(mono) for mono in {'GlcNAc', 'Glc', 'Xyl'}) else 'alpha'
  mono_tables = get_contact_tables(glycan, stereo=stereo)
  dict_list = [monosaccharide_preference_structure(dist, monosaccharide, threshold, mode) for dist in mono_tables]
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
  """Extracts frequency data for all glycan clusters from GlycoShape.
  Args:
      fresh (bool): If True, fetches fresh data from GlycoShape.
  Returns:
      dict: Dictionary mapping IUPAC sequences to cluster frequency lists.
  """
  data = {}
  if fresh:
    response = requests.get("https://glycoshape.org/database/GLYCOSHAPE.json")
    if response.status_code == 200:
      data = response.json()
  else:
    data = glycoshape_mirror
  return {value["iupac"]: [100.0] if list(value["clusters"].values()) == ['None'] else list(value["clusters"].values()) for key, value in data.items()}


def glycan_cluster_pattern(threshold = 70, mute = False, fresh=False) :
  """Categorizes glycans based on their cluster distribution patterns.
  Args:
      threshold (float): Percentage threshold for major cluster classification.
      mute (bool): If True, suppresses print output.
      fresh (bool): If True, fetches fresh data from GlycoShape.
  Returns:
      tuple: (major_clusters_list, minor_clusters_list) sorted by cluster pattern.
  """
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


def get_sasa_table(glycan, stereo = None, my_path=None, fresh=False):
  """Calculates solvent accessible surface area (SASA) for each monosaccharide.
  Args:
      glycan (str): IUPAC glycan sequence.
      stereo (str, optional): 'alpha' or 'beta' stereochemistry.
      my_path (str, optional): Custom path to PDB folders.
      fresh (bool): If True, fetches fresh cluster frequencies.
  Returns:
      pd.DataFrame: Table with SASA values and statistics for each monosaccharide.
  """
  is_single_pdb = my_path is not None and "." in my_path
  if stereo is None:
    stereo = 'beta' if any(glycan.endswith(mono) for mono in {'GlcNAc', 'Glc', 'Xyl'}) else 'alpha'
  if my_path is None:
    pdb_dir = global_path / glycan
    pdb_files = sorted(pdb_dir / pdb for pdb in os.listdir(pdb_dir) if stereo in pdb)
  else:
    pdb_files = sorted(str(p) for p in Path(f"{my_path}{glycan}").glob(f"*{stereo}*")) if not is_single_pdb else [my_path]
  df = pd.DataFrame()
  for pdb_file in pdb_files:
    df, _ = get_annotation(glycan, pdb_file)
    if len(df) > 0:
      break
  if len(df) < 1:
    return pd.DataFrame(columns=['Monosaccharide_id', 'Monosaccharide', 'SASA', 'Standard Deviation', 'Coefficient of Variation'])
  if not is_single_pdb:
    weights = np.array(get_all_clusters_frequency(fresh=fresh)[glycan]) / 100
    weights = np.tile(weights, 2) if len(weights) != len(pdb_files) else weights
    weights = [1.0]*len(pdb_files) if len(weights) != len(pdb_files) else weights
  else:
    weights = [1.0]
  residue_modifications = df.set_index('residue_number')['IUPAC'].to_dict()
  # Process each PDB file
  sasa_values = {}
  for pdb_file in pdb_files:
    structure = md.load(pdb_file)
    if is_single_pdb:
        glycan_residues = set(df['residue_number'])
        atom_indices = [atom.index for atom in structure.topology.atoms if atom.residue.resSeq in glycan_residues or 
                     (atom.residue.name in NON_MONO and atom.residue.resSeq in [r.resSeq for r in structure.topology.residues 
                       if r.resSeq in glycan_residues])]
        structure = structure.atom_slice(atom_indices)
    sasa = md.shrake_rupley(structure, mode='atom')
    # Group SASA by residue
    mono_sasa, modification_to_parent = {}, {}
    # First pass: identify modification groups and their parent residues
    parent_resSeq = None
    for res in structure.topology.residues:
      if res.name == 'PCX':
        # Special case: find the residue with PCho in residue_modifications
        for resSeq, resName in residue_modifications.items():
          if 'PCho' in resName:
            modification_to_parent[res.resSeq] = resSeq
            break
      elif res.name not in NON_MONO:
        parent_resSeq = res.resSeq
      else:
        # If this is a modification, assign it to the last seen non-modification residue
        if parent_resSeq is not None:
          modification_to_parent[res.resSeq] = parent_resSeq
    # Second pass: calculate SASA value
    for atom in structure.topology.atoms:
      res = atom.residue
      res_seq = res.resSeq
      # If this is a modification residue, get its parent's resSeq
      if res.name in NON_MONO and res_seq in modification_to_parent:
        parent_resSeq = modification_to_parent[res_seq]
        if parent_resSeq not in mono_sasa:
          mono_sasa[parent_resSeq] = {
                'resName': residue_modifications.get(parent_resSeq, 'NAG'),  # Use IUPAC name if available
                'sasa': 0
                }
        mono_sasa[parent_resSeq]['sasa'] += sasa[0][atom.index]
        continue
      if res.name in NON_MONO:
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
    'SASA': [], 'Standard Deviation': [],
    'Coefficient of Variation': []
    }
  for resSeq, data in stats.items():
    values = np.array(data['values'])
    resName = data['resName']
    #if is_single_pdb and '(' not in resName:
     # continue
    df_data['Monosaccharide_id'].append(resSeq)
    df_data['Monosaccharide'].append(resName)
    if is_single_pdb:
      df_data['SASA'].append(values[0])
      df_data['Standard Deviation'].append(float('nan'))
      df_data['Coefficient of Variation'].append(float('nan'))
    else:
      mean = np.mean(values)
      df_data['SASA'].append(np.average(values, weights=weights))
      std = np.std(values)
      df_data['Standard Deviation'].append(std)
      df_data['Coefficient of Variation'].append(std / mean if mean != 0 else 0)
  return pd.DataFrame(df_data)


def convert_glycan_to_class(glycan):
  """Converts monosaccharides in a glycan string to abstract classes.
  Args:
      glycan (str): IUPAC glycan sequence.
  Returns:
      str: Modified glycan string with abstracted monosaccharide classes.
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
  MONO_MAP = {mono: class_name for class_name, monos in MONO_CLASSES.items() for mono in monos}
  CLASS_NAMES = {'Hex': 'X', 'dHex': 'dX', 'HexA': 'XA', 'HexN': 'XN', 'HexNAc': 'XNAc', 'Pen': 'Pen', 'Sia': 'Sia'}
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
  """Groups glycans by their topological silhouette/branching pattern.
  Args:
      glycan_list (list): List of IUPAC glycan sequences.
      mode (str): 'X' for simple abstraction or 'class' for detailed classes.
  Returns:
      pd.DataFrame: DataFrame of glycans annotated with silhouette and group.
  """
  silhouettes, topo_groups = {}, {}
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


def compute_merge_SASA_flexibility(glycan, mode='weighted', stereo=None, my_path=None) :
  """Merges SASA and flexibility data for a glycan structure.
  Args:
      glycan (str): IUPAC glycan sequence.
      mode (str, optional): 'standard', 'amplify', or 'weighted' for flexibility calculation.
      stereo (str, optional): 'alpha' or 'beta' stereochemistry.
      my_path (str, optional): Custom path to PDB folders.
  Returns:
      pd.DataFrame: Combined table with SASA and flexibility (as RMSF) metrics.
  """
  if stereo is None:
    stereo = 'beta' if any(glycan.endswith(mono) for mono in {'GlcNAc', 'Glc', 'Xyl'}) else 'alpha'
  sasa = get_sasa_table(glycan, stereo=stereo, my_path=my_path)
  if my_path is not None and "." in my_path:
    df, _ = get_annotation(glycan, my_path)
    flexibility = df.groupby('residue_number')['temperature_factor'].mean()
    flexibility_rmsf = np.sqrt(3 * flexibility / (8 * np.pi**2))
    monosaccharides = df.drop_duplicates('residue_number').set_index('residue_number')['IUPAC']
    flex_df = pd.DataFrame({'Monosaccharide_id': df.residue_number.unique(), 'Monosaccharide': monosaccharides, 'flexibility': flexibility_rmsf}).reset_index(drop=True)
  else:
    flex = (inter_structure_variability_table(glycan, stereo=stereo, mode=mode, my_path=my_path)).mean()
    conversion_factor = np.sqrt(np.pi/2)  # converts mean absolute deviation to standard deviation
    flex_rmsf = {monosac: value * conversion_factor for monosac, value in flex.items()}
    flex_df = pd.DataFrame(sorted(flex_rmsf.items(), key=lambda x: x[1]), columns=['Monosaccharide_id_Monosaccharide', 'flexibility'])
    flex_df['Monosaccharide_id'] = flex_df['Monosaccharide_id_Monosaccharide'].str.split('_').str[0].astype(int)
  if sasa.empty:
    return flex_df
  return pd.merge(sasa, flex_df[['Monosaccharide_id', 'flexibility']], on='Monosaccharide_id', how='left')


def map_data_to_graph(computed_df, interaction_dict, ring_conf_df=None, torsion_df=None) :
  """Creates a NetworkX graph with node-level structural data.
  Args:
      computed_df (pd.DataFrame): DataFrame with computed monosaccharide properties.
      interaction_dict (dict): Dictionary of glycosidic linkages.
      ring_conf_df (pd.DataFrame, optional): Ring conformation data.
      torsion_df (pd.DataFrame, optional): Torsion angle data.
  Returns:
      nx.Graph: Graph with nodes/edges representing glycan structure and properties.
  """
  edges = {(int(k.split('_')[0]), int(v.split('_')[0])) for k, values in interaction_dict.items() for v in values if k.split('_')[0] != v.split('_')[0]}
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
      res_nums = [match.group() for match in re.finditer(r'(\d+)(?=_)', row['linkage'])]  # Gets ["5", "3"] from "5_FUC-3_GAL"
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
    for col in ['SASA']:
      if col in row:
        attrs[col] = row[col]
    # Add flexibility if available
    if 'flexibility' in row:
      attrs['flexibility'] = row['flexibility']
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
  """Processes a graph by removing odd-indexed nodes and concatenating labels.
  Args:
      graph (nx.Graph): NetworkX graph object.
  Returns:
      nx.Graph: Modified graph with simplified structure.
  """
  graph = graph.copy()
  nodes_to_remove = []  # List to store nodes that need to be removed
  # Iterate through nodes in sorted order to ensure proper handling
  for node in sorted(graph.nodes):
    if node % 2 == 1:
      # When removing a node, look for who points TO it and where it points TO
      predecessors = list(graph.predecessors(node))
      successors = list(graph.successors(node))
      # Connect each node that pointed to this one to each node this one pointed to
      for pred in predecessors:
        for succ in successors:
          graph.add_edge(pred, succ)
      # Handle label concatenation
      predecessor = node - 1
      if predecessor in graph.nodes:
        pred_label = graph.nodes[predecessor].get("string_labels", "")
        current_label = graph.nodes[node].get("string_labels", "")
        graph.nodes[predecessor]["string_labels"] = f"{pred_label}({current_label})"
      nodes_to_remove.append(node)
  # Remove the odd-indexed nodes after processing
  graph.remove_nodes_from(nodes_to_remove)
  return graph


def trim_gcontact(G_contact):
  """Removes node 1 (-R terminal) from glycontact graph and connects its neighbors.
  Args:
      G_contact (nx.Graph): Glycontact graph.
  Returns:
      None: Modifies graph in-place.
  """
  # Remove node 1 which corresponds to -R, absent from G_work
  if 1 in G_contact and G_contact.nodes[1].get("Monosaccharide") == "-R":
    neighbors = list(G_contact.neighbors(1))  # Get the neighbors of node 1
    if len(neighbors) > 1:  # If node 1 has more than one neighbor
      for i in range(len(neighbors)):
        for j in range(i + 1, len(neighbors)):
          G_contact.add_edge(neighbors[i], neighbors[j])  # Add edge between neighbors
    G_contact.remove_node(1)  # Remove node 1


def compare_graphs_with_attributes(G_contact, G_work):
  """Performs attribute-aware isomorphism check between two glycan graphs.
  Args:
      G_contact (nx.Graph): Glycontact graph.
      G_work (nx.Graph): Glycowork graph.
  Returns:
      dict: Mapping between node indices or empty dict if not isomorphic.
  """
  # Define a custom node matcher
  def node_match(node_attrs1, node_attrs2):
    # Ensure 'string_labels' in G is part of 'Monosaccharide' in G2
    return (
        'string_labels' in node_attrs1 and 'Monosaccharide' in node_attrs2
        and node_attrs1['string_labels'] in node_attrs2['Monosaccharide']
        )
  # Create an isomorphism matcher with the custom node matcher
  matcher = nx.isomorphism.GraphMatcher(G_work.to_undirected(), G_contact, node_match=node_match)
  mapping_dict = {} # format= gcontact_index: gwork_index
  if matcher.is_isomorphic():  # Check if the graphs are isomorphic
    # Extract the mapping of nodes
    mapping = matcher.mapping
    for node_g, node_g2 in mapping.items():
      mapping_dict[node_g2] = node_g
  else:
    print("The graphs are not isomorphic with the given attribute constraints.")
  return mapping_dict


def create_glycontact_annotated_graph(glycan: str, mapping_dict, g_contact, libr=None) -> nx.Graph:
  """Creates a glycowork graph annotated with glycontact structural data.
  Args:
      glycan (str): IUPAC glycan sequence.
      mapping_dict (dict): Node mapping from compare_graphs_with_attributes.
      g_contact (nx.Graph): Glycontact graph with structural attributes.
      libr (dict, optional): Custom library for glycan_to_nxGraph.
  Returns:
      nx.Graph: Annotated glycowork graph with combined information.
  """
  glycowork_graph = glycan_to_nxGraph(glycan, libr=libr)
  original_labels = {node: data.get('labels', None) for node, data in glycowork_graph.nodes(data=True)}
  node_attributes = {node: g_contact.nodes[node] for node in g_contact.nodes}
  # Map attributes to the glycowork graph nodes
  flex_attribute_mapping = {
      mapping_dict[gcontact_node]: attributes
      for gcontact_node, attributes in node_attributes.items()
      if gcontact_node in mapping_dict
      }
  # Assign the mapped attributes to the glycowork graph
  nx.set_node_attributes(glycowork_graph, flex_attribute_mapping)
  for node, label in original_labels.items():
    if label is not None:
      glycowork_graph.nodes[node]['labels'] = label
  # Map torsion angles to linkage nodes
  edge_attributes = nx.get_edge_attributes(g_contact, 'phi_angle')
  for (u, v), phi in edge_attributes.items():
    # Find the linkage node between these monosaccharides in glycowork_graph
    u_mapped = mapping_dict[u]
    v_mapped = mapping_dict[v]
    # Find the node that represents the linkage between u_mapped and v_mapped
    linkage_node = min(u_mapped, v_mapped) + 1
    glycowork_graph.nodes[linkage_node].update({
        'phi_angle': g_contact[u][v]['phi_angle'],
        'psi_angle': g_contact[u][v]['psi_angle']
        })
  return glycowork_graph


def get_structure_graph(glycan, stereo=None, libr=None, my_path=None):
  """Creates a complete annotated structure graph for a glycan.
  Args:
      glycan (str): IUPAC glycan sequence.
      stereo (str, optional): 'alpha' or 'beta' stereochemistry.
      libr (dict, optional): Custom library for glycan_to_nxGraph.
      my_path (str, optional): Path to a specific PDB.
  Returns:
      nx.Graph: Fully annotated structure graph with all available properties.
  """
  glycan = canonicalize_iupac(glycan)
  if stereo is None:
    stereo = 'beta' if any(glycan.endswith(mono) for mono in {'GlcNAc', 'Glc', 'Xyl'}) else 'alpha'
  merged = compute_merge_SASA_flexibility(glycan, mode='weighted', stereo=stereo, my_path=my_path)
  example = my_path if my_path is not None else get_example_pdb(glycan, stereo=stereo)
  res, datadict = get_annotation(glycan, example, threshold=3.5)
  ring_conf = get_ring_conformations(res)
  torsion_angles = get_glycosidic_torsions(res, datadict)
  G_contact = map_data_to_graph(merged, datadict, ring_conf_df=ring_conf, torsion_df=torsion_angles)
  G_work = glycan_to_nxGraph(glycan)
  G_work = remove_and_concatenate_labels(G_work)
  trim_gcontact(G_contact)
  m_dict = compare_graphs_with_attributes(G_contact, G_work)
  return create_glycontact_annotated_graph(glycan, mapping_dict=m_dict, g_contact=G_contact, libr=libr)


def check_graph_content(G) :
  """Prints node and edge information from a graph for inspection.
  Args:
      G (nx.Graph): NetworkX graph object.
  Returns:
      None: Prints information to console.
  """
  print("Graph Nodes and Their Attributes:")
  for node, attrs in G.nodes(data=True):
    print(f"Node {node}: {attrs}")
  print("\nGraph Edges:")
  for edge in G.edges():
    print(edge)


def extract_glycan_coords(pdb_filepath, residue_ids=None, main_chain_only=False):
  """Extracts coordinates of glycan residues from a PDB file.
  Args:
      pdb_filepath (str): Path to PDB file.
      residue_ids (list, optional): List of residue numbers to extract.
      main_chain_only (bool): If True, extracts only main chain atoms.
  Returns:
      tuple: (coordinates_array, atom_labels).
  """
  df = extract_3D_coordinates(pdb_filepath)
  if residue_ids:
    df = df[df['residue_number'].isin(residue_ids)]
  # Get common atoms present in most glycans
  if main_chain_only:
    common_atoms = ['C1', 'C2', 'C3', 'C4', 'C5', 'O5']
    df = df[df['atom_name'].isin(common_atoms)]
  else:
    df = df[~df['atom_name'].str.startswith('H')]
  coords, atom_labels = [], []
  for _, row in df.iterrows():
    coords.append([row['x'], row['y'], row['z']])
    atom_labels.append(f"{row['residue_number']}_{row['monosaccharide']}_{row['atom_name']}")
  return np.array(coords), atom_labels


def align_point_sets(mobile_coords, ref_coords):
  """Find optimal rigid transformation to align two point sets.
  Args:
    mobile_coords (np.ndarray): Nx3 array of coordinates to transform
    ref_coords (np.ndarray): Mx3 array of reference coordinates
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


def superimpose_glycans(ref_glycan, mobile_glycan, ref_residues=None, mobile_residues=None, main_chain_only=False):
  """Superimpose two glycan structures and calculate RMSD.
  Args:
    ref_glycan (str): Reference glycan or PDB path.
    mobile_glycan (str): Mobile glycan or PDB path to superimpose.
    ref_residues (list, optional): Residue numbers for reference glycan.
    mobile_residues (list, optional): Residue numbers for mobile glycan.
    main_chain_only (bool): If True, uses only main chain atoms.
  Returns:
    Dict containing:
        - ref_coords: Original coordinates of reference
        - transformed_coords: Aligned mobile coordinates
        - rmsd: Root mean square deviation
        - ref_labels: Atom labels from reference structure
        - mobile_labels: Atom labels from mobile structure
        - ref_conformer: PDB path of reference conformer
        - mobile_conformer: PDB path of mobile conformer
  """
  if '.' not in ref_glycan+mobile_glycan:
    ref_conformers = list((global_path / canonicalize_iupac(ref_glycan)).glob('*.pdb'))
    mobile_conformers = list((global_path / canonicalize_iupac(mobile_glycan)).glob('*.pdb'))
  else:
    ref_conformers = [ref_glycan]
    mobile_conformers = [mobile_glycan]
  best_rmsd = float('inf')
  best_result = None
  # Iterate over all possible pairs of conformers
  for ref_pdb in ref_conformers:  # Extract coordinates for reference conformer
    ref_coords, ref_labels = extract_glycan_coords(ref_pdb, ref_residues, main_chain_only)
    for mobile_pdb in mobile_conformers:  # Extract coordinates for mobile conformer
      mobile_coords, mobile_labels = extract_glycan_coords(mobile_pdb, mobile_residues, main_chain_only)
      transformed_coords, rmsd = align_point_sets(mobile_coords, ref_coords)
      if rmsd < best_rmsd:
        best_rmsd = rmsd
        best_result = {
            'ref_coords': ref_coords,
            'transformed_coords': transformed_coords,
            'rmsd': rmsd,
            'ref_labels': ref_labels,
            'mobile_labels': mobile_labels,
            'ref_conformer': ref_pdb,
            'mobile_conformer': mobile_pdb
            }
  return best_result


def _process_single_glycan(args):
  glycan, query_coords, rmsd_cutoff = args
  best_rmsd = float('inf')
  best_structure = None
  pdb_files = list((global_path / glycan).glob('*.pdb'))
  for pdb_file in pdb_files:
    try:
      coords, _ = extract_glycan_coords(pdb_file)
      if abs(len(coords) - len(query_coords)) <= 5:
        transformed, rmsd = align_point_sets(coords, query_coords)
        if rmsd < best_rmsd:
          best_rmsd = rmsd
          best_structure = pdb_file
    except Exception:
      continue
  return glycan, best_rmsd, best_structure


def get_similar_glycans(query_glycan, pdb_path=None, glycan_database=None, rmsd_cutoff=2.0):
  """Search for structurally similar glycans by comparing against all available
  conformers/structures and keeping the best match for each glycan.
  Args:
    query_glycan (str): PDB file or coordinates of query structure
    pdb_path (str, optional): Optional specific path to query PDB file
    glycan_database (list, optional): List of candidate glycan structures
    rmsd_cutoff (float): Maximum RMSD to consider as similar
  Returns:
    List of (glycan_id, rmsd, best_structure) tuples sorted by similarity
  """
  query_glycan = canonicalize_iupac(query_glycan)
  glycans = get_glycoshape_IUPAC() if glycan_database is None else glycan_database
  glycans = [g for g in glycans if (global_path / g).exists() and any((global_path / g).iterdir()) and g!=query_glycan]
  # Get query coordinates once
  query_glycan = get_example_pdb(query_glycan) if pdb_path is None else pdb_path
  query_coords, _ = extract_glycan_coords(query_glycan)
  # Prepare args for parallel processing
  process_args = [(g, query_coords, rmsd_cutoff) for g in glycans]
  results = []
  with Pool() as pool:
    for glycan, rmsd, best_structure in tqdm(pool.imap_unordered(_process_single_glycan, process_args),
                                             total=len(glycans), desc="Searching for similar glycans"):
      if rmsd <= rmsd_cutoff and best_structure is not None:
        conformer = '_'.join(best_structure.stem.split('_')[-2:])
        results.append({
                'glycan': glycan,
                'rmsd': round(rmsd, 3),
                'conformer': conformer
                })
  return sorted(results, key=lambda x: x['rmsd'])


def calculate_torsion_angle(coords: List[List[float]]) -> float:
  """Calculate torsion angle from 4 xyz coordinates.
  Args:
    coords (list): List of 4 [x,y,z] coordinates
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
    df (pd.DataFrame): DataFrame with PDB atomic coordinates
    interaction_dict (dict): Dictionary of glycosidic linkages
  Returns:
    pd.DataFrame: Phi/psi angles for each linkage
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
    donor = df[df['residue_number'] == donor_res]
    acceptor = df[df['residue_number'] == acceptor_res]
    # Special handling for sialic acid
    if any(mono in donor_key for mono in {'SIA', 'NGC'}):
      o5_name = 'O6'  # In sialic acid, O5 is actually O6
      c1_name = 'C2'      # Use C2 instead of C1 for sialic acid
    else:
      o5_name = 'O5'
      c1_name = 'C1'      # Normal C1 for other residues
    o_pos = f'O{pos}'
    coords_phi = [
        donor[donor['atom_name'] == o5_name].iloc[0][['x', 'y', 'z']].values.astype(float),
        donor[donor['atom_name'] == c1_name].iloc[0][['x', 'y', 'z']].values.astype(float),
        acceptor[acceptor['atom_name'] == o_pos].iloc[0][['x', 'y', 'z']].values.astype(float),
        acceptor[acceptor['atom_name'] == f'C{pos}'].iloc[0][['x', 'y', 'z']].values.astype(float)
        ]
    has_c6 = not acceptor[acceptor['atom_name'] == 'C6'].empty
    next_c = pos + 1 if (pos < 6 and has_c6) or (pos < 5 and not has_c6) else 1
    coords_psi = [coords_phi[1], coords_phi[2], coords_phi[3], acceptor[acceptor['atom_name'] == f'C{next_c}'].iloc[0][['x', 'y', 'z']].values.astype(float)]
    results.append({
        'linkage': f"{donor_key}-{acceptor_id}",
        'phi': round(calculate_torsion_angle(coords_phi), 2),
        'psi': round(calculate_torsion_angle(coords_psi), 2),
        'anomeric_form': aform,
        'position': pos
        })
  return pd.DataFrame(results)


def calculate_ring_pucker(df: pd.DataFrame, residue_number: int) -> Dict:
  """Calculate ring puckering parameters for a monosaccharide using the Cremer-Pople method.
  Args:
    df (pd.DataFrame): DataFrame with PDB coordinates
    residue_number (int): Residue number to analyze
  Returns:
    dict: Dictionary with puckering parameters
  """
  residue = df[df['residue_number'] == residue_number]
  mono_type = residue['monosaccharide'].iloc[0]
  is_l_sugar = mono_type in {'FUC', 'RAM', 'ARA'}
  # Get ring atoms based on monosaccharide type
  iupac_type = residue['IUPAC'].iloc[0]
  is_sialic = any(x in iupac_type for x in {'Neu', 'Kdn'})
  is_furanose = any(x in iupac_type for x in {'Araf', 'Galf', 'Fruf'})
  if is_sialic:  # 9-atom sialic acid rings
    ring_atoms = ['C2', 'C3', 'C4', 'C5', 'C6', 'O6']
  elif is_furanose:  # 5-membered furanose rings
    ring_atoms = ['C1', 'C2', 'C3', 'C4', 'O4']
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
    qm_sin, qm_cos = 0, 0
    for j in range(n):
      angle = 2 * np.pi * (m + 1) * j / n
      qm_sin += zj[j] * np.sin(angle)
      qm_cos += zj[j] * np.cos(angle)
    qm[m] = np.sqrt(qm_sin**2 + qm_cos**2) * (2/n)
    phi[m] = np.degrees(np.arctan2(qm_sin, qm_cos)) % 360
  # Total puckering amplitude
  Q = np.sqrt(np.sum(qm**2))
  conformation = "Unknown"
  # Phase angle θ
  if is_furanose:  # For 5-membered rings, there are only two puckering parameters (q2 and φ2)
    q2 = qm[0]  # First (and only meaningful) puckering coordinate
    # For furanoses, use φ2 to determine conformation
    theta = phi[0]
    # Envelope and twist conformations for furanoses
    if q2 < 0.1:  # Almost planar
      conformation = "Planar"
    else:  # Determine envelope or twist based on the phase angle
      envelope_types = {0: "C3-endo", 72: "C4-endo", 144: "O4-endo", 216: "C1-endo", 288: "C2-endo"}
      twist_types = {36: "3T4", 108: "4TO", 180: "OT1", 252: "1T2", 324: "2T3"}
      if abs(theta % 72) < 18:  # Within 18° of a multiple of 72°
        # Envelope conformations (E)
        closest_angle = round(theta / 72) * 72
        conformation = envelope_types.get(closest_angle % 360, "")
      else:
        # Twist conformations (T)
        closest_angle = round((theta - 36) / 72) * 72 + 36
        conformation = twist_types.get(closest_angle % 360, "")
  else:
    if is_sialic:
      # For sialic acid rings (using second largest amplitude)
      theta = np.degrees(np.arccos(qm[2] / Q))
      # Adjust the phase angle calculation for the larger ring
      phi = [np.degrees(np.arctan2(
        np.sum([zj[j] * np.sin(2 * np.pi * (m + 1) * j / n) for j in range(n)]),
        np.sum([zj[j] * np.cos(2 * np.pi * (m + 1) * j / n) for j in range(n)])
        )) % 360 for m in range(n//2)]
      # Determine conformation
      if theta < 30:  # Sialic acids typically prefer a 2C5 chair conformation
        conformation = "2C5"
      elif theta > 150:
        conformation = "5C2"  # Less common inverted chair
      elif theta < 90:
        phi_main = phi[2]
        conformation = "B2,5" if (330 <= phi_main or phi_main < 30) else "B3,O6" if (150 <= phi_main < 210) else "S3,5"
      else:
        conformation = "S3,5"  # Most common skew form
    else:
      # For 6-membered rings
      q2, q3 = qm[1], qm[2]  # Second/Third puckering coordinate
      theta = np.degrees(np.arctan2(q2, q3))
      # Determine conformation
      if theta < 45:
        conformation = "4C1" if not is_l_sugar else "1C4"
      elif theta > 135:
        conformation = "1C4" if not is_l_sugar else "4C1"
      else:
        # Check for boat/skew-boat
        boat_types = {0: "B1,4", 60: "B2,5", 120: "B3,6", 180: "B1,4", 240: "B2,5", 300: "B3,6"}
        skew_types = {30: "1S3", 90: "2S6", 150: "3S1", 210: "4S2", 270: "5S3", 330: "6S4"}
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


def get_ring_conformations(df: pd.DataFrame, exclude_types: List[str] = ['ROH', 'MEX', 'PCX', 'SO3', 'ACX']) -> pd.DataFrame:
  """Analyze ring conformations for all residues in structure.
  Args:
    df (pd.DataFrame): DataFrame with PDB coordinates
    exclude_types (list): List of residue types to exclude
  Returns:
    pd.DataFrame: DataFrame with ring parameters for each residue
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
