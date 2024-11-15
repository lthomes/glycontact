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
from urllib.parse import quote
from glycowork.motif.annotate import *
from glycowork.motif.graph import *
from glycowork.motif.tokenization import *
import mdtraj as md


def get_glycoshape_IUPAC() :
    #get the list of available glycans on glycoshape
    curl_command = 'curl -X GET https://glycoshape.io/api/available_glycans'
    x = subprocess.run(curl_command, shell=True, capture_output=True, text=True)
    parsed_dict = json.loads(x.stdout)
    return(parsed_dict['glycan_list'])

def download_from_glycoshape(my_path, IUPAC):
    # Download pdb files given an IUPAC sequence that exists in the glycoshape database
    if ')' not in IUPAC:
       print('This IUPAC corresponds to a single monosaccharide: ignored')
       return False
    if IUPAC[-1]==']':
       print('This IUPAC is not formated properly: ignored')
       return False
    outpath = my_path + '/' + IUPAC
    IUPAC_name = quote(IUPAC)
    os.makedirs(outpath, exist_ok=True)

    for linktype in ['alpha']:
        for i in range(0, 500):
            output = '_' + linktype + '_' + str(i) + '.pdb'
            curl_command = f'curl -o {output} "https://glycoshape.io/database/{IUPAC_name}/PDB_format_ATOM/{IUPAC_name}_cluster{i}_{linktype}.PDB.pdb"'
            tiny_command = f'curl "https://glycoshape.io/database/{IUPAC_name}/PDB_format_ATOM/{IUPAC_name}_cluster{i}_{linktype}.PDB.pdb"'

            try:
                result = subprocess.run(tiny_command, shell=True, capture_output=True, text=True)

                if "404 Not Found" in result.stdout:
                    break  # Continue to the next iteration of the loop

                result = subprocess.run(curl_command, shell=True, capture_output=True, text=True)
                current_file_name = output
                new_file_name = IUPAC + current_file_name

                os.rename(current_file_name, new_file_name)
                shutil.move(new_file_name, outpath)

            except Exception as e:
                print(f"Error: {e}")

    for linktype in ['beta']:
        for i in range(0, 500):

            output = '_' + linktype + '_' + str(i) + '.pdb'

            curl_command = f'curl -o {output} "https://glycoshape.io/database/{IUPAC_name}/PDB_format_ATOM/{IUPAC_name}_cluster{i}_{linktype}.PDB.pdb"'
            tiny_command = f'curl "https://glycoshape.io/database/{IUPAC_name}/PDB_format_ATOM/{IUPAC_name}_cluster{i}_{linktype}.PDB.pdb"'

            try:
                result = subprocess.run(tiny_command, shell=True, capture_output=True, text=True)

                if "404 Not Found" in result.stdout:
                    break  # Continue to the next iteration of the loop

                result = subprocess.run(curl_command, shell=True, capture_output=True, text=True)
                current_file_name = output
                new_file_name = IUPAC + current_file_name

                os.rename(current_file_name, new_file_name)
                shutil.move(new_file_name, outpath)

            except Exception as e:
                print(f"Error: {e}")



def check_available_pdb(glycan) :
    available_files = os.listdir(glycan)
    return(available_files)



def extract_3D_coordinates(pdb_file):
    """
    Extract 3D coordinates from a PDB file and return them as a DataFrame.

    Parameters:
    - pdb_file (str): Path to the PDB file.

    Returns:
    - pd.DataFrame: DataFrame containing the extracted coordinates.
    """

    # Open a temporary file to store relevant lines
    tmp = open('tmp.txt', 'w')

    # Open the PDB file for reading
    with open(pdb_file, 'r') as pdb_f:
        # Iterate through lines in the PDB file
        for line in pdb_f:
            # Check for 'ATOM' and exclude 'REMARK' lines
            if 'ATOM   ' in line and 'REMARK' not in line:
                tmp.write(line)

    # Close the temporary and PDB files
    tmp.close()

    # Define column names for the DataFrame
    columns = ['record_name', 'atom_number', 'atom_name', 'monosaccharide', 'chain_id', 'residue_number',
               'x', 'y', 'z', 'occupancy', 'temperature_factor', 'element']

    # Read the temporary file into a DataFrame using fixed-width format
    df = pd.read_fwf('tmp.txt', names=columns, colspecs=[(0, 6), (6, 11), (12, 16), (17, 20), (21, 22), (22, 26),
                                                         (30, 38), (38, 46), (46, 54), (54, 60), (60, 66), (76, 78)])

    # Display the DataFrame
    return df

def make_atom_contact_table(coord_df, threshold = 10, mode = 'exclusive') :
    ### Create a contact table of atoms of a given PDB file processed into a dataframe by extract_3D_coordinates()
    # coord_df : a dataframe as returned by extract_3D_coordinates()
    # threshold : maximal distance to be considered. Otherwise set to threshold + 1
    # mode : 'exclusive' to avoid intra-residue distance calculation/representation, or 'inclusive' to include intra-residue values

    distanceMap = pd.DataFrame()
    atom_list = coord_df['atom_name'].to_list()
    anum_list = coord_df['atom_number'].to_list()

    mono_nomenclature = 'IUPAC'
    try :
      mono_list = coord_df[mono_nomenclature].to_list()
    except :
      mono_nomenclature = 'monosaccharide'
      mono_list = coord_df[mono_nomenclature].to_list()
    num_list = coord_df['residue_number'].to_list()
    x_list = coord_df['x'].to_list()
    y_list = coord_df['y'].to_list()
    z_list = coord_df['z'].to_list()

    for i in range(0,len(x_list)) :
        current_pos = str(num_list[i]) + '_' + mono_list[i] + '_' + atom_list[i] + '_' + str(anum_list[i])
        c_x = x_list[i]
        c_y = y_list[i]
        c_z = z_list[i]
        distanceList = []
        for j in range(0,len(x_list)) :
            if mode == 'exclusive':
                if str(num_list[i]) != str(num_list[j]) :
                    n_x = x_list[j]
                    n_y = y_list[j]
                    n_z = z_list[j]
                    x_diff = n_x - c_x
                    y_diff = n_y - c_y
                    z_diff = n_z - c_z
                    absdist = abs(x_diff) + abs(y_diff) + abs(z_diff)
                    if absdist <= threshold:
                        distanceList.append(absdist)
                    else :
                        distanceList.append(threshold+1)
                else :
                    distanceList.append(0)
            if mode=='inclusive' :
                n_x = x_list[j]
                n_y = y_list[j]
                n_z = z_list[j]
                x_diff = n_x - c_x
                y_diff = n_y - c_y
                z_diff = n_z - c_z
                absdist = abs(x_diff) + abs(y_diff) + abs(z_diff)
                if absdist <= threshold:
                    distanceList.append(absdist)

                else :
                    distanceList.append(threshold+1)
        distanceMap[current_pos] = distanceList

    distanceMap.index = distanceMap.columns
    return(distanceMap)

def make_monosaccharide_contact_table(coord_df, threshold = 10, mode = 'binary') :
    # threshold : maximal distance to be considered. Otherwise set to threshold + 1
    # mode : can be either binary (return a table with 0 or 1 based on threshold), distance (return a table with the distance or threshold+1 based on threshold), or both
    distanceMap = pd.DataFrame()
    distanceMap2 = pd.DataFrame()

    atom_list = coord_df['atom_name'].to_list()
    anum_list = coord_df['atom_number'].to_list()

    mono_nomenclature = 'IUPAC'
    try :
      mono_list = coord_df[mono_nomenclature].to_list()
    except :
      mono_nomenclature = 'monosaccharide'
      mono_list = coord_df[mono_nomenclature].to_list()
    num_list = coord_df['residue_number'].to_list()


    for i in list(set(num_list)) :
        ndf = coord_df[coord_df['residue_number']==i]
        current_pos = str(i) + '_' + ndf[mono_nomenclature].to_list()[0]
        distanceList = []
        distanceList2 = []

        x_list = ndf['x'].to_list()
        y_list = ndf['y'].to_list()
        z_list = ndf['z'].to_list()

        for j in list(set(num_list)) :
            adverse_df = coord_df[coord_df['residue_number']==j]
            adverse_pos = str(j) + '_' + adverse_df[mono_nomenclature].to_list()[0]
            added = False
            nx_list = adverse_df['x'].to_list()
            ny_list = adverse_df['y'].to_list()
            nz_list = adverse_df['z'].to_list()

            for k in range(0,len(x_list)) :
                c_x = x_list[k]
                c_y = y_list[k]
                c_z = z_list[k]

                for l in range(0,len(nx_list)) :
                    n_x = nx_list[l]
                    n_y = ny_list[l]
                    n_z = nz_list[l]
                    x_diff = n_x - c_x
                    y_diff = n_y - c_y
                    z_diff = n_z - c_z
                    absdist = abs(x_diff) + abs(y_diff) + abs(z_diff)
                    if absdist <= threshold and added == False :
                        distanceList.append(0)
                        distToAppend = absdist
                        added = True
                    if absdist <= threshold and added == True :
                        if absdist < distToAppend :
                            distToAppend = absdist
            if added == True  :
                distanceList2.append(distToAppend)
            if added == False :
                distanceList.append(1)
                distanceList2.append(threshold+1)

        distanceMap[current_pos] = distanceList
        distanceMap2[current_pos] = distanceList2

    distanceMap.index = distanceMap.columns
    distanceMap2.index = distanceMap2.columns

    if mode == 'binary' :
        return(distanceMap)
    if mode == 'distance' :
        return(distanceMap2)
    if mode == 'both' :
        return([distanceMap,distanceMap2])

def focus_table_on_residue(table, residue) :
    ### Take a monosaccharide Contact Table and focus it to keep only one residue type (ie MAN)
    table['y'] = table.columns.to_list()
    table = table[table['y'].str.contains(residue,regex = False)==True]
    new_table = table[[f for f in table.columns.to_list() if residue in f]]
    return new_table

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

    dfs = []

    pdbs = check_available_pdb(my_path + glycan)
    if link_type == 'alpha':
        pdb_files = [my_path + glycan + "/" + pdb for pdb in pdbs if 'alpha' in pdb]
    if link_type == 'beta':
        pdb_files = [my_path + glycan + "/" + pdb for pdb in pdbs if 'beta' in pdb]
    pdb_files.sort()

    for f in pdb_files:
        df = explore_threshold(f, glycan, threshold_list=[2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 2.45, 2.55, 2.65, 2.75, 2.85, 2.95, 3, 2.2, 2.25, 2.3, 2.35, 3.5])
        dist_table = make_monosaccharide_contact_table(df, mode='distance', threshold=200)
        dfs.append(dist_table)

    col_to_parse = dfs[0].columns.to_list()

    outdf = pd.DataFrame(columns=col_to_parse)
    outdf_power = pd.DataFrame(columns=col_to_parse)

    # Only compute cluster frequencies if mode is 'weighted'
    if mode == 'weighted':
        cluster_frequencies = get_glycan_clusters_frequency(glycan)
        weights = [n / 100 for n in cluster_frequencies]
        outdf_weighted = pd.DataFrame(columns=col_to_parse)
    
    for col_index in range(len(col_to_parse)):
        current_column = col_to_parse[col_index]
        new_column = []
        new_column2 = []
        if mode == 'weighted':
            weighted_column = []
        
        list_of_values_lists = [df[current_column].to_list() for df in dfs]
        
        for y in range(len(list_of_values_lists[0])):
            values = [liste[y] for liste in list_of_values_lists]
            mean = np.mean(values)
            deviation_from_mean = [abs(v - mean) for v in values]
            sum_of_deviations = sum(deviation_from_mean)
            power_of_deviations = sum_of_deviations ** 2

            new_column.append(sum_of_deviations)
            new_column2.append(power_of_deviations)
            
            if mode == 'weighted':
                weighted_deviation = np.average(deviation_from_mean, weights=weights, axis=0)
                weighted_column.append(weighted_deviation)

        outdf[current_column] = new_column
        outdf_power[current_column] = new_column2
        
        if mode == 'weighted':
            outdf_weighted[current_column] = weighted_column

    if mode == 'standard':
        return outdf
    elif mode == 'amplify':
        return outdf_power
    elif mode == 'weighted':
        return outdf_weighted

def make_correlation_matrix(my_path, glycan, link_type):
    ### Compute a Pearson correlation matrix
    # my_path : path to the folder containing all PDB folders
    # glycan : glycan in IUPAC sequence
    # link_type : 'alpha' or 'beta' to work with alpha- or beta-linked glycans
    
    dfs = []

    pdbs = check_available_pdb(my_path+glycan)
    if link_type == 'alpha':
        pdb_files = [my_path+glycan+"/"+pdb for pdb in pdbs if 'alpha' in pdb]
    if link_type == 'beta':
        pdb_files = [my_path+glycan+"/"+pdb for pdb in pdbs if 'beta' in pdb]
    pdb_files.sort()

    for f in pdb_files :
        df = explore_threshold(f, glycan, threshold_list=[2.4,2.5,2.6,2.7,2.8,2.9,2.45,2.55,2.65,2.75,2.85,2.95,3,2.2,2.25,2.3,2.35,3.5])
        dist_table = make_monosaccharide_contact_table(df,mode='distance', threshold = 200)
        dfs.append(dist_table)

    # Create an empty correlation matrix
    correlation_matrix = np.zeros((len(dfs[0]), len(dfs[0])))

    # Calculate the correlation matrix based on the distances
    for df in dfs:
        distances = df.values  
        correlation_matrix += np.corrcoef(distances, rowvar=False)

    # Average the correlation matrix
    correlation_matrix /= len(dfs)
    corr_df = pd.DataFrame(correlation_matrix, columns=df.columns, index=df.columns)
    return corr_df

def inter_structure_frequency_table(my_path, glycan, link_type, threshold = 5):
    ### Creates a table as make_atom_contact_table() or the monosaccharide equivalent but values represent the frequency of monosaccharides/atoms pairs that crossed a threshold distance across different PDB of the same molecule
    # my_path : path to the folder containing all PDB folders
    # glycan : glycan in IUPAC sequence
    # link_type : 'alpha' or 'beta' to work with alpha- or beta-linked glycans
    # threshold : maximal distance a pair can show to be counted as a contact

    dfs = []

    pdbs = check_available_pdb(my_path+glycan)
    if link_type == 'alpha':
        pdb_files = [my_path+glycan+"/"+pdb for pdb in pdbs if 'alpha' in pdb]
    if link_type == 'beta':
        pdb_files = [my_path+glycan+"/"+pdb for pdb in pdbs if 'beta' in pdb]
    pdb_files.sort()

    for f in pdb_files :
        df = explore_threshold(f, glycan, threshold_list=[2.4,2.5,2.6,2.7,2.8,2.9,2.45,2.55,2.65,2.75,2.85,2.95,3,2.2,2.25,2.3,2.35,3.5])
        dist_table = make_monosaccharide_contact_table(df,mode='distance', threshold = 200)
        dfs.append(dist_table)

    # Apply thresholding and create a new list of transformed DataFrames
    transformed_dfs = [df.applymap(lambda x: 1 if x < threshold else 0) for df in dfs]
    # Sum up the transformed DataFrames to create the final DataFrame
    final_df = pd.DataFrame(sum(transformed_dfs))
    return(final_df)

def extract_binary_interactions_from_PDB(coordinates_df, threshold):
    """
    Extract binary interactions between C1-2 atoms and oxygen atoms from a DataFrame.

    Parameters:
    - coordinates_df (pd.DataFrame): DataFrame obtained using extract_3D_coordinates.
    - threshold (float): Distance threshold for considering interactions.

    Returns:
    - pd.DataFrame: DataFrame with columns 'Atom', 'Column', and 'Value' representing interactions.
    """
    #coordinates_df =  correct_dataframe(extract_3D_coordinates(coordinates_df))
    carbon_1_2_df = coordinates_df[(coordinates_df['atom_name'] == 'C1') | (coordinates_df['atom_name'] == 'C2')]
    oxygen_df = coordinates_df[coordinates_df['element'] == 'O']

    c_dict = {f"{r}_{m}_{a}": [x, y, z] for r, m, a, x, y, z in carbon_1_2_df[['residue_number', 'monosaccharide', 'atom_name', 'x', 'y', 'z']].values}
    o_dict = {f"{r}_{m}_{a}": [x, y, z] for r, m, a, x, y, z in oxygen_df[['residue_number', 'monosaccharide', 'atom_name', 'x', 'y', 'z']].values}

    atom = []
    column = []
    value = []

    for c_key, c_coords in c_dict.items():
        smallest_distance = 1000
        closest_residue = ''
        c_resnum = c_key.split('_')[0]

        for o_key, o_coords in o_dict.items():
            o_resnum = o_key.split('_')[0]

            if c_resnum != o_resnum:
                sum_dist = np.sum(np.abs(np.array(c_coords) - np.array(o_coords)))

                if sum_dist < smallest_distance:
                    smallest_distance = sum_dist
                    closest_residue = o_key

        if smallest_distance < threshold:
            atom.append(c_key)
            column.append(closest_residue)
            value.append(smallest_distance)

    interactions_df = pd.DataFrame({'Atom': atom, 'Column': column, 'Value': value})
    return interactions_df


def get_glycan_sequence_from_path(pdb_file) :
  # Simply extract the glycan sequence from path and filename
  seq = pdb_file.split('/')[0]
  return(seq)

def extract_numbers(input_string):
    # Use regular expression to extract numbers
    numbers = re.findall(r'\d+', input_string)

    # Join the extracted numbers into a single string
    result = ''.join(numbers)

    return result

def PDB_to_IUPAC(pdb_mono):
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
  # MAN indicates either alpha and beta bonds, instead of just alpha.. this is a problem
  # GalNAc is recorded as "GLC" which is wrong: need for a checker function that counts the number of atoms - Glc = 21 (<25), GalNAc = 28 (>25)
  mono_core = map_dict[pdb_mono.split('_')[1]]
  return(mono_core)

def create_mapping_dict_and_interactions(df, valid_fragments, n_glycan) :
  #df is an interaction dataframe as returned by extract_binary_interactions_from_PDB()
  # valid_fragments : obtained from glycowork to ensure that we only append valid monolinks into mapping dict
  # n_glycan : True or False, indicates if the first mannose should be corrected or not
  mapping_dict = {}
  interaction_dict = {}
  interaction_dict2 = {}
  first_mono_list = df['Atom'].to_list()
  #print(first_mono_list)
  second_mono_list = df['Column'].to_list()
  #print(second_mono_list)
  mapping_dict['1_ROH']='-R'
  wrong_mannose = []
  individual_entities = []

  for m in range(0,len(first_mono_list)) :

    mono = first_mono_list[m].replace("_"+first_mono_list[m].split('_')[-1],'')
    second_mono = second_mono_list[m].replace("_"+second_mono_list[m].split('_')[-1],'')
    first_val = first_mono_list[m].split('_')[-1]
    first_val = extract_numbers(first_val)
    last_val = extract_numbers(second_mono_list[m].split('_')[-1])
    if mono not in individual_entities:
      individual_entities.append(mono)
    if second_mono not in individual_entities:
      individual_entities.append(second_mono)

    if mono.split('_')[1] + '(' + first_val + '-' + last_val + ')' == "MAN(1-4)" and n_glycan == True:
      wrong_mannose.append(mono)

    if second_mono in wrong_mannose :
      second_mono = second_mono.split('_')[0]+"_BMA"

    if mono in wrong_mannose :
      mono = mono.split('_')[0]+"_BMA"
    
    mapped_to_check = PDB_to_IUPAC(mono) + first_val + '-' + last_val + ')'
    #print("mapped_to_check:" + str(mapped_to_check))
    #print(valid_fragments)

    if mapped_to_check in valid_fragments :
      mapping_dict[mono] = PDB_to_IUPAC(mono) + first_val + '-' + last_val + ')'
    if mapped_to_check == 'Man(a1-4)':
      mapping_dict[mono] = 'Man(b1-4)'
    if mapped_to_check == 'GlcNAc(a1-1)' or mapped_to_check == 'GlcNAc(b1-1)' or mapped_to_check == '-R' or mapped_to_check =='GalNAc(a1-1)' or mapped_to_check == 'GalNAc(b1-1)' or mapped_to_check =='Glc(a1-1)' or mapped_to_check == 'Glc(b1-1)' or mapped_to_check =='Rha(a1-1)' or mapped_to_check == 'Rha(b1-1)' or mapped_to_check =='Neu5Ac(a2-1)' or mapped_to_check == 'Neu5Ac(b2-1)' or mapped_to_check =='Man(a1-1)' or mapped_to_check == 'Man(b1-1)' or mapped_to_check =='Gal(a1-1)' or mapped_to_check == 'Gal(b1-1)' or mapped_to_check =='Fuc(a1-1)' or mapped_to_check == 'Fuc(b1-1)' or mapped_to_check =='Xyl(a1-1)' or mapped_to_check == 'Xyl(b1-1)' or mapped_to_check =='GlcA(a1-1)' or mapped_to_check == 'GlcA(b1-1)' or mapped_to_check =='GlcNS(a1-1)' or mapped_to_check == 'GlcNS(b1-1)' or mapped_to_check =='GlcNAc6S(a1-1)' or mapped_to_check == 'GlcNAc6S(b1-1)' or mapped_to_check =='GlcNS6S(a1-1)' or mapped_to_check == 'GlcNS6S(b1-1)' or mapped_to_check == 'GlcNS3S6S(a1-1)' or mapped_to_check == 'GlcNS3S6S(b1-1)' or mapped_to_check == '2-4-diacetimido-2-4-6-trideoxyhexose(a1-1)' or mapped_to_check == 'GlcA2S(a1-1)' or mapped_to_check == 'GlcA2S(b1-1)' or mapped_to_check == 'Ara(a1-1)' or mapped_to_check == 'Ara(b1-1)' or mapped_to_check == 'Fru(a1-1)' or mapped_to_check == 'Fru(b1-1)' or mapped_to_check == 'ManNAc(a1-1)' or mapped_to_check == 'ManNAc(b1-1)':
      mapping_dict[mono] = mapped_to_check


    if mono in interaction_dict :
      if second_mono not in interaction_dict[mono] :
        interaction_dict[mono].append(second_mono)
        interaction_dict2[mono] = [mono.split('_')[0]+'_(' + PDB_to_IUPAC(mono).split('(')[1] + first_val + '-' + last_val + ')']
        interaction_dict2[mono.split('_')[0]+'_(' + PDB_to_IUPAC(mono).split('(')[1] + first_val + '-' + last_val + ')'] = [second_mono] #added but eventually wrong, make everything else fail later
    if mono not in interaction_dict :
      if mapped_to_check in valid_fragments or mapped_to_check == 'Man(a1-4)' or mapped_to_check == '-R' or mapped_to_check == 'GlcNAc(a1-1)' or mapped_to_check == 'GlcNAc(b1-1)' or mapped_to_check =='GalNAc(a1-1)' or mapped_to_check == 'GalNAc(b1-1)' or mapped_to_check =='Glc(a1-1)' or mapped_to_check == 'Glc(b1-1)' or mapped_to_check =='Rha(a1-1)' or mapped_to_check == 'Rha(b1-1)' or mapped_to_check =='Neu5Ac(a2-1)' or mapped_to_check == 'Neu5Ac(b2-1)' or mapped_to_check =='Man(a1-1)' or mapped_to_check == 'Man(b1-1)' or mapped_to_check =='Gal(a1-1)' or mapped_to_check == 'Gal(b1-1)' or mapped_to_check =='Fuc(a1-1)' or mapped_to_check == 'Fuc(b1-1)' or mapped_to_check =='Xyl(a1-1)' or mapped_to_check == 'Xyl(b1-1)' or mapped_to_check =='GlcA(a1-1)' or mapped_to_check == 'GlcA(b1-1)' or mapped_to_check =='GlcNS(a1-1)' or mapped_to_check == 'GlcNS(b1-1)' or mapped_to_check =='GlcNAc6S(a1-1)' or mapped_to_check == 'GlcNAc6S(b1-1)' or mapped_to_check =='GlcNS6S(a1-1)' or mapped_to_check == 'GlcNS6S(b1-1)' or mapped_to_check == 'GlcNS3S6S(a1-1)' or mapped_to_check == 'GlcNS3S6S(b1-1)' or mapped_to_check == '2-4-diacetimido-2-4-6-trideoxyhexose(a1-1)' or mapped_to_check == 'GlcA2S(a1-1)' or mapped_to_check == 'GlcA2S(b1-1)' or mapped_to_check == 'Ara(a1-1)' or mapped_to_check == 'Ara(b1-1)' or mapped_to_check == 'Fru(a1-1)' or mapped_to_check == 'Fru(b1-1)' or mapped_to_check == 'ManNAc(a1-1)' or mapped_to_check == 'ManNAc(b1-1)':
        interaction_dict[mono] = [second_mono]
        interaction_dict2[mono] = [mono.split('_')[0]+'_(' + PDB_to_IUPAC(mono).split('(')[1] + first_val + '-' + last_val + ')']

      if mono.split('_')[0]+'_(' + PDB_to_IUPAC(mono).split('(')[1] + first_val + '-' + last_val + ')' in interaction_dict2 :
        if mapped_to_check in valid_fragments or mapped_to_check == 'Man(a1-4)' or mapped_to_check == '-R' or mapped_to_check == 'GlcNAc(a1-1)' or mapped_to_check == 'GlcNAc(b1-1)' or mapped_to_check =='GalNAc(a1-1)' or mapped_to_check == 'GalNAc(b1-1)' or mapped_to_check =='Glc(a1-1)' or mapped_to_check == 'Glc(b1-1)' or mapped_to_check =='Rha(a1-1)' or mapped_to_check == 'Rha(b1-1)' or mapped_to_check =='Neu5Ac(a2-1)' or mapped_to_check == 'Neu5Ac(b2-1)' or mapped_to_check =='Man(a1-1)' or mapped_to_check == 'Man(b1-1)' or mapped_to_check =='Gal(a1-1)' or mapped_to_check == 'Gal(b1-1)' or mapped_to_check =='Fuc(a1-1)' or mapped_to_check == 'Fuc(b1-1)' or mapped_to_check =='Xyl(a1-1)' or mapped_to_check == 'Xyl(b1-1)' or mapped_to_check =='GlcA(a1-1)' or mapped_to_check == 'GlcA(b1-1)' or mapped_to_check =='GlcNS(a1-1)' or mapped_to_check == 'GlcNS(b1-1)' or mapped_to_check =='GlcNAc6S(a1-1)' or mapped_to_check == 'GlcNAc6S(b1-1)' or mapped_to_check =='GlcNS6S(a1-1)' or mapped_to_check == 'GlcNS6S(b1-1)' or mapped_to_check == 'GlcNS3S6S(a1-1)' or mapped_to_check == 'GlcNS3S6S(b1-1)' or mapped_to_check == '2-4-diacetimido-2-4-6-trideoxyhexose(a1-1)' or mapped_to_check == 'GlcA2S(a1-1)' or mapped_to_check == 'GlcA2S(b1-1)' or mapped_to_check == 'Ara(a1-1)' or mapped_to_check == 'Ara(b1-1)' or mapped_to_check == 'Fru(a1-1)' or mapped_to_check == 'Fru(b1-1)' or mapped_to_check == 'ManNAc(a1-1)' or mapped_to_check == 'ManNAc(b1-1)':
          interaction_dict2[mono.split('_')[0]+'_(' + PDB_to_IUPAC(mono).split('(')[1] + first_val + '-' + last_val + ')'].append(second_mono)

      if mono.split('_')[0]+'_(' + PDB_to_IUPAC(mono).split('(')[1] + first_val + '-' + last_val + ')' not in interaction_dict2 :
        if mapped_to_check in valid_fragments or mapped_to_check == 'Man(a1-4)' or mapped_to_check == '-R' or mapped_to_check == 'GlcNAc(a1-1)' or mapped_to_check == 'GlcNAc(b1-1)' or mapped_to_check =='GalNAc(a1-1)' or mapped_to_check == 'GalNAc(b1-1)' or mapped_to_check =='Glc(a1-1)' or mapped_to_check == 'Glc(b1-1)' or mapped_to_check =='Rha(a1-1)' or mapped_to_check == 'Rha(b1-1)' or mapped_to_check =='Neu5Ac(a2-1)' or mapped_to_check == 'Neu5Ac(b2-1)' or mapped_to_check =='Man(a1-1)' or mapped_to_check == 'Man(b1-1)' or mapped_to_check =='Gal(a1-1)' or mapped_to_check == 'Gal(b1-1)' or mapped_to_check =='Fuc(a1-1)' or mapped_to_check == 'Fuc(b1-1)' or mapped_to_check =='Xyl(a1-1)' or mapped_to_check == 'Xyl(b1-1)' or mapped_to_check =='GlcA(a1-1)' or mapped_to_check == 'GlcA(b1-1)' or mapped_to_check =='GlcNS(a1-1)' or mapped_to_check == 'GlcNS(b1-1)' or mapped_to_check =='GlcNAc6S(a1-1)' or mapped_to_check == 'GlcNAc6S(b1-1)' or mapped_to_check =='GlcNS6S(a1-1)' or mapped_to_check == 'GlcNS6S(b1-1)' or mapped_to_check == 'GlcNS3S6S(a1-1)' or mapped_to_check == 'GlcNS3S6S(b1-1)' or mapped_to_check == '2-4-diacetimido-2-4-6-trideoxyhexose(a1-1)' or mapped_to_check == 'GlcA2S(a1-1)' or mapped_to_check == 'GlcA2S(b1-1)' or mapped_to_check == 'Ara(a1-1)' or mapped_to_check == 'Ara(b1-1)' or mapped_to_check == 'Fru(a1-1)' or mapped_to_check == 'Fru(b1-1)' or mapped_to_check == 'ManNAc(a1-1)' or mapped_to_check == 'ManNAc(b1-1)':
          interaction_dict2[mono.split('_')[0]+'_(' + PDB_to_IUPAC(mono).split('(')[1] + first_val + '-' + last_val + ')'] = [second_mono]

  return(mapping_dict, interaction_dict2)

def extract_binary_glycontact_interactions(interaction_dict):
  # transform the interactions detected in the PDB file into IUPAC binary interactions for further comparison to glycowork
  # interaction_dict formatted as: {'12_GAL': ['12_(b1-4)'], '12_(b1-4)': ['11_NAG'], '13_AFL': ['13_(a1-3)'], '13_(a1-3)': ['11_NAG']}
  result_list = []
  for k in interaction_dict:
      if '(' in k:
          new_k = k.split('_')[1].replace('(', '').replace(')', '')
      else:
          new_k = PDB_to_IUPAC(k).split('(')[0]

      if '(' in interaction_dict[k][0]:
          new_v = interaction_dict[k][0].split('_')[1].replace('(', '').replace(')', '')
      else:
          new_v = PDB_to_IUPAC(interaction_dict[k][0]).split('(')[0]
      result_list.append((new_k, new_v))

  return(result_list)

def extract_binary_glycowork_interactions(graph_output):
    """
    Extracts a list of binary interactions from the output of glycan_to_graph function.

    Parameters:
    - graph_output (tuple): The output tuple from glycan_to_graph function.

    Returns:
    - list of binary interactions as pairs of labels.
    """
    mask_dic, adj_matrix = graph_output
    n = adj_matrix.shape[0]

    interactions_with_labels = []
    for k in range(n):
        for j in range(k + 1, n):
            if adj_matrix[k, j] == 1:
                label_k = mask_dic[k]
                label_j = mask_dic[j]
                interactions_with_labels.append((label_k, label_j))
    return(interactions_with_labels)

def glycowork_vs_glycontact_interactions(glycowork_interactions, glycontact_interactions) :
  # Take two sets of binary interactions to compare them and return any difference other than GlcNAc-a1-1 and a1-1-R (only considered by glycontact)

  # Convert the lists to sets for easy comparison
  result_set = set(glycontact_interactions)
  interactions_set = set(glycowork_interactions)

  # Calculate the differences
  differences = result_set.symmetric_difference(interactions_set)

  # Convert the differences set back to a list of tuples
  differences_list = list(differences)

  # Pairs to be ignored because specific to glycontact
  ignore_pairs = {('GlcNAc', 'a1-1'), ('a1-1', ' '), ('a2-1', ' '), ('b2-1', ' '), ('GlcNAc', 'b1-1'), ('b1-1', ' '), ('GalNAc', 'a1-1'), ('GalNAc', 'b1-1'), ('Glc', 'a1-1'), ('Glc', 'b1-1'), ('Rha', 'b1-1'), ('Rha', 'a1-1'), ('Neu5Ac', 'b2-1'), ('Neu5Ac', 'a2-1'), ('Man', 'b1-1'), ('Man', 'a1-1'), ('Gal', 'b1-1'), ('Gal', 'a1-1'), ('Fuc', 'b1-1'), ('Fuc', 'a1-1'), ('Xyl', 'b1-1'), ('Xyl', 'a1-1'), ('GlcA', 'a1-1'), ('GlcA', 'b1-1'), ('GlcNS', 'a1-1'), ('GlcNS', 'b1-1'), ('GlcNAc6S', 'a1-1'), ('GlcNAc6S','b1-1') , ('GlcNS6S', 'a1-1'), ('GlcNS6S','b1-1'),  ('GlcNS3S6S', 'a1-1'), ('GlcNS3S6S','b1-1'), ('2-4-diacetimido-2-4-6-trideoxyhexose','a1-1'), ('GlcA2S','a1-1'), ('GlcA2S','b1-1'), ('Ara','a1-1'), ('Ara','b1-1'), ('Fru','a1-1'), ('Fru','b1-1'), ('ManNAc','a1-1'), ('ManNAc','b1-1')}

  # Filter out pairs to be ignored
  filtered_differences = [pair for pair in differences_list if pair not in ignore_pairs]

  # Print or use the filtered_differences as needed
  #print("Filtered Differences:", filtered_differences)
  if filtered_differences == [] and  (len(glycontact_interactions) > len(glycowork_interactions)):
    return(True)
  else :
    if filtered_differences != [] :
      #print('Differences in annotations')
      return(False)
    if (len(glycontact_interactions) <= len(glycowork_interactions)) :
      #print("Missing monosaccharide in mapping_dict")
      return(False)

def check_reconstructed_interactions(interaction_dict) :
  # Use the interaction_dict to build a NetworkX network and check that it contains a single component (meaning that the glycan has been correctly reconstructed from the PDB file)

  # Create a directed graph
  G_dict = nx.Graph()

  # Add nodes and edges from dictionary interactions
  for node, neighbors in interaction_dict.items():
    G_dict.add_node(node)
    G_dict.add_edges_from((node, neighbor) for neighbor in neighbors)

  is_single_component = nx.is_connected(G_dict)

  if is_single_component:
      #print("The graph has only one connected component.")
      return(True)
  else:
      #print("The graph has more than one connected component.")
      return(False)

def annotate_pdb_data(pdb_dataframe, mapping_dict) :
  mono_list = pdb_dataframe['monosaccharide'].to_list()
  id_list = pdb_dataframe['residue_number'].to_list()
  m_dict = copy.deepcopy(mapping_dict)
  for m in m_dict :
    if "BMA" in m :
      mapping_dict[m.split('_')[0]+"_"+"MAN"] = mapping_dict[m] #restore the corrected mannose into a wrong one for annotation
  IUPAC = []
  for m in range(0,len(mono_list)):
    monosaccharide = str(id_list[m]) + "_" + mono_list[m]
    try :
      IUPAC.append(mapping_dict[monosaccharide])
    except :
      IUPAC.append(mono_list[m])
  pdb_dataframe['IUPAC']= IUPAC
  return(pdb_dataframe)

def correct_dataframe(df):
  #Correct an annotated dataframe, transforming unexpected GLC into GalNAc based on the number of C atom they contain
  ### WARNING: this is a modified version of the function, assuming that it is always GalNAc(b which is wrong
  resnum = list(set(df['residue_number'].tolist()))

  for x in resnum:
    #Correcting GLC to GalNAc
    condition = (df['monosaccharide'] == 'GLC') & (df['residue_number'] == x) & (df[(df['residue_number'] == x) & (df['element'] == 'C')]['element'].count() >= 7)

    if condition.any():
        df.loc[condition, 'monosaccharide'] = df.loc[condition, 'monosaccharide'].map(lambda x: x.replace('GLC', 'NGA'))

    condition = (df['monosaccharide'] == 'BGC') & (df['residue_number'] == x) & (df[(df['residue_number'] == x) & (df['element'] == 'C')]['element'].count() >= 7)

    if condition.any():
        df.loc[condition, 'monosaccharide'] = df.loc[condition, 'monosaccharide'].map(lambda x: x.replace('GLC', 'A2G'))

  return df

def annotation_pipeline(pdb_file, glycan,threshold =2.7) :
  ### Huge function combining all smaller ones required to annotate a PDB file into IUPAC nomenclature, ensuring that the conversion is correct
  ### It allows also to determine if PDB to IUPAC conversion at the monosaccharide level works fine

  # In pipeline, if IUPAC is detected as containing modified monosaccharide, pre-step of annotation of pdbfile
  modified_mono = ["GlcNAc6S", "GalNAc4S", "IdoA2S", "GlcA3S", "GlcA2S", "Neu5Ac9Ac", 
                  "Man3Me", "Neu5Ac9Me", "Neu5Gc9Me", "GlcA4Me", "Gal6S", "GlcNAc6Pc",
                  "GlcNS6S", "GlcNS3S6S"]
  ### note: GalNAcXS will be annotated as GLC as well, so I need to correct it as well

  # List of non-monosaccharide
  non_mono_list=['SO3', 'ACX', 'MEX', 'PCX']

  # Dict of modifications {link_modif:IUPAC} 
  #modif_dict = {"6SO3":"6S"} #--> allows to write GlcNAc

  # Custom PDB codes to IUPAC (check GLC/GalNAc thingy)
  custom_pdb = {"NAG6SO3":"GlcNAc6S", "NDG6SO3":"GlcNAc6S",  "NDG3SO3":"GlcNAc3S6S", "NGA4SO3":"GalNAc4S", "IDR2SO3":"IdoA2S", 
                "BDP3SO3":"GlcA3S", "BDP2SO3":"GlcA2S", "SIA9ACX":"Neu5Ac9Ac", "MAN3MEX":"Man3Me", 
                "SIA9MEX":"Neu5Ac9Me", "NGC9MEX":"Neu5Gc9Me", "BDP4MEX":"GlcA4Me", "GAL6SO3":"Gal6S", 
                "NAG6PCX":"GlcNAc6Pc", "UYS6SO3":"GlcNS6S", "4YS6SO3":"GlcNS6S", "6YS6SO3":"GlcNS6S", "GCU2SO3":"GlcA2S",  
              'VYS3SO3':'GlcNS3S6S',  'VYS6SO3':'GlcNS3S6S', "QYS3SO3":"GlcNS3S6S", "QYS6SO3":"GlcNS3S6S"}
  

  ### Extract glycan sequence from filename
  #glycan_sequence = get_glycan_sequence_from_path(pdb_file)
  glycan_sequence = glycan
  n_glycan = False
  if 'Man(b1-4)GlcNAc(b1-4)' in glycan :
    n_glycan = True

  # To modify dict
  to_modify_dict = {}

  df = correct_dataframe(extract_3D_coordinates(pdb_file))

  modified_glycan = False 
  for mm in modified_mono :
      if mm in glycan_sequence : 
          modified_glycan = True

  if modified_glycan == True :

    #list of residue_number
    resnum = list(set(df.residue_number.to_list()))
    resdict = {}
    for x in resnum :
        mono = list(set(df['monosaccharide'][df['residue_number']==x].to_list()))[0]
        resdict[x] = mono

    #make an atomic distance table
    dist_table = make_atom_contact_table(df)
    #print(dist_table)

    #For each element in resdict, those that are non-monosaccharide must be investigated
    for key in resdict :
        val = resdict[key]
        if val in non_mono_list :
            element = str(key) + "_" + val
            contact_table = dist_table.filter(regex=element) #keep only columns with a given non-monosaccharide
            
            mask = ~contact_table.index.str.contains('|'.join(contact_table.columns))
            contact_table = contact_table.loc[mask] #keep only lines without this given non-monosaccharide
            
            split_index = contact_table.index.str.split('_')

            # créer un masque pour filtrer les lignes dont le troisième élément ne contient pas "H"
            mask = ['H' not in x[2] for x in split_index]

            # filtrer les lignes de la dataframe en utilisant le masque
            filtered_table = contact_table.loc[mask]

            partners = filtered_table[filtered_table != 0].stack().idxmin() #valeur non nulle la plus faible de la dataframe
            sugar_partner = partners[0]

            #get monosaccharide resnum and non-mono resnum to give them mono resnum but custom monosaccharide annotation
            sugar_resnum, sugar, atom, atom_num = sugar_partner.split("_")
            #link_pos = str([''.join(c for c in s if c.isdigit()) for s in atom][-1])
            link_pos = str(re.findall(r'\d+', atom)[0])
            modif = link_pos+val
            modified_mono = sugar + modif 
            modified_mono_iupac = custom_pdb[modified_mono]

            #print(link_pos)
            #print(modif)
            #print(modified_mono)
            #print(modified_mono_iupac)

            #List all resnum lines that will require modification and which modif
            to_modify_dict[int(sugar_resnum)] =  modified_mono
            to_modify_dict[key] =  [modified_mono, sugar_resnum]




    # charger la dataframe à partir du fichier file.pdb
    df = correct_dataframe(extract_3D_coordinates(pdb_file))

    # parcourir chaque ligne de la dataframe et appliquer les règles
    for index, row in df.iterrows():
        residue_number = row['residue_number']
        if residue_number in to_modify_dict:
            if type(to_modify_dict[residue_number]) is str : 
                monosaccharide = to_modify_dict[residue_number]
                df.at[index, 'monosaccharide'] = monosaccharide
            
            if type(to_modify_dict[residue_number]) is list:
                monosaccharide = to_modify_dict[residue_number][0]
                df.at[index, 'monosaccharide'] = monosaccharide
                new_residue_number = int(to_modify_dict[residue_number][1])
                df.at[index, 'residue_number'] = new_residue_number

    # conserver la dataframe modifiée en variable
    df_modified = df.copy()
    df= df_modified.sort_values(by='residue_number', key=lambda x: x.astype(int))

  
  ### Using glycowork, extract valid fragments (fragment = monolink like GlcNAc(b1-4))
  valid_fragments = [x.split(')')[0]+')' for x in link_find(glycan_sequence)]
  #print(valid_fragments)
  #print(list(set(df['monosaccharide'].to_list())))
  ### Detect binary connections (covalent linkages) using a maximal distance threshold and valid_fragments + build a mapping dictionnary
  res = extract_binary_interactions_from_PDB(df,threshold)

  mapping_dict, interaction_dict = create_mapping_dict_and_interactions(res,valid_fragments, n_glycan)
  #print(mapping_dict)
  #print(interaction_dict)
  #print(len(mapping_dict))
  #print(len(interaction_dict))

  ### Comparison of glycowork linkages and glycontact linkages to ensure correct extraction from PDB
  # Extract glycowork interactions:
  graph_output = glycan_to_graph(glycan_sequence)
  interactions_with_labels = extract_binary_glycowork_interactions(graph_output)
  #print(interactions_with_labels)

  # Extract glycontact interactions:
  result_list = extract_binary_glycontact_interactions(interaction_dict)
  #print("result list:" + str(result_list))
  # Compare glycowork IUPAC to graph versus glycontact PDB to graph to ensure glycontact detection of covalent linkages is correct (must return True)
  if glycowork_vs_glycontact_interactions(interactions_with_labels, result_list) == True :
    #print("glycowork and glycontact agree on the list of covalent linkages")

    if check_reconstructed_interactions(interaction_dict) == True :
      #print("Building a network from glycontact interactions generate a single molecule, as expected")

      ### When everything is validated: Annotation including correction of GalNAc annotated as GLC
      #df = correct_dataframe(df)
      result_df = annotate_pdb_data(df, mapping_dict)
    else :
      #print("Although the fragments building binary interactions seem fine, some interactions are missed resulting in the reconstruction of multiple submolecules")
      return(pd.DataFrame(),{})
  else :
    #print("glycowork and glycontact do not agree on the list of covalent linkages in this glycan. It is probable that glycontact encountered a problem with PDB monosaccharide conversion, or detecting linkages")
    return(pd.DataFrame(),{})
  return(result_df,interaction_dict)

def explore_threshold(pdb_file, glycan, threshold_list=[2.2,2.4,2.5,2.6,2.7,2.8,2.9,2.25,2.45,2.55,2.65,2.75,2.85,2.95,3],output='df'):
  # Apply the annotation pipeline with different threshold, and return a correct df if found
  # output can be 'df' to get the annotated df (default), or 'interactions' to  get binary interactions
  #print(glycan)
  completed = False
  for x in threshold_list :
    #print('threshold:' + str(x))
    res, binary_interactions = annotation_pipeline(pdb_file,glycan,x)
    if len(res) != 0 and output == 'df':
      completed = True
      return(res)
    if len(binary_interactions) != 0 and output == 'interactions':
      completed = True
      return(binary_interactions)
  if completed == False :
    #print('None of these thresholds allows to correctly annotate your PDB file:' + str(threshold_list))
    if output == 'df' :
      return(pd.DataFrame())
    else :
      return({})
  


def monosaccharide_preference_structure(df,monosaccharide,threshold, mode='default'):
  #return the preferred partner of a given monosaccharide, except those closer than the threshold (which will be considered as covalent linkages)
  #df must be a monosaccharide distance table correctly reanotated
  #mode can be 'default' (check individual monosaccharides in glycan), 'monolink' (check monosaccharide-linkages in glycan), 'monosaccharide' (check monosaccharide types)
  
  # should the observed frequencies be normalized based on the occurence of each monosaccharide? Indeed, if GlcNAc is often close to Man, is it by choice, or because it is surrounded by so many Man that it has no other choice?
  entities = df.columns.to_list()
  preferred_partners = {}
  preferred_partners_distances = []
  for x in range(0,len(entities)):
    if '(' not in monosaccharide :
      current_mono = entities[x].split('_')[1].split('(')[0]
    if '(' in monosaccharide :
      current_mono = entities[x].split('_')[1]
    if current_mono == monosaccharide :

      distlist = df[entities[x]].to_list()
      shortest_dist = max(distlist)
      for d_index in range(0,len(distlist)) :
        if distlist[d_index] != 0 and  distlist[d_index] >= threshold and distlist[d_index] < shortest_dist :
          shortest_dist =  distlist[d_index]
          closest_index = d_index

      preferred_partners[entities[x]] = entities[closest_index]
      preferred_partners_distances.append(distlist[closest_index])
  if mode =='default':
    return(preferred_partners)
  if mode == 'monolink' :
    monolink_dict = {x:preferred_partners[x].split('_')[1] for x in preferred_partners}
    return(monolink_dict)
  if mode =='monosaccharide' :
    mono_dict = {x:preferred_partners[x].split('_')[1].split('(')[0] for x in preferred_partners}
    return(mono_dict)
  
def show_monosaccharide_preference_structure(df,monosaccharide,threshold, mode='default'):
  #df must be a monosaccharide distance table correctly reanotated
  #mode can be 'default' (check individual monosaccharides in glycan), 'monolink' (check monosaccharide-linkages in glycan), 'monosaccharide' (check monosaccharide types)

  res_dict = monosaccharide_preference_structure(df,monosaccharide,threshold,mode)

  # Count occurrences of each value
  value_counts = Counter(dict(Counter(res_dict.values()).most_common()))

  # Plotting the histogram
  plt.bar(value_counts.keys(), value_counts.values())
  plt.xlabel('Values')
  plt.ylabel('Frequency')
  plt.title('Frequency of Encountered Values for ' + monosaccharide + ' above the distance threshold ' + str(threshold))
  plt.show()

def multi_glycan_monosaccharide_preference_structure(prefix,suffix,glycan_sequence,monosaccharide,threshold, mode='default'):
  ### with multiple dicts accross multiple structures
  # prefix : directory (ex: "PDB_format_ATOM2")
  # suffix : 'alpha' or 'beta'
  # glycan_sequence : IUPAC

  dict_list = []

  for x in range(0,100):
    try :
      print(prefix+"/"+glycan_sequence+"_"+suffix+"_"+str(x)+ ".pdb")
      pdb_file = prefix+"/"+glycan_sequence+"_"+suffix+"_"+str(x) + ".pdb"
      annotated_df = explore_threshold(pdb_file)
      dist_table = make_monosaccharide_contact_table(annotated_df,mode='distance')
      data_dict = monosaccharide_preference_structure(dist_table,monosaccharide,threshold,mode)
      dict_list.append(data_dict)
    except :
      pass

  # Combine values from all dictionaries into a single list
  all_values = [value for d in dict_list for value in d.values()]
  print(all_values)

  # Count occurrences of each value
  value_counts = Counter(dict(Counter(all_values).most_common()))

  # Plotting the histogram
  plt.bar(value_counts.keys(), value_counts.values())
  plt.xlabel('Values')
  plt.ylabel('Frequency')
  plt.title('Frequency of Encountered Values for ' + monosaccharide + ' above the distance threshold ' + str(threshold) + ' across all possible structures given')
  plt.show()

def get_all_clusters_frequency():
  ### Extract all glycan cluster frequencies from glycoshape and returns a dict
    
  # Send a GET request to the URL
  response = requests.get("https://glycoshape.org/database/GLYCOSHAPE.json")

  # Initialize an empty dictionary to store the data
  data_dict = {}

  # Check if the request was successful
  if response.status_code == 200:
      # Parse the JSON data
      data = response.json()

      # Extract the IUPAC sequence and cluster values
      for key, value in data.items():
          iupac_sequence = value["iupac"]
          clusters = [value["clusters"][key] for key in value["clusters"]]
          data_dict[iupac_sequence] = clusters
  else:
      print("Failed to retrieve data from the URL.")

  return(data_dict)

def get_glycan_clusters_frequency(glycan):
  # Extract cluster frequencies for a given glycan
  all_frequencies = get_all_clusters_frequency()
  return(all_frequencies[glycan])

def glycan_cluster_pattern(threshold = 70, mute = False) :
    ### Parse all clusters of all glycans on glycoshape. 
    ### Returns glycans with one major cluster AND glycans with many minor clusters
    ### Classification is performed based on a proportion threshold (default = 70)
    # threshold: proportion that the main cluster must have to be considered as a major cluster
    # if mute is set to True, then the prints are ignored
    # If the proportion of the main cluster is lower, the current glycan is assumed to be represented by multiple structural clusters
    
    all_frequencies = get_all_clusters_frequency()

    glycans_with_major_cluster = []
    glycans_without_major_cluster = []

    for key in all_frequencies :
        try :
            nb_clust = len(all_frequencies[key])
            if float(all_frequencies[key][0]) >= threshold:
                glycans_with_major_cluster.append(key)
            else :
                glycans_without_major_cluster.append(key)
        except:
            pass
    if mute == False :
      print("Number of glycans with one major cluster: " + str(len(glycans_with_major_cluster)))
      print("Number of glycans without a major cluster: " + str(len(glycans_without_major_cluster)))

    return(glycans_with_major_cluster,glycans_without_major_cluster)

def get_sasa_table(my_path, glycan, mode = 'alpha') :
    #mode determines if we are analysing alpha- or beta-linked glycans
    pdbs = check_available_pdb(my_path+glycan)
    if mode == 'alpha':
        pdb_files = [my_path+glycan+"/"+pdb for pdb in pdbs if 'alpha' in pdb]
    if mode == 'beta':
        pdb_files = [my_path+glycan+"/"+pdb for pdb in pdbs if 'beta' in pdb]
    pdb_files.sort()
    sasa_values = {}

    # Loop over PDB files
    for pdb_file in pdb_files:
        # Load structure
        structure = md.load(pdb_file)

        # Calculate SASA for each atom
        sasa = md.shrake_rupley(structure, mode='atom')

        # Calculate the SASA for each monosaccharide
        monosaccharide_sasa = {}
        for atom in structure.topology.atoms:
            resSeq = atom.residue.resSeq
            resName = atom.residue.name
            if resSeq not in monosaccharide_sasa:
                monosaccharide_sasa[resSeq] = {'resName': resName, 'sasa': 0}
            monosaccharide_sasa[resSeq]['sasa'] += sasa[0][atom.index]

        # Store SASA values for this conformation
        sasa_values[pdb_file] = monosaccharide_sasa

    # Calculate accessibility scores and measures of variability for each monosaccharide
    mean_scores = {}
    median_scores = {}
    weighted_scores = {}
    std_dev = {}
    coeff_var = {}
    resNameList = []
    cluster_frequencies = get_glycan_clusters_frequency(glycan)
    for resSeq in sasa_values[pdb_files[0]].keys():
        resName = sasa_values[pdb_files[0]][resSeq]['resName']
        resNameList.append(resName)
        monosaccharide_sasa_values = [sasa_values[pdb_file][resSeq]['sasa'] for pdb_file in pdb_files]
        mean_scores[resSeq] = np.mean(monosaccharide_sasa_values)
        median_scores[resSeq] = np.median(monosaccharide_sasa_values)
        weights = [n / 100 for n in cluster_frequencies]
        weighted_scores[resSeq] = np.average(monosaccharide_sasa_values, weights=weights, axis=0)
        std_dev[resSeq] = np.std(monosaccharide_sasa_values)
        coeff_var[resSeq] = np.std(monosaccharide_sasa_values) / np.mean(monosaccharide_sasa_values)

    # Generate final table with all monosaccharides and their accessibility scores and measures of variability
    table = pd.DataFrame({'Monosaccharide_id': list(mean_scores.keys()),
                        'Monosaccharide': list(resNameList),
                        'Mean Score': list(mean_scores.values()),
                        'Median Score': list(median_scores.values()),
                        'Weighted Score': list(weighted_scores.values()),
                        'Standard Deviation': list(std_dev.values()),
                        'Coefficient of Variation': list(coeff_var.values())})
    
    # get a mapping dict for annotations

    pdb = pdb_files[0]
    df = explore_threshold(pdb,glycan, threshold_list=[2.4,2.5,2.6,2.7,2.8,2.9,2.45,2.55,2.65,2.75,2.85,2.95,3])
    dist_table = make_monosaccharide_contact_table(df,mode='distance', threshold = 200)
    mapping_dict = df.set_index('residue_number')['IUPAC'].to_dict()
    table['Monosaccharide'] = table['Monosaccharide_id'].map(mapping_dict)
    
    return(table)

def convert_glycan_to_X(glycan):
    """
    Converts every monosaccharide(linkage) and single monosaccharide into 'X' in a glycan string.
    
    Parameters:
    - glycan (str): A string representing the glycan in IUPAC format.
    
    Returns:
    - str: The modified glycan string with each monosaccharide replaced by 'X'.
    """
    # Regular expression to match monosaccharide(linkage) or single monosaccharide
    pattern = r'[A-Za-z0-9]+(?:\([^\)]+\))?'

    # Replace each matched pattern with 'X'
    converted_glycan = re.sub(pattern, 'X', glycan)

    return converted_glycan

def convert_glycan_to_class(glycan):
    """
    Converts every monosaccharide(linkage) and single monosaccharide into X, XNAc,XA, XN, dX, Sia, Pen in a glycan string.
    
    Parameters:
    - glycan (str): A string representing the glycan in IUPAC format.
    
    Returns:
    - str: The modified glycan string with each monosaccharide replaced by 'X'.
    """
    # Regular expression to match monosaccharide(linkage) or single monosaccharide
    Hex = ['Glc', 'Gal', 'Man', 'Ins', 'Galf', 'Hex']
    dHex = ['Fuc', 'Qui', 'Rha', 'dHex']
    HexA = ['GlcA', 'ManA', 'GalA', 'IdoA', 'HexA']
    HexN = ['GlcN', 'ManN', 'GalN', 'HexN']
    HexNAc = ['GlcNAc', 'GalNAc', 'ManNAc', 'HexNAc']
    Pen = ['Ara', 'Xyl', 'Rib', 'Lyx', 'Pen']
    Sia = ['Neu5Ac', 'Neu5Gc', 'Kdn', 'Sia']

    glycan = stemify_glycan(glycan)
   
    mono_list = glycan.split(')')
    mono_list = [element.split('(')[0] if '(' in element else element for element in mono_list]
    m_list = []
    for m in mono_list :
      if '[' not in m and ']' not in m :
        m_list.append(m)
      if '[' in m :
        m_list.append('[')
        m_list.append(m.split('[')[1])
      if ']' in m :
        m_list.append(']')
        m_list.append(m.split(']')[1])
        
    silhouette = ''
    
    for element in m_list :

      if element in Hex :
        silhouette = silhouette + 'X'
      if element in dHex :
        silhouette = silhouette + 'dX'
      if element in HexA :
        silhouette = silhouette + 'XA'
      if element in HexN :
        silhouette = silhouette + 'XN'
      if element in HexNAc :
        silhouette = silhouette + 'XNAc'
      if element in Pen :
        silhouette = silhouette + 'Pen'
      if element in Sia :
        silhouette = silhouette + 'Sia'
      if element not in Hex+dHex+HexA+HexN+HexNAc+Pen+Sia :
        if element == '[' :
          silhouette = silhouette + '['
        if element == ']' :
          silhouette = silhouette + ']'
        if element not in  ['[', ']'] : 
          silhouette = silhouette + 'Unk'

    return silhouette

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
    silhouettes = pd.DataFrame()
    topo_groups = [] # groups of same topology/silhouette
    nullified_list = []
    group_list = []

    for g in glycan_list :
        if mode == 'X' :
          nullified = convert_glycan_to_X(g)
        if mode == 'class' :
          nullified = convert_glycan_to_class(g)
        if nullified in topo_groups :
            group = topo_groups.index(nullified)
        else :
            topo_groups.append(nullified)
            group = topo_groups.index(nullified)
        nullified_list.append(nullified)
        group_list.append(group)

    silhouettes['glycan']=glycan_list
    silhouettes['silhouette']=nullified_list
    silhouettes['topological_group']=group_list

    return silhouettes.sort_values(by ='topological_group')


def global_monosaccharide_unstability(variability_table, mode='sum'):
    # plot monolink variability for all clusters of a given glycan
    # possible formats: png, pdf
    # mode: sum, mean
    residue_overall_stability = {}
    for c in variability_table.columns.to_list():
        if mode == 'sum':
            residue_overall_stability[c] = sum(variability_table[c].to_list())
        if mode == 'mean':
            residue_overall_stability[c] = sum(variability_table[c].to_list())


    sorted_residue_overall_stability = sorted(residue_overall_stability.items(), key=lambda x:x[1])
    return(sorted_residue_overall_stability)

def compute_merge_SASA_flexibility(mypath, glycan, flex_mode, global_flex_mode) :
    # flex_mode : standard, amplify, weighted
    # global_flex_mode : sum, mean
    
    try :
        sasa = get_sasa_table(mypath,glycan,'beta')
    except :
        print('SASA failed, lets continue with empty table')
        try : 
            flex = inter_structure_variability_table(mypath,glycan,'beta', mode=flex_mode)
            print(flex)
            mean_flex = global_monosaccharide_unstability(flex,mode=global_flex_mode)
            global_flexibility_df = pd.DataFrame(mean_flex, columns=['Monosaccharide_id_Monosaccharide', flex_mode+'_'+global_flex_mode+'_flexibility'])
            global_flexibility_df['Monosaccharide_id'] = global_flexibility_df['Monosaccharide_id_Monosaccharide'].str.split('_').str[0].astype(int)
        except :
            print('Both SASA and Flexibility failed, lets return an empty table then...')
            merged_df = pd.DataFrame({'Monosaccharide_id': [],
                            'Monosaccharide': [],
                            'Mean Score': [],
                            'Median Score': [],
                            'Weighted Score': [],
                            'Standard Deviation': [],
                            'Coefficient of Variation': [],
                            flex_mode+'_'+global_flex_mode+'_flexibility': []})
            return(merged_df)
            
    try : 
        flex = inter_structure_variability_table(mypath,glycan,'beta', mode=flex_mode)
        mean_flex = global_monosaccharide_unstability(flex,mode=global_flex_mode)
        #print(mean_flex)
    
        global_flexibility_df = pd.DataFrame(mean_flex, columns=['Monosaccharide_id_Monosaccharide', flex_mode+'_'+global_flex_mode+'_flexibility'])
        # Parse the Monosaccharide_id from the string
        global_flexibility_df['Monosaccharide_id'] = global_flexibility_df['Monosaccharide_id_Monosaccharide'].str.split('_').str[0].astype(int)
    except :
        print("Flex failed")
        global_flexibility_df = pd.DataFrame(columns=['Monosaccharide_id', flex_mode+'_'+global_flex_mode+'_flexibility'])
    # Step 3: Merge the two DataFrames on Monosaccharide_id
    merged_df = pd.merge(sasa, global_flexibility_df[['Monosaccharide_id', flex_mode+'_'+global_flex_mode+'_flexibility']], on='Monosaccharide_id', how='left')

    # Display the merged DataFrame
    return(merged_df)

def map_data_to_graph(computed_df, interaction_dict) :
    # map the interaction dict to SASA/Flex values computed to produce a graph with node-level information

    # Simplify data by keeping only the numbers before "_"
    simplified_edges = set()
    for key, values in interaction_dict.items():
        key_num = key.split('_')[0]  # Extract number before "_"
        for value in values:
            value_num = value.split('_')[0]  # Extract number before "_"
            # Add to set if the numbers are different
            if key_num != value_num:
                simplified_edges.add((int(key_num), int(value_num)))

    # Create a graph with networkx
    G = nx.Graph()
    G.add_edges_from(simplified_edges)


    df = computed_df
    # Add node attributes from the DataFrame
    for _, row in df.iterrows():
        node_id = row['Monosaccharide_id']
        # Add attributes from the DataFrame row
        try :
          G.nodes[node_id]['Monosaccharide'] = row['Monosaccharide']
        except :
          G.nodes[node_id]['Monosaccharide'] = node_id
        try :
          G.nodes[node_id]['Mean Score'] = row['Mean Score']
          G.nodes[node_id]['Median Score'] = row['Median Score']
          G.nodes[node_id]['Weighted Score'] = row['Weighted Score']
        except :
           print("Expected columns from a correct SASA table are missing")
        try :
          G.nodes[node_id]['weighted_mean_flexibility'] = row['weighted_mean_flexibility']
        except :
           print("Expected columns from a correct flexibility table are missing")
    
    return(G)

def check_graph_content(G) : 
    # Print the nodes and their attributes
    print("Graph Nodes and Their Attributes:")
    for node, attributes in G.nodes(data=True):
        print(f"Node {node}: {attributes}")

    # Print the edges
    print("\nGraph Edges:")
    for edge in G.edges():
        print(edge)

def get_score_list(datatable, my_glycans_path, glycan, mode, column):
    #try to extract score in the same order as glycan string to ensure glycoDraw will plot them correctly
    # datatable is either a SASA table, a flex table, or a merged table

    score_list = datatable[column].to_list()[::-1]
    mono_order = datatable['Monosaccharide'].to_list()[::-1] 
    g_mono_order = g.replace('[','').replace(']','').split(')')[:-1]
    g_mono_order = [m+')' if '(' in m else m for m in g_mono_order]


    new_score_list = []
    for x in range (0,len(g_mono_order)):
        if g_mono_order[x] == mono_order[x]:
            new_score_list.append(score_list[x])
        if g_mono_order[x] != mono_order[x] and g_mono_order[x] == mono_order[x+1] :
            new_score_list.append(score_list[x+1])
            new_score_list.append(score_list[x])
            x +=1
        
    
    new_score_list.append(score_list[x+1])
    new_score_list.append(score_list[x+2])

    if len(new_score_list) == len(score_list):
        return(new_score_list)
    if len(new_score_list) != len(score_list):
        return(score_list)