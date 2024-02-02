import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import re
from collections import Counter
import subprocess
import json
import shutil
from urllib.parse import quote


def get_glycoshape_IUPAC() :
    #get the list of available glycans on glycoshape
    curl_command = 'curl -X GET https://glycoshape.io/api/available_glycans'
    x = subprocess.run(curl_command, shell=True, capture_output=True, text=True)
    parsed_dict = json.loads(x.stdout)
    return(parsed_dict['glycan_list'])

def download_from_glycoshape(IUPAC) :
    #download pdb files given a IUPAC sequence that exists in the glycoshape database

    outpath = IUPAC
    IUPAC_name = quote(IUPAC)
    os.makedirs(outpath, exist_ok=True)
    for linktype in ['alpha','beta'] :
        for i in range(0,500) :

            output = linktype + '_' + str(i) +'.pdb'

            # Construct the curl command with string formatting
            curl_command = f'curl -o {output} "https://glycoshape.io/database/{IUPAC_name}/PDB_format_ATOM/{IUPAC_name}_cluster{i}_{linktype}.PDB.pdb"'
            tiny_command = f'curl "https://glycoshape.io/database/{IUPAC_name}/PDB_format_ATOM/{IUPAC_name}_cluster{i}_{linktype}.PDB.pdb"'

            try : 
                # Use subprocess to run the command and capture the output
                result = subprocess.run(tiny_command, shell=True, capture_output=True, text=True)

                if "404 Not Found" in result.stdout:
                    return  # Stop the function

                result = subprocess.run(curl_command, shell=True, capture_output=True, text=True)
                # Specify the current and new file names
                current_file_name = output
                new_file_name = IUPAC + current_file_name

                # Rename the file
                os.rename(current_file_name, new_file_name)
                shutil.move(new_file_name, outpath)

            except :
                return 

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
    mono_list = coord_df['monosaccharide'].to_list()
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

    return(distanceMap)

def make_monosaccharide_contact_table(coord_df, threshold = 10, mode = 'binary') :
    # threshold : maximal distance to be considered. Otherwise set to threshold + 1
    # mode : can be either binary (return a table with 0 or 1 based on threshold), distance (return a table with the distance or threshold+1 based on threshold), or both
    distanceMap = pd.DataFrame()
    distanceMap2 = pd.DataFrame()

    atom_list = coord_df['atom_name'].to_list()
    anum_list = coord_df['atom_number'].to_list()
    mono_list = coord_df['monosaccharide'].to_list()
    num_list = coord_df['residue_number'].to_list()


    for i in list(set(num_list)) :
        ndf = coord_df[coord_df['residue_number']==i]
        current_pos = str(i) + '_' + ndf['monosaccharide'].to_list()[0]
        distanceList = []
        distanceList2 = []
        
        x_list = ndf['x'].to_list()
        y_list = ndf['y'].to_list()
        z_list = ndf['z'].to_list()

        for j in list(set(num_list)) :
            adverse_df = coord_df[coord_df['residue_number']==j]
            adverse_pos = str(j) + '_' + adverse_df['monosaccharide'].to_list()[0]
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

def inter_structure_variability_table(dfs, mode = 'standard'):
    ### Creates a table as make_atom_contact_table() or the monosaccharide equivalent but values represent the stability of monosaccharides/atoms across different PDB of the same molecule
    # dfs : list of dataframes being tables returned by make_atom_contact_table() or make_monosaccharide_contact_table()
    # mode : can be 'standard' (compute the sum of the absolute distances to the mean) or 'amplify' (uses the power 2 of the sum which decreases noise and increases outliers importance)

    col_to_parse = dfs[0].columns.to_list()

    outdf = pd.DataFrame(columns=col_to_parse)
    outdf_power = pd.DataFrame(columns=col_to_parse)
    for col_index in range(0,len(col_to_parse)):
        current_column = col_to_parse[col_index]
        new_column = []
        new_column2 = []
        list_of_values_lists = []
        for df_index in range(0,len(dfs)): 
            list_of_values_lists.append(dfs[df_index][current_column].to_list()) 
        for y in range(0,len(list_of_values_lists[0])) : #
            values = []
            mean = 0
            deviation_from_mean = []
            sum_of_deviations = 0
            power_of_deviations = 0
            
            for liste in list_of_values_lists :
                values.append(liste[y])
            mean = np.mean(values)
            for v in values :
                deviation_from_mean.append(abs(v - mean))
            sum_of_deviations = sum(deviation_from_mean)
            power_of_deviations = sum_of_deviations**2
            new_column.append(sum_of_deviations)
            new_column2.append(power_of_deviations)

        outdf[current_column] = new_column
        outdf_power[current_column] = new_column2
    if mode == 'standard' :
        return(outdf)
    if mode == 'amplify' :
        return(outdf_power)

def make_correlation_matrix(dfs):
    ### Take a list of dataframes (dfs) containing inter-monosaccharide/atom distances across different iterations of the same glycan to compute a Pearson correlation matrix

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

def inter_structure_frequency_table(dfs, threshold = 5):
    ### Creates a table as make_atom_contact_table() or the monosaccharide equivalent but values represent the frequency of monosaccharides/atoms pairs that crossed a threshold distance across different PDB of the same molecule
    # dfs : list of dataframes being tables returned by make_atom_contact_table() or make_monosaccharide_contact_table()
    # threshold : maximal distance a pair can show to be counted as a contact

    # Apply thresholding and create a new list of transformed DataFrames
    transformed_dfs = [df.applymap(lambda x: 1 if x < threshold else 0) for df in dfs]
    # Sum up the transformed DataFrames to create the final DataFrame
    final_df = pd.DataFrame(sum(transformed_dfs))
    return(final_df)

def extract_binary_interactions_from_PDB(df, threshold):
    """
    Extract binary interactions between C1-2 atoms and oxygen atoms from a DataFrame obtained using extract_3D_coordinates.

    Parameters:
    - df (pd.DataFrame): DataFrame containing 3D coordinates obtained from extract_3D_coordinates.
    - threshold (float): Distance threshold for considering interactions.

    Returns:
    - pd.DataFrame: DataFrame containing information about binary interactions.
    """

    # Extract sub-DataFrames: one with only C1-2 atoms, the other with only oxygens
    carbon_1_2_df = df[(df['atom_name'] == 'C1') | (df['atom_name'] == 'C2')]
    oxygen_df = df[df['element'] == 'O']

    # Create dictionaries to store coordinates of C1-2 atoms and oxygen atoms
    c_dict = {}
    for x in range(len(carbon_1_2_df)):
        key = f"{carbon_1_2_df['residue_number'].iloc[x]}_{carbon_1_2_df['monosaccharide'].iloc[x]}_{carbon_1_2_df['atom_name'].iloc[x]}"
        c_dict[key] = [carbon_1_2_df['x'].iloc[x], carbon_1_2_df['y'].iloc[x], carbon_1_2_df['z'].iloc[x]]

    o_dict = {}
    for x in range(len(oxygen_df)):
        key = f"{oxygen_df['residue_number'].iloc[x]}_{oxygen_df['monosaccharide'].iloc[x]}_{oxygen_df['atom_name'].iloc[x]}"
        o_dict[key] = [oxygen_df['x'].iloc[x], oxygen_df['y'].iloc[x], oxygen_df['z'].iloc[x]]

    # Lists to store results
    atom = []
    column = []
    value = []

    # Iterate through C1-2 atoms and find closest oxygen atoms
    for key in c_dict:
        smallest_distance = 1000
        closest_residue = ''
        c_x, c_y, c_z = c_dict[key]
        c_resnum = key.split('_')[0]

        # Iterate through oxygen atoms
        for okey in o_dict:
            o_resnum = okey.split('_')[0]

            # Check if the atoms belong to different residues
            if c_resnum != o_resnum:
                delta_x = abs(c_x - o_dict[okey][0])
                delta_y = abs(c_y - o_dict[okey][1])
                delta_z = abs(c_z - o_dict[okey][2])
                sum_dist = delta_x + delta_y + delta_z

                # Check if the current oxygen atom is closer than previous closest
                if sum_dist < smallest_distance:
                    smallest_distance = sum_dist
                    closest_residue = okey

        # Check if the smallest distance is below the threshold
        if smallest_distance < threshold:
            print(f"{key} linked to {closest_residue} by length: {smallest_distance}")

            # Append results to lists
            atom.append(key)
            column.append(closest_residue)
            value.append(smallest_distance)

    # Create a DataFrame from the results
    resdf = pd.DataFrame({'Atom': atom, 'Column': column, 'Value': value})
    return resdf

def get_glycan_sequence_from_path(pdb_file) :
  # Simply extract the glycan sequence from path and filename
  seq = pdb_file.split('/')[-1].split('_')[0]
  return(seq)

def extract_numbers(input_string):
    # Use regular expression to extract numbers
    numbers = re.findall(r'\d+', input_string)

    # Join the extracted numbers into a single string
    result = ''.join(numbers)

    return result

def PDB_to_IUPAC(pdb_mono):
  map_dict = {'NDG':'GlcNAc(a','NAG':'GlcNAc(b','MAN':'Man(a', 'BMA':'Man(b', 'AFL':'Fuc(a',
              'FUC':'Fuc(a', 'FUL':'Fuc(b', 'FCA':'dFuc(a', 'FCB':'dFuc(b', 'GYE':'dFucf(b',
              'GAL':'Gal(b', 'GLA':'Gal(a', 'GIV':'lGal(b', 'GXL':'lGal(a', 'GZL':'Galf(b',
              'GLC':'Glc(a', 'IDR':'IdoA(a', 'RAM':'Rha(a', 'RM4':'Rha(b', 'XXR':'dRha(a',
              'A2G':'GalNAc(a', 'NGA': 'GalNAc(b', 'YYQ':'lGlcNAc(a', 'XYP':'Xyl(b', 'XYS':'Xyl(a',
              'XYZ':'Xylf(b', 'LXC':'lXyl(b', 'HSY':'lXyl(a', 'SIA':'Neu5Ac(a', 'SLB':'Neu5Ac(b',
              'NGC':'Neu5Gc(a', 'NGE':'Neu5Gc(b', 'BDP':'GlcA(b', 'GCU':'GlcA(a', 'GCS':'GlcN(b', 'PA1':'GlcN(a',
              'ROH':' '}
  # MAN indicates either alpha and beta bonds, instead of just alpha.. this is a problem
  # GalNAc is recorded as "GLC" which is wrong: need for a checker function that counts the number of atoms - Glc = 21 (<25), GalNAc = 28 (>25)
  mono_core = map_dict[pdb_mono.split('_')[1]]
  return(mono_core)

def create_mapping_dict_and_interactions(df, valid_fragments) :
  #df is an interaction dataframe as returned by extract_binary_interactions_from_PDB()
  # valid_fragments : obtained from glycowork to ensure that we only append valid monolinks into mapping dict
  mapping_dict = {}
  interaction_dict = {}
  interaction_dict2 = {}
  first_mono_list = df['Atom'].to_list()
  second_mono_list = df['Column'].to_list()
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

    if mono.split('_')[1] + '(' + first_val + '-' + last_val + ')' == "MAN(1-4)":
      wrong_mannose.append(mono)

    if second_mono in wrong_mannose :
      second_mono = second_mono.split('_')[0]+"_BMA"

    if mono in wrong_mannose :
      mono = mono.split('_')[0]+"_BMA"
    mapped_to_check = PDB_to_IUPAC(mono) + first_val + '-' + last_val + ')'

    if mapped_to_check in valid_fragments :
      mapping_dict[mono] = PDB_to_IUPAC(mono) + first_val + '-' + last_val + ')'
    if mapped_to_check == 'Man(a1-4)':
      mapping_dict[mono] = 'Man(b1-4)'
    if mapped_to_check == 'GlcNAc(a1-1)' or mapped_to_check == 'GlcNAc(b1-1)' or mapped_to_check == '-R' :
      mapping_dict[mono] = mapped_to_check


    if mono in interaction_dict :
      if second_mono not in interaction_dict[mono] :
        interaction_dict[mono].append(second_mono)
        interaction_dict2[mono] = [mono.split('_')[0]+'_(' + PDB_to_IUPAC(mono).split('(')[1] + first_val + '-' + last_val + ')']
        interaction_dict2[mono.split('_')[0]+'_(' + PDB_to_IUPAC(mono).split('(')[1] + first_val + '-' + last_val + ')'] = [second_mono] #added but eventually wrong, make everything else fail later
    if mono not in interaction_dict :
      if mapped_to_check in valid_fragments or mapped_to_check == 'Man(a1-4)' or mapped_to_check == '-R' or mapped_to_check == 'GlcNAc(a1-1)' or mapped_to_check == 'GlcNAc(b1-1)':
        interaction_dict[mono] = [second_mono]
        interaction_dict2[mono] = [mono.split('_')[0]+'_(' + PDB_to_IUPAC(mono).split('(')[1] + first_val + '-' + last_val + ')']

      if mono.split('_')[0]+'_(' + PDB_to_IUPAC(mono).split('(')[1] + first_val + '-' + last_val + ')' in interaction_dict2 :
        if mapped_to_check in valid_fragments or mapped_to_check == 'Man(a1-4)' or mapped_to_check == '-R' or mapped_to_check == 'GlcNAc(a1-1)' or mapped_to_check == 'GlcNAc(b1-1)':
          interaction_dict2[mono.split('_')[0]+'_(' + PDB_to_IUPAC(mono).split('(')[1] + first_val + '-' + last_val + ')'].append(second_mono)

      if mono.split('_')[0]+'_(' + PDB_to_IUPAC(mono).split('(')[1] + first_val + '-' + last_val + ')' not in interaction_dict2 :
        if mapped_to_check in valid_fragments or mapped_to_check == 'Man(a1-4)' or mapped_to_check == '-R' or mapped_to_check == 'GlcNAc(a1-1)' or mapped_to_check == 'GlcNAc(b1-1)' :
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
  ignore_pairs = {('GlcNAc', 'a1-1'), ('a1-1', ' '),('GlcNAc', 'b1-1'), ('b1-1', ' ')}

  # Filter out pairs to be ignored
  filtered_differences = [pair for pair in differences_list if pair not in ignore_pairs]

  # Print or use the filtered_differences as needed
  print("Filtered Differences:", filtered_differences)
  if filtered_differences == [] and  (len(glycontact_interactions) > len(glycowork_interactions)):
    return(True)
  else :
    if filtered_differences != [] :
      print('Differences in annotations')
      print(glycowork_interactions)
      print(glycontact_interactions)
      return(False)
    if (len(glycontact_interactions) <= len(glycowork_interactions)) :
      print("Missing monosaccharide in mapping_dict")
      return(False)

def check_reconstructed_interactions(interaction_dict) :
  # Use the interaction_dict to build a NetworkX network and check that it contains a single component (meaning that the glycan has been correctly reconstructed from the PDB file)

  # Create a directed graph
  G_dict = nx.Graph()

  # Add nodes and edges from dictionary interactions
  for node, neighbors in interaction_dict.items():
    G_dict.add_node(node)
    G_dict.add_edges_from((node, neighbor) for neighbor in neighbors)

  # Draw and display the graphs
  plt.figure(figsize=(10, 5))

  plt.subplot(122)
  nx.draw(G_dict, with_labels=True, font_weight='bold', node_color='lightcoral', arrowsize=20)
  plt.title('Graph from Dictionary Interactions')

  plt.show()

  is_single_component = nx.is_connected(G_dict)

  if is_single_component:
      print("The graph has only one connected component.")
      return(True)
  else:
      print("The graph has more than one connected component.")
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
  #Correct an annotated dataframe, transforming unexpected GLC into GalNAc based on the number of atom they contain
  resnum = list(set(df['residue_number'].tolist()))

  for x in resnum:
    condition = (df['monosaccharide'] == 'GLC') & (df['residue_number'] == x) & (len(df[df['residue_number'] == x]) > 22)

    if condition.any():
        print(len(df[df['residue_number'] == x]))
        df.loc[condition, 'IUPAC'] = df.loc[condition, 'IUPAC'].map(lambda x: x.replace('Glc', 'GalNAc'))

  return df

def annotation_pipeline(pdf_file,threshold =2.7) :
  ### Huge function combining all smaller ones required to annotate a PDB file into IUPAC nomenclature, ensuring that the conversion is correct
  ### It allows also to determine if PDB to IUPAC conversion at the monosaccharide level works fine

  ### Extract glycan sequence from filename
  glycan_sequence = get_glycan_sequence_from_path(pdb_file)
  print(glycan_sequence)

  ### Using glycowork, extract valid fragments (fragment = monolink like GlcNAc(b1-4))
  valid_fragments = [x.split(')')[0]+')' for x in link_find(glycan_sequence)]
  print(valid_fragments)

  ### Detect binary connections (covalent linkages) using a maximal distance threshold and valid_fragments + build a mapping dictionnary
  res = extract_binary_interactions_from_PDB(pdb_file,threshold)
  mapping_dict, interaction_dict = create_mapping_dict_and_interactions(res,valid_fragments)
  print(mapping_dict)
  print(len(mapping_dict))
  print(len(interaction_dict))

  ### Comparison of glycowork linkages and glycontact linkages to ensure correct extraction from PDB
  # Extract glycowork interactions:
  graph_output = glycan_to_graph(glycan_sequence)
  interactions_with_labels = extract_binary_glycowork_interactions(graph_output)

  # Extract glycontact interactions:
  result_list = extract_binary_glycontact_interactions(interaction_dict)
  print(result_list)
  # Compare glycowork IUPAC to graph versus glycontact PDB to graph to ensure glycontact detection of covalent linkages is correct (must return True)
  if glycowork_vs_glycontact_interactions(interactions_with_labels, result_list) == True :
    print("glycowork and glycontact agree on the list of covalent linkages")

    if check_reconstructed_interactions(interaction_dict) == True :
      print("Building a network from glycontact interactions generate a single molecule, as expected")

      ### When everything is validated: Annotation
      df = extract_3D_coordinates(pdb_file)
      annotated_df = annotate_pdb_data(df, mapping_dict)

      # Correction of GalNAc incorrectly annotated as GLC in PDB
      result_df = correct_dataframe(annotated_df)

    else :
      print("Although the fragments building binary interactions seem fine, some interactions are missed resulting in the reconstruction of multiple submolecules")
      return(pd.DataFrame())
  else :
    print("glycowork and glycontact do not agree on the list of covalent linkages in this glycan. It is probable that glycontact encountered a problem with PDB monosaccharide conversion, or detecting linkages")
    return(pd.DataFrame())
  return(result_df)

def explore_threshold(pdb_file, threshold_list=[2.2,2.4,2.5,2.6,2.7,2.8,2.9,2.25,2.45,2.55,2.65,2.75,2.85,2.95,3]):
  # Apply the annotation pipeline with different threshold, and return a correct df if found

  completed = False
  for x in threshold_list :
    print('threshold:' + str(x))
    res = annotation_pipeline(pdb_file,x)
    if len(res) != 0 :
      completed = True
      return(res)
  if completed == False :
    print('None of these thresholds allows to correctly annotate your PDB file:' + str(threshold_list))
    return(pd.DataFrame())

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
  print(np.mean(preferred_partners_distances))
  if mode =='default':
    return(preferred_partners)
  if mode == 'monolink' :
    monolink_dict = {x:preferred_partners[x].split('_')[1] for x in preferred_partners}
    return(monolink_dict)
  if mode =='monosaccharide' :
    mono_dict = {x:preferred_partners[x].split('_')[1].split('(')[0] for x in preferred_partners}
    return(mono_dict)