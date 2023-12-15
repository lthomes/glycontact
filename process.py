import pandas as pd
import numpy as np

def extract_3D_coordinates(pdb_file):
    ### Take a pdb file to extract the coordinates and return a dataframe
    tmp = open('tmp.txt','w')
    pdb_f = open(pdb_file,'r')
    for l in pdb_f :
        if 'ATOM   ' in l and 'REMARK' not in l:
            tmp.write(l)
    tmp.close()
    pdb_f.close()
    columns = ['record_name', 'atom_number', 'atom_name', 'monosaccharide', 'chain_id', 'residue_number', 'x', 'y', 'z', 'occupancy', 'temperature_factor', 'element']

    # Read the PDB file into a DataFrame
    df = pd.read_fwf('tmp.txt', names=columns, colspecs=[(0, 6), (6, 11), (12, 16), (17, 20), (21, 22), (22, 26), (30, 38), (38, 46), (46, 54), (54, 60), (60, 66), (76, 78)])

    # Display the DataFrame
    return(df)

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