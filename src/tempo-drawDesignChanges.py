import pandas as pd
import os
import json
import shutil
from const_project import DIRS_DATA, DIRS_OUT_RES, FILE_INIT_RVT

# U1_OK_d_wl_ew7 = U1_OK_d_wl_ew35 - 5,765 m

res_plot_nr = [0, 1488, 1519, 1249, 1522, 1749, 669, 1271]
sampling_data = os.path.join(DIRS_DATA, 'ss-134-2', 'all_data.csv')

df = pd.read_csv(sampling_data)

# Define the index filter list
res_plot_nr = [0, 1488, 1519, 1249, 1522, 1749, 669, 1271]

# Filter rows by index (rows are manually selected now for result demonstration).
filtered_df = df.loc[res_plot_nr]
filtered_df = filtered_df.round(3)

# Define the path for the new folder and create the folder if it doesn't already exist.
for index  in res_plot_nr:

    # make the subfoloders
    subfolder_path = os.path.join(DIRS_OUT_RES, str(index))
    os.makedirs(subfolder_path, exist_ok=True)

    # Copy the initial design to the folder with the index number as its name
    variant_file_path = os.path.join(subfolder_path, f"{index}{os.path.splitext(FILE_INIT_RVT)[1]}")
    shutil.copy(FILE_INIT_RVT, variant_file_path)

    # Create a JSON file with the values from the row of the DataFrame
    row_data = filtered_df.loc[index].to_dict()  # Convert row to dictionary format
    row_data.pop('Unnamed: 0', None)
    row_data_with_index = {'designnumber': index, **row_data}  # Add 'designnumber' as index key
    
    json_file_path = os.path.join(subfolder_path, f"{index}.json")
    
    # Write the JSON file
    with open(json_file_path, 'w') as json_file:
        json.dump(row_data_with_index, json_file, indent=4)


