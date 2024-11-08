from const_project import DIRS_DATA, DIRS_OUT_RES, FILE_INIT_RVT

import pandas as pd
import os
import json
import shutil

import cv2
from pdf2image import convert_from_path

def crop_img(img, c_t=0, c_b=0, c_l=0, c_r=0):
    """Crop image by percentage ratios from top, bottom, left, and right."""
    im_h, im_w = img.shape[:2]

    # Convert percentages to absolute pixel values
    top = int(im_h * c_t)
    bottom = int(im_h * c_b)
    left = int(im_w * c_l)
    right = int(im_w * c_r)

    # Crop the image
    crop_im = img[top:im_h - bottom, left:im_w - right]
    return crop_im

# U1_OK_d_wl_ew7 = U1_OK_d_wl_ew35 - 5,765 m
res_plot_nr = [0, 1488, 1519, 1249, 1522, 1749, 669, 1271]
sampling_data = os.path.join(DIRS_DATA, 'ss-134-2', 'all_data.csv')

# Filter rows by index (rows are manually selected now for result demonstration).
df = pd.read_csv(sampling_data)
filtered_df = df.loc[res_plot_nr]
filtered_df = filtered_df.round(3)

# # ========================================
# # Step1: producing the design alternatives.
# # ========================================

# # Define the path for the new folder and create the folder if it doesn't already exist.
# for index in res_plot_nr:

#     # make the subfoloders
#     subfolder_path = os.path.join(DIRS_OUT_RES, str(index))
#     os.makedirs(subfolder_path, exist_ok=True)

#     # Copy the initial design to the folder with the index number as its name
#     variant_file_path = os.path.join(subfolder_path, f"{index}{os.path.splitext(FILE_INIT_RVT)[1]}")
#     shutil.copy(FILE_INIT_RVT, variant_file_path)

#     # Create a JSON file with the values from the row of the DataFrame
#     row_data = filtered_df.loc[index].to_dict()  # Convert row to dictionary format
#     row_data.pop('Unnamed: 0', None)
#     row_data_with_index = {'designnumber': index, **row_data}  # Add 'designnumber' as index key
    
#     json_file_path = os.path.join(subfolder_path, f"{index}.json")
    
#     # Write the JSON file
#     with open(json_file_path, 'w') as json_file:
#         json.dump(row_data_with_index, json_file, indent=4)

# # ========================================
# # Step2: PDF2PNG
# # ========================================

# for index in res_plot_nr:
#     subfolder_path = os.path.join(DIRS_OUT_RES, str(index))
    
#     # Find the PDF file in the subfolder
#     pdf_path = next((f for f in os.listdir(subfolder_path) if f.endswith('.pdf')), None)
#     if pdf_path is None:
#         print(f"No PDF found in {subfolder_path}")
#         continue
    
#     pdf_full_path = os.path.join(subfolder_path, pdf_path)

#     # Convert PDF to PNG for the single page
#     page = convert_from_path(pdf_full_path, dpi=500)[0]  # Only take the first page
    
#     # Save the PNG file with the same base name as the PDF
#     png_path = os.path.join(subfolder_path, f"{os.path.splitext(pdf_path)[0]}.png")
#     page.save(png_path, 'PNG')

# # ========================================
# # Step3: CROPPING
# # ========================================
# C_T = 0.045
# C_B = 0.650
# C_L = 0.045
# C_R = 0.380

# for index in res_plot_nr:
    
#     # Find the PNG file in the subfolder
#     subfolder_path = os.path.join(DIRS_OUT_RES, str(index))
#     png_path = next((f for f in os.listdir(subfolder_path) if f.endswith('.png') and not f.endswith('_cropped.png')), None)
#     if png_path is None:
#         print(f"No PNG found in {subfolder_path} for cropping.")
#         continue
    
#     png_full_path = os.path.join(subfolder_path, png_path)
#     img = cv2.imread(png_full_path)
#     cropped_img = crop_img(img, c_t=C_T, c_b=C_B, c_l=C_L, c_r=C_R)
#     cropped_png_path = os.path.join(subfolder_path, f"{os.path.splitext(png_path)[0]}_cropped.png")
#     cv2.imwrite(cropped_png_path, cropped_img)

# ========================================
# Step4: convert the labelled pdfs to pngs  + do recutting.
# ========================================

# labelled_pdf_path = os.path.join(DIRS_OUT_RES, 'summary.pdf')
# pages = convert_from_path(labelled_pdf_path, dpi=500)

# for i, page in enumerate(pages):
#     if i < len(res_plot_nr):
#         nr = res_plot_nr[i]
#         png_path = os.path.join(DIRS_OUT_RES, f"labelled_{nr}.png")
#         page.save(png_path, 'PNG')
#     else:
#         print(f"Warning: More pages in PDF than labels in res_plot_nr. Skipping page {i}.")

C_T = 0.00
C_B = 0.03
C_L = 0.20
C_R = 0.20
for nr in res_plot_nr:

    # Define the path of each PNG file
    png_path = os.path.join(DIRS_OUT_RES, f"labelled_{nr}.png")
    if not os.path.exists(png_path):
        print(f"File {png_path} does not exist, skipping.")
        continue

    # Load the image
    img = cv2.imread(png_path)
    cropped_img = crop_img(img, c_t=C_T, c_b=C_B, c_l=C_L, c_r=C_R)
    cropped_png_path = os.path.join(DIRS_OUT_RES, f"cropped_labelled_{nr}.png")
    cv2.imwrite(cropped_png_path, cropped_img)