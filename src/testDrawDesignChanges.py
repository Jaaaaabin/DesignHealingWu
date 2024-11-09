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
# designnumber (nr)     sample index (idx)      weighted-euc 
# 1282                  381                     0.419
# 1522                  482                     0.454
# 1718                  1079                    0.577
# 1512                  1559                    0.690
# 669                   1657                    0.739
res_plot_nr = [0, 1282, 1522, 1718, 1512, 669]

# # ========================================
# # Step0: collect the generated data.
# # ========================================
# sampling_data = os.path.join(DIRS_DATA, 'ss-134-2', 'valid_via_weighted_distance.csv')
# df = pd.read_csv(sampling_data, index_col='designnumber')
# filtered_df = df.loc[df.index.intersection(res_plot_nr)]
# filtered_df = filtered_df.round(3)
# filtered_df = filtered_df.loc[:, ~filtered_df.columns.str.startswith('U1_')]
 
# # ========================================
# # Step1: producing the design alternatives.
# # ========================================
# for nr in res_plot_nr:
#     # make the subfoloders
#     subfolder_path = os.path.join(DIRS_OUT_RES, str(nr))
#     os.makedirs(subfolder_path, exist_ok=True)

#     # Copy the initial design to the folder with the nr number as its name
#     variant_file_path = os.path.join(subfolder_path, f"{nr}{os.path.splitext(FILE_INIT_RVT)[1]}")
#     shutil.copy(FILE_INIT_RVT, variant_file_path)

#     # Create a JSON file with the values from the row of the DataFrame
#     if nr in filtered_df.index:
#         row_data = filtered_df.loc[nr].to_dict()  # Convert row to dictionary format
#         row_data.pop('Unnamed: 0', None)
#         row_data_with_nr = {'designnumber': nr, **row_data}  # Add 'designnumber' as nr key
        
#         json_file_path = os.path.join(subfolder_path, f"{nr}.json")
        
#         # Write the JSON file
#         with open(json_file_path, 'w') as json_file:
#             json.dump(row_data_with_nr, json_file, indent=4)

# # ========================================
# # Step2: PDF2PNG
# # ========================================

# for plot_order, nr in enumerate(res_plot_nr):

#     subfolder_path = os.path.join(DIRS_OUT_RES, str(nr))
#     pdf_path = next((f for f in os.listdir(subfolder_path) if f.endswith('.pdf')), None)
#     if pdf_path is None:
#         print(f"No PDF found in {subfolder_path}")
#         continue
#     pdf_full_path = os.path.join(subfolder_path, pdf_path)

#     # Convert PDF to PNG for the single page
#     page = convert_from_path(pdf_full_path, dpi=500)[0]  # Only take the first page
#     png_path = os.path.join(subfolder_path, f"{plot_order}_{os.path.splitext(pdf_path)[0]}.png")
#     page.save(png_path, 'PNG')

# # ========================================
# # Step3: CROPPING
# # ========================================
# C_T = 0.06
# C_B = 0.675
# C_L = 0.05
# C_R = 0.37

# for plot_order, nr in enumerate(res_plot_nr):
    
#     subfolder_path = os.path.join(DIRS_OUT_RES, str(nr))
#     png_path = next((f for f in os.listdir(subfolder_path) if f.endswith('.png') and not f.endswith('_cropped.png')), None)
#     if png_path is None:
#         print(f"No PNG found in {subfolder_path} for cropping.")
#         continue
    
#     png_full_path = os.path.join(subfolder_path, png_path)
#     img = cv2.imread(png_full_path)
#     cropped_img = crop_img(img, c_t=C_T, c_b=C_B, c_l=C_L, c_r=C_R)
#     cropped_png_path = os.path.join(subfolder_path, f"{os.path.splitext(png_path)[0]}_cropped.png")
#     cv2.imwrite(cropped_png_path, cropped_img)

# # ========================================
# # Step4: convert the labelled pdfs to pngs  + do recutting.
# # ========================================

# labelled_pdf_path = os.path.join(DIRS_OUT_RES, 'summary.pdf')
# pages = convert_from_path(labelled_pdf_path, dpi=600)

# for i, page in enumerate(pages):
#     if i < len(res_plot_nr):
#         nr = res_plot_nr[i]
#         png_path = os.path.join(DIRS_OUT_RES, f"labelled_{i}_{nr}.png")
#         page.save(png_path, 'PNG')
#     else:
#         print(f"Warning: More pages in PDF than labels in res_plot_nr. Skipping page {i}.")

# C_T = 0.07
# C_B = 0.07
# C_L = 0.20
# C_R = 0.215

# for i,nr in enumerate(res_plot_nr):
    
#     png_path = os.path.join(DIRS_OUT_RES, f"labelled_{i}_{nr}.png")
#     if not os.path.exists(png_path):
#         print(f"File {png_path} does not exist, skipping.")
#         continue

#     img = cv2.imread(png_path)
#     cropped_img = crop_img(img, c_t=C_T, c_b=C_B, c_l=C_L, c_r=C_R)
#     cropped_png_path = os.path.join(DIRS_OUT_RES, f"cropped_labelled_{i}_{nr}.png")
#     cv2.imwrite(cropped_png_path, cropped_img)