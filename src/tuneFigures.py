#
# tuneFigures.py
#

# import modules
import cv2

FOLDER_PAPER_FIGURES = r'C:\dev\phd\ModelHealer\paper-figures'
FOLDER_PAPER_FIGURES_NEW = r'C:\dev\phd\ModelHealer\paper-figures\new'
figure_name = r'\U1_OK_RES_IBC1020_2_neighbor_3.png'

figure_file = FOLDER_PAPER_FIGURES + figure_name
figure_file_new  = FOLDER_PAPER_FIGURES_NEW + figure_name

label_dict = dict()
#-
label_name = 'ew35'
new_dict = {
    'im_text': label_name,
    'im_x': 2020,
    'im_y': 300,
    }
label_dict.update({label_name: new_dict})
#-
label_name = 'ew9'
new_dict = {
    'im_text': label_name,
    'im_x': 2220,
    'im_y': 300,
    }
label_dict.update({label_name: new_dict})
#-
label_name = 'ew7'
new_dict = {
    'im_text': label_name,
    'im_x': 2360,
    'im_y': 300,
    }
label_dict.update({label_name: new_dict})
#-
label_name = 'ew6'
new_dict = {
    'im_text': label_name,
    'im_x': 2450,
    'im_y': 300,
    }
label_dict.update({label_name: new_dict})

#----------------------------------------
label_name = 'sn21'
new_dict = {
    'im_text': label_name,
    'im_x': 2650,
    'im_y': 490,
    }
label_dict.update({label_name: new_dict})
#-
label_name = 'sn10'
new_dict = {
    'im_text': label_name,
    'im_x': 2650,
    'im_y': 620,
    }
label_dict.update({label_name: new_dict})
#-
label_name = 'sn26'
new_dict = {
    'im_text': label_name,
    'im_x': 2650,
    'im_y': 770,
    }
label_dict.update({label_name: new_dict})
#-
label_name = 'sn25'
new_dict = {
    'im_text': label_name,
    'im_x': 2650,
    'im_y': 970,
    }
label_dict.update({label_name: new_dict})
#-
label_name = 'sn19'
new_dict = {
    'im_text': label_name,
    'im_x': 2650,
    'im_y': 1120,
    }
label_dict.update({label_name: new_dict})

#----------------------------------------
label_name = 'sn9'
new_dict = {
    'im_text': label_name,
    'im_x': 2330,
    'im_y': 460,
    }
label_dict.update({label_name: new_dict})
#-
label_name = 'sn23'
new_dict = {
    'im_text': label_name,
    'im_x': 2330,
    'im_y': 830,
    }
label_dict.update({label_name: new_dict})

im = cv2.imread(figure_file)

im_h, im_w = im.shape[:2]

for key in label_dict.keys():

    # constant values.
    font = cv2.FONT_HERSHEY_DUPLEX

    # variant values.
    cv2.putText(
        im,
        label_dict[key]['im_text'],
        (label_dict[key]['im_x'],label_dict[key]['im_y']),
        font,
        0.9,
        (0, 0, 0),
        2,
        cv2.LINE_AA)

x,y = 0,0

c_t = 100
c_b = 260
c_l = 350
c_r = 250

crop_im = im[y+c_t:y+im_h-c_b, x+c_l:x+im_w-c_r]
cv2.imwrite(figure_file_new, crop_im)


# FONT_HERSHEY_SIMPLEX = 0,
# FONT_HERSHEY_PLAIN = 1,
# FONT_HERSHEY_DUPLEX = 2,
# FONT_HERSHEY_COMPLEX = 3,
# FONT_HERSHEY_TRIPLEX = 4,
# FONT_HERSHEY_COMPLEX_SMALL = 5,
# FONT_HERSHEY_SCRIPT_SIMPLEX = 6,
# FONT_HERSHEY_SCRIPT_COMPLEX = 7,
# FONT_ITALIC = 16