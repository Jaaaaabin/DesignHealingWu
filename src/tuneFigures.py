#
# tuneFigures.py
#

# import modules
import cv2

FOLDER_PAPER_FIGURES = r'C:\dev\phd\ModelHealer\paper-figures'
FOLDER_PAPER_FIGURES_NEW = r'C:\dev\phd\ModelHealer\paper-figures\new'
figure_name = r'\Space_sa-34-0.3_ss-134-0_ss-134-1_ss-134-2_pairwise_relationship_compliance.png'
figure_file = FOLDER_PAPER_FIGURES + figure_name
figure_file_new  = FOLDER_PAPER_FIGURES_NEW + figure_name

def set_parameter_texts():

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
    
    return label_dict


def draw_parameter_texts(im, text_dict):

    for key in text_dict.keys():

        # constant values.
        font = cv2.FONT_HERSHEY_DUPLEX

        # variant values.
        cv2.putText(
            im,
            text_dict[key]['im_text'],
            (text_dict[key]['im_x'],text_dict[key]['im_y']),
            font,
            0.9,
            (0, 0, 0),
            2,
            cv2.LINE_AA)
    
    return im


def crop_img(
    img,
    c_t = 0,
    c_b = 0,
    c_l = 0,
    c_r = 0,
    ):

    x,y = 0,0
    im_h, im_w = img.shape[:2]
    crop_im = im[y+c_t:y+im_h-c_b, x+c_l:x+im_w-c_r]
    
    return crop_im


im = cv2.imread(figure_file)

# for floor plans.
# parameter_labels = set_parameter_texts()
# im = draw_parameter_texts(im, parameter_labels)
# im = crop_img(im, c_t = 130, c_b = 260, c_l = 350, c_r = 250,)

# for XY pairplots.
im = crop_img(im, c_t = 0, c_b = 50, c_l = 130, c_r = 800,)

# for SA mu plots.
# im = crop_img(im, c_t = 280, c_b = 150, c_l = 330, c_r = 200,)

# for SA mu convar.
# im = crop_img(im, c_t = 120, c_b = 60, c_l = 320, c_r = 100,)

# im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

cv2.imwrite(figure_file_new, im)

# FONT_HERSHEY_SIMPLEX = 0,
# FONT_HERSHEY_PLAIN = 1,
# FONT_HERSHEY_DUPLEX = 2,
# FONT_HERSHEY_COMPLEX = 3,
# FONT_HERSHEY_TRIPLEX = 4,
# FONT_HERSHEY_COMPLEX_SMALL = 5,
# FONT_HERSHEY_SCRIPT_SIMPLEX = 6,
# FONT_HERSHEY_SCRIPT_COMPLEX = 7,
# FONT_ITALIC = 16