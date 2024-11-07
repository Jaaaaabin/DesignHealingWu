#
# tuneFigures.py
#

# import modules
import cv2

FOLDER_PAPER_FIGURES = r'C:\dev\phd\ModelHealer\paper-figures'
FOLDER_PAPER_FIGURES_NEW = r'C:\dev\phd\ModelHealer\paper-figures\new'

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
    label_name = 'sn1'
    new_dict = {
        'im_text': label_name,
        'im_x': 2650,
        'im_y': 1120,
        }
    label_dict.update({label_name: new_dict})

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
    
    #---------------------------------------- fixed
    label_name = 'sn0'
    new_dict = {
        'im_text': label_name,
        'im_x': 1510,
        'im_y': 600,
        }
    label_dict.update({label_name: new_dict})

    #-
    label_name = 'sn18'
    new_dict = {
        'im_text': label_name,
        'im_x': 1500,
        'im_y': 670,
        }
    label_dict.update({label_name: new_dict})

    #-
    label_name = 'sn24'
    new_dict = {
        'im_text': label_name,
        'im_x': 1500,
        'im_y': 1000,
        }
    label_dict.update({label_name: new_dict})
    #-
    label_name = 'sn19'
    new_dict = {
        'im_text': label_name,
        'im_x': 1500,
        'im_y': 1105,
        }
    label_dict.update({label_name: new_dict})


    #-
    label_name = 'ew2'
    new_dict = {
        'im_text': label_name,
        'im_x': 2180,
        'im_y': 900,
        }
    label_dict.update({label_name: new_dict})
    #-
    label_name = 'ew3'
    new_dict = {
        'im_text': label_name,
        'im_x': 2050,
        'im_y': 900,
        }
    label_dict.update({label_name: new_dict})
    #-
    label_name = 'ew4'
    new_dict = {
        'im_text': label_name,
        'im_x': 1950,
        'im_y': 900,
        }
    label_dict.update({label_name: new_dict})
    #-
    label_name = 'ew5'
    new_dict = {
        'im_text': label_name,
        'im_x': 1530,
        'im_y': 900,
        }
    label_dict.update({label_name: new_dict})
    #-
    label_name = 'ew18'
    new_dict = {
        'im_text': label_name,
        'im_x': 1440,
        'im_y': 900,
        }
    label_dict.update({label_name: new_dict})
    #-
    label_name = 'ew33'
    new_dict = {
        'im_text': label_name,
        'im_x': 850,
        'im_y': 900,
        }
    label_dict.update({label_name: new_dict})
    #-
    label_name = 'ew28'
    new_dict = {
        'im_text': label_name,
        'im_x': 650,
        'im_y': 900,
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


def set_line_parameters():

    line_dict = dict()

    #-
    label_name = 'sn21'
    new_dict = {
        'x0': 493,
        'y0': 345,
        'x1': 493,
        'y1': 541,
        }
    line_dict.update({label_name: new_dict})
    
    label_name = 'sn10'
    new_dict = {
        'x0': 649,
        'y0': 345,
        'x1': 649,
        'y1': 541,
        }
    line_dict.update({label_name: new_dict})
    
    label_name = 'sn26'
    new_dict = {
        'x0': 840,
        'y0': 345,
        'x1': 840,
        'y1': 541,
        }
    line_dict.update({label_name: new_dict})
    
    label_name = 'ew35'
    new_dict = {
        'x0': 295,
        'y0': 541,
        'x1': 1090,
        'y1': 541,
        }
    line_dict.update({label_name: new_dict})
    
    label_name = 'ew6'
    new_dict = {
        'x0': 295,
        'y0': 1026,
        'x1': 610,
        'y1': 1026,
        }
    line_dict.update({label_name: new_dict})

    return line_dict

def draw_line(im, line_dict):
    
    color = (0, 0, 255)
    thickness = 2
    
    for key in line_dict.keys():

        start_point = (line_dict[key]['x0'], line_dict[key]['y0'])
        end_point = (line_dict[key]['x1'], line_dict[key]['y1'])
        im = cv2.line(im, start_point, end_point, color, thickness)

    return im
  
# Displaying the image 

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


figure_name = r'\SA_mu_IBC1020_2_morris_Si_indices_convar_beta_0.png'
figure_file = FOLDER_PAPER_FIGURES + figure_name
figure_file_new  = FOLDER_PAPER_FIGURES_NEW + figure_name

im = cv2.imread(figure_file)
# parameter_labels = set_parameter_texts()
# im = draw_parameter_texts(im, parameter_labels)

im = crop_img(im, c_t = 130, c_b = 60, c_l = 330, c_r = 100,)
cv2.imwrite(figure_file_new, im)


# for floor plans.
# im = crop_img(im, c_t = 130, c_b = 260, c_l = 350, c_r = 250,)

# for XY pairplots.
# im = crop_img(im, c_t = 0, c_b = 50, c_l = 130, c_r = 800,)

# for SA mu plots.
# im = crop_img(im, c_t = 280, c_b = 50, c_l = 0, c_r = 200,)

# for SA mu convar.
# im = crop_img(im, c_t = 120, c_b = 60, c_l = 320, c_r = 100,)

# im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)


# figure_names = [r'\0.png',r'\669.png',r'\1282.png',r'\1439.png',r'\1522.png',r'\1718.png']

# for figure_name in figure_names:

#     figure_file = FOLDER_PAPER_FIGURES + figure_name
#     figure_file_new  = FOLDER_PAPER_FIGURES_NEW + figure_name

#     im = cv2.imread(figure_file)
#     # for final region output.
#     line_labels = set_line_parameters()
#     im = draw_line(im,line_labels)
#     im = crop_img(im, c_t = 330, c_b = 2250, c_l = 280, c_r = 280,)

#     cv2.imwrite(figure_file_new, im)


# FONT_HERSHEY_SIMPLEX = 0,
# FONT_HERSHEY_PLAIN = 1,
# FONT_HERSHEY_DUPLEX = 2,
# FONT_HERSHEY_COMPLEX = 3,
# FONT_HERSHEY_TRIPLEX = 4,
# FONT_HERSHEY_COMPLEX_SMALL = 5,
# FONT_HERSHEY_SCRIPT_SIMPLEX = 6,
# FONT_HERSHEY_SCRIPT_COMPLEX = 7,
# FONT_ITALIC = 16