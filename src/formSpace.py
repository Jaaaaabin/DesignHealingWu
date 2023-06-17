#
# formSpace.py
#

# import modules

from base_external_packages import *
from funct_data import save_dict, load_dict, get_problems_from_paths, flatten
from Space import SolutionSpace
from const_project import DIRS_DATA


def formSolutionSpace(dataset=[]):

    pathIni = DIRS_DATA + dataset[0] + r'\DesignIni.pickle'
    pathRes = DIRS_DATA + dataset[0] + r'\res'
    problems =  get_problems_from_paths(pathRes)

    designIni = load_dict(pathIni)
    # del designIni.parameters["U1_OK_d_wl_sn25"]

    pathsNew = [DIRS_DATA + set + r'\DesignsNew.pickle' for set in dataset]
    designsNew = flatten([load_dict(path) for path in pathsNew])

    firstSpace = SolutionSpace(problems)
    firstSpace.set_center(designIni)
    firstSpace.form_space(designsNew)
    firstSpace.subdivide_space()

    nameFileSpace = '_'.join([data.replace('\\','') for data in dataset])
    save_dict(firstSpace, DIRS_DATA + r'\Space_' + nameFileSpace + r'.pickle')
    
    print(firstSpace)
    
    