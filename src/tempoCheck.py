#
# tempoCheck.py
#

# import modules

from const_project import DIRS_INI_RES, DIRS_DATA_SA, DIRS_DATA_SA_RES, DIRS_DATA_SA_FIG
from const_project import FILE_SA_PARAM_LIST, FILE_SA_VARY_SOBOL, FILE_SA_VARY_MORRIS
from const_ibcrule import BUILDING_RULES
from const_sensi import K_LEVEL_PARAMETER, SA_CALC_SECOND_ORDER, NAME_FLOOR, NAME_FAILURES

from funct_data import analyze_h5s, save_dict, load_dict


tempo = load_dict(r'C:\dev\phd\ModelHealer\data\sa-16\sa_morris_indices.pickle')

print('end')