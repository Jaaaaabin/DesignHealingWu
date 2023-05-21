#
# testSensitivity.py
#

from const_project import DIRS_INI_RES, DIRS_DATA_SA_RES, DIRS_DATA_SA
from const_project import FILE_SA_PARAM_LIST, FILE_SA_VARY
from const_ibcrule import BUILDING_RULES
from const_sensi import K_LEVEL_PARAMETER

from funct_data import analyze_h5s, save_dict, load_dict
from funct_sensi import *

from Design import Design

# dictionary: design - > rule -> target -> distance & compliance.
ini_dictCheckResult_h5s = analyze_h5s(DIRS_INI_RES, BUILDING_RULES)
dictCheckResult_h5s = analyze_h5s(DIRS_DATA_SA_RES, BUILDING_RULES)

# create the initial design: "DesignIni"
DesignIni = Design(list(ini_dictCheckResult_h5s.keys())[0], BUILDING_RULES)
sa_ini_parameter_names, sa_ini_parameter_values, sa_ini_parameter_num = collect_ini_sa_parameters(
    FILE_SA_PARAM_LIST, K_LEVEL_PARAMETER)
DesignIni.set_parameters({k:v for k,v in zip(sa_ini_parameter_names,sa_ini_parameter_values)})
DesignIni.set_checkresults(ini_dictCheckResult_h5s[0])

# create the new designs: "DesignsNew"
DesignsNew  = [Design(nr, BUILDING_RULES) for nr in list(dictCheckResult_h5s.keys())]
sa_new_parameter_names = pd.read_csv(FILE_SA_VARY, index_col=0, header=None).T.columns.tolist()
sa_new_parameter_values_all = pd.read_csv(FILE_SA_VARY, index_col=0, header=None).T.values.tolist()
for newDesign, sa_new_parameter_values in zip(DesignsNew, sa_new_parameter_values_all):
    newDesign.set_parameters({k:v for k,v in zip(sa_new_parameter_names, sa_new_parameter_values)})
    newDesign.set_checkresults(dictCheckResult_h5s[newDesign.number])

save_dict(DesignIni, DIRS_DATA_SA + r'\DesignIni.pickle')
save_dict(DesignsNew, DIRS_DATA_SA + r'\DesignsNew.pickle')

# DesignIni = load_dict(DIRS_DATA_SA + r'\DesignIni.pickle')
# DesignsNew = load_dict(DIRS_DATA_SA + r'\DesignsNew.pickle')


print ('end')