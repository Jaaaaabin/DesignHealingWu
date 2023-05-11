#
# const_project.py
# 

# overall information
DIRS_ZERO = r'C:\dev\phd\jw\ModelHealer'

# bim information
DIRS_BIM = DIRS_ZERO + r'\data\onestorey'
DIRS_BIM_RES = DIRS_BIM + r'\res'
DIRS_BIM_LOG = DIRS_BIM + r'\log'
FILE_INIT_PARAM = DIRS_BIM + r'\0_initial_parameters.csv'
FILE_INIT_RVT = DIRS_BIM + r'\initial_design.rvt'
FILE_DYN_ADJUST = DIRS_BIM + r'\healing_adjusting.dyn'
FILE_SAMPLES_VARIATION = "0_variationData.txt"

# data transfer information
EXE_NAME = 'StartProcess'
DIRS_TRAS = DIRS_ZERO + r'\src\gorvt\HealingRVT'
FILE_TRAS_JSON = DIRS_ZERO + r'\transfer.json'
FILE_TRAS_EXE = DIRS_TRAS + '\\' + EXE_NAME + r'\bin\Debug' + '\\' + EXE_NAME + r'.exe'

# h5 data
INPUT_H5_KEY = 'geom_Input'
OUTPUT_H5_DATAKEY = 'checking'
