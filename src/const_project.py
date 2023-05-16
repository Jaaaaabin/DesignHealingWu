#
# const_project.py
# 

# directory: overall
DIRS_ZERO = r'C:\dev\phd\ModelHealer'


# directory: initial design
DIRS_INIT =  DIRS_ZERO + r'\ini'

FILE_INIT_RVT = DIRS_INIT + r'\model\ini.rvt'
FILE_INIT_RVT = DIRS_INIT + r'\model\ini_param.rvt'
FILE_INIT_RES = DIRS_INIT + r'\res\0.h5'

# DIRS_DATA_TOPO = r'C:\dev\phd\enrichgraph\data\ini\res'
DIRS_DATA_TOPO = DIRS_INIT + r'\topo'
FILE_LIST_GP = DIRS_DATA_TOPO + r'\df_parameter.csv'

FILE_RELATED_FL_PERRULE = DIRS_DATA_TOPO + r'\res_ini_perrule_failure.csv'
FILE_RELATED_NB_PERRULE = DIRS_DATA_TOPO + r'\res_ini_perrule_failureneighbor.csv'
FILE_RELATED_GP_PERRULE = DIRS_DATA_TOPO + r'\res_ini_perrule_parameter.csv'
FILE_RELATED_GP_INI = DIRS_DATA_TOPO + r'\res_ini_parameter.csv'

NAME_TOPO_INSTANCES = ['door','window','wall','slab','space','parameter']
NAME_INSTANCE_COLLECTION = r'\collected_instances_'

NAME_TOPO_OBJECT = r'\collected_topology_wall_'
NAME_TOPO_SPACE = r'\collected_topology_space_'
NAME_TOPO_PARAMETER = r'\collected_topology_parameter_'

# directory: dynamo scripts
DIRS_SCRIPT_DYN = DIRS_ZERO + r'\dyns'

FILE_CONTROL_RVT = DIRS_SCRIPT_DYN + r'\control.rvt'


# directory: external scripts
DIRS_SCRIPT_EXT = DIRS_ZERO + r'\src'


# directory: data
DIRS_DATA = DIRS_ZERO + r'\data'

# directory: data-sa
DIRS_DATA_SA = DIRS_DATA + r'\sa'


# directory: tests
DIRS_TEST = DIRS_ZERO + r'\tests'


# DIRS_BIM = DIRS_ZERO + r'\data\onestorey'
# DIRS_BIM_RES = DIRS_BIM + r'\res'
# DIRS_BIM_LOG = DIRS_BIM + r'\log'
# FILE_INIT_PARAM = DIRS_BIM + r'\0_initial_parameters.csv'
# FILE_INIT_RVT = DIRS_BIM + r'\initial_design.rvt'
# FILE_DYN_ADJUST = DIRS_BIM + r'\healing_adjusting.dyn'
# FILE_SAMPLES_VARIATION = "0_variationData.txt"

# # data transfer information
# EXE_NAME = 'StartProcess'
# DIRS_TRAS = DIRS_ZERO + r'\src\gorvt\HealingRVT'
# FILE_TRAS_JSON = DIRS_ZERO + r'\transfer.json'
# FILE_TRAS_EXE = DIRS_TRAS + '\\' + EXE_NAME + r'\bin\Debug' + '\\' + EXE_NAME + r'.exe'

# # h5 data
# INPUT_H5_KEY = 'geom_Input'
# OUTPUT_H5_DATAKEY = 'checking'
