#
# const_project.py
#

# # # # # # # # #
# for graphEnrich
# EXECUTION_NR  = 11 
#               or 31 
#               or 51

# # # # # # # # #
# for SA sobol
# EXECUTION_NR  = 15(first)
#               or 55(second)

# for SA morris
# EXECUTION_NR  = 12(first)
#               or 52(second)

EXECUTION_NR = 52
TEST_PLOT = 0.02

# directory: overall
DIRS_ZERO = r'C:\dev\phd\ModelHealer'

# directory: initial design
DIRS_INIT =  DIRS_ZERO + r'\ini'
DIRS_INI_RES = DIRS_INIT + r'\res'

FILE_INIT_RVT = DIRS_INIT + r'\model\ini.rvt'

FILE_INIT_RVT = DIRS_INIT + r'\model\ini_param.rvt'
FILE_INIT_SKL_RVT = DIRS_INIT + r'\model\ini_param_skeleton.rvt'
FILE_INIT_RES = DIRS_INIT + r'\res\0.h5'

# DIRS_DATA_TOPO = r'C:\dev\phd\enrichgraph\data\ini\res'
DIRS_DATA_TOPO = DIRS_INIT + r'\topo'
FILE_LIST_GP = DIRS_DATA_TOPO + r'\df_parameter.csv'

FILE_RELATED_FL_PERRULE = DIRS_DATA_TOPO + r'\res_ini_perrule_failure.csv'
FILE_RELATED_NB_PERRULE = DIRS_DATA_TOPO + r'\res_ini_perrule_failureneighbor.csv'
FILE_RELATED_GP_PERRULE = DIRS_DATA_TOPO + r'\res_ini_perrule_parameter.csv'
FILE_RELATED_GP_INI = DIRS_DATA_TOPO + r'\res_ini_parameter_values.csv'

FILE_INI_GRAPH = DIRS_DATA_TOPO + r'\res_graph.txt'

NAME_TOPO_INSTANCES = ['door','window','wall','slab','space','parameter']
NAME_INSTANCE_COLLECTION = r'\collected_instances_'

NAME_TOPO_OBJECT = r'\collected_topology_wall_'
NAME_TOPO_SPACE = r'\collected_topology_space_'
NAME_TOPO_PARAMETER = r'\collected_topology_parameter_'

FILE_SA_PARAM_LIST = DIRS_DATA_TOPO + r'\neighbor_tbd\res_ini_parameter_values.csv'

# directory: dynamo scripts
DIRS_SCRIPT_DYN = DIRS_ZERO + r'\dyns'

FILE_CONTROL_RVT = DIRS_SCRIPT_DYN + r'\control.rvt'

# directory: external scripts
DIRS_SCRIPT_EXT = DIRS_ZERO + r'\src'

# directory: data
DIRS_DATA = DIRS_ZERO + r'\data'

# directory: data-sa-

DIRS_DATA_SA = DIRS_DATA + r'\sa-' + str(EXECUTION_NR) if TEST_PLOT==0 else DIRS_DATA + r'\sa-' + str(EXECUTION_NR) + '-' + str(TEST_PLOT)  
FILE_SA_VARY_SOBOL = DIRS_DATA_SA + r'\sa_vary_sobol.csv'
FILE_SA_VARY_MORRIS = DIRS_DATA_SA + r'\sa_vary_morris.csv'
DIRS_DATA_SA_DUP = DIRS_DATA_SA + r'\dups'
DIRS_DATA_SA_VARY = DIRS_DATA_SA + r'\vary'
DIRS_DATA_SA_RES = DIRS_DATA_SA + r'\res'
DIRS_DATA_SA_FIG = DIRS_DATA_SA + r'\fig'

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

NR_PROJECT = 100