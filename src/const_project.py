#
# const_project.py
#

# # # # # # # # #
# for SA morris
# EXECUTION_NR  = 14(first)
#               or 54(second)

EXECUTION_NR = 34 #SA
SOLUTION_NR = EXECUTION_NR + 100 #SS

# directory: overall
DIRS_ZERO = r'C:\dev\phd\ModelHealer'

# directory: initial design
DIRS_INIT =  DIRS_ZERO + r'\ini'
DIRS_INI_RES = DIRS_INIT + r'\res'

FILE_INIT_RVT = DIRS_INIT + r'\model\ini.rvt'

FILE_INIT_RVT = DIRS_INIT + r'\model\ini_param.rvt'
FILE_INIT_SKL_RVT = DIRS_INIT + r'\model\ini_param_skeleton.rvt'
FILE_INIT_RES = DIRS_INIT + r'\res\0.h5'

DIRS_DATA_TOPO = DIRS_INIT + r'\topo' # journal paper experiments.
# DIRS_DATA_TOPO = DIRS_INIT + r'\topo_simplified' # journal paper figure maker.

FILE_LIST_GP = DIRS_DATA_TOPO + r'\df_parameter.csv'

FILE_RELATED_FL_PERRULE = DIRS_DATA_TOPO + r'\res_ini_perrule_failure.csv'
FILE_RELATED_EL_PERRULE = DIRS_DATA_TOPO + r'\res_ini_perrule_failureelement.csv'
FILE_RELATED_GP_PERRULE = DIRS_DATA_TOPO + r'\res_ini_perrule_parameter.csv'
FILE_RELATED_EL_CONSTRAINT_PERRULE = DIRS_DATA_TOPO + r'\res_ini_perrule_failureelement_constraint.csv'
FILE_RELATED_GP_CONSTRAINT_PERRULE = DIRS_DATA_TOPO + r'\res_ini_perrule_parameter_constraint.csv'

FILE_RELATED_GP_INI = DIRS_DATA_TOPO + r'\res_ini_parameter_values.csv'

FILE_INI_GRAPH = DIRS_DATA_TOPO + r'\res_graph.txt'
FILE_CONSTRAINTS = DIRS_DATA_TOPO + r'\res_ini_constraints.json'
FILE_CONSTRAINTS_APPLY = DIRS_DATA_TOPO + r'\res_ini_constraints_applied.json'

NAME_TOPO_INSTANCES = ['door','window','wall','slab','space','separationline']
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