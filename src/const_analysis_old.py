#
# const_analysis.py
#

# sa overall information
SET_SA_GROUP = False
N_SMP = 64
SA_CALC_SECOND_ORDER = True

# sa boundary information
BOUNDARY_STRATEGY = "absolute"      # "absolute", "percentage"
BOUNDARY_VALUES = 0.02
SET_SA_DISTRIBUTION ="unif"         # "unif", "norm" 

# sa grouping type information
GROUP_STRATEGY = "location"         # "direction","hierarchy","location"

# sa freezer information
GROUP_FROZEN = "group_frozen"
GROUP_FREEZERS = ["st1_x", "st1_y", "st1_z"]

# other sa information
SALTELLI_SKIP = 1024

# SET_SA_DISTRIBUTION
# SET_SA_DISTRIBUTION = "unif" #uniform distribution
# SET_SA_DISTRIBUTION = "norm" #normal distribution
# BOUNDARY_STRATEGY
# BOUNDARY_STRATEGY = "absolute" #absolute boundary values
# BOUNDARY_STRATEGY = "percentage" #percentage boundary values
# GROUP_STRATEGY
# GROUP_STRATEGY = "direction" #group the parameter by x, y, z direction
# GROUP_STRATEGY = "hierarchy" #group the parameter by herarchical classes
# GROUP_STRATEGY = "location" #group the parameter by locational distribution


"""pca setting"""
# pca overall information
PCA_DIM = 3
PCA_RESULT_LABEL_TYPE = 'validity'

PCS_4_2D = [1, 2]
PCS_4_3D = [1, 2, 3]

# PCA_RESULT_LABEL_TYPE
# PCA_RESULT_LABEL_TYPE = 'to_1_0' #distance to 1 or 0.
# PCA_RESULT_LABEL_TYPE = 'to_bool' #distance to True or False.
# PCA_RESULT_LABEL_TYPE = 'validity' #distance to String.


"""clusters setting"""
# cluster overall information
PRE_CLUSTERS = 6
PRE_REGION_NB = 1