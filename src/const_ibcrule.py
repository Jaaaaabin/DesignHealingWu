#
# const_ibcrule.py
#

# building rules
BUILDING_RULES = ['IBC1020_2', 'IBC1207_1', 'IBC1207_3']
BUILDING_RULES_4PCA = ['IBC1020_2', 'IBC1207_1', 'IBC1207_3']
BUILDING_RULES_4REGION = ['IBC1020_2', 'IBC1207_1', 'IBC1207_3', 'IBC_selected']

# failure analysis
LABEL_FAILURE_LOCATION = 'failure'
LABEL_FAILURE_NEIGHBOR = 'failure_neighbor'
LABLE_ASSOCIATED_GP = 'parameter_associated'

# neighbor search parameters.
LEVEL_FAILURE_NEIGHBOR = 3
CLASS_LINKAGE = ['wall', 'space', 'wall']
EXCEPTION_LINKAGE = [True, True, True]
EXCEPTION_NAME_LINKAGE = ['isexternal','name','isexternal']
EXCEPTION_VALUE_LINKAGE = [1, 'Stairway', 1]

# LEVEL_FAILURE_NEIGHBOR = 1
# CLASS_LINKAGE = ['wall']
# EXCEPTION_LINKAGE = [True]
# EXCEPTION_NAME_LINKAGE = ['isexternal']
# EXCEPTION_VALUE_LINKAGE = [1]

# sweep target
APPROACH_TARGET = 'IBC_selected'