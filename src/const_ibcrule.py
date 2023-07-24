#
# const_ibcrule.py
#
from const_sensi import K_LEVEL_PARAMETER

# building rules
BUILDING_RULES = ['IBC1020_2', 'IBC1207_1', 'IBC1207_3']
BUILDING_RULES_4PCA = ['IBC1020_2', 'IBC1207_1', 'IBC1207_3']
BUILDING_RULES_4REGION = ['IBC1020_2', 'IBC1207_1', 'IBC1207_3', 'IBC_selected']

# failure analysis
LABEL_FAILURE_LOCATION = 'failure'
LABEL_FAILURE_NEIGHBOR = 'failure_neighbor'
LABLE_ASSOCIATED_GP = 'parameter_associated'

LEVEL_FAILURE_NEIGHBOR = K_LEVEL_PARAMETER

dictRestriction= {

    'belonging': {       
        'constraint_object': 'wall',
        'property_type': 'isexternal',
        'property_value': 1,
        'constraint_level': 'hard', 
        'constraint_type': ('location', 'fix'),
        },
    'propagating': {      
        'constraint_object': 'space',
        'property_type': 'name',
        'property_value': ['Stairway','Toilet'],
        'constraint_level': 'hard', 
        'constraint_type': ('location', 'fix'),
        },
    'supplementing': {      
        'constraint_object': 'space',
        'property_type': 'name',
        'property_value': 'Corridor',
        'constraint_level': 'hard',  
        'constraint_type': ('boundary','align'),
        },
    'selecting': {      
        'constraint_object': 'space',
        'property_type': 'name',
        'property_value': 'Power supply room',
        'constraint_level': 'soft',  
        'constraint_type': ('totalarea', 'fix'),
        },
        }

# if LEVEL_FAILURE_NEIGHBOR == 1:
#     EXCEPTION_LINK_SEQUENCE = [
#         ['wall_external','wall_stairway'] #0
#         ]
# elif LEVEL_FAILURE_NEIGHBOR == 3:
#     EXCEPTION_LINK_SEQUENCE = [
#         ['wall_external','wall_stairway'],  #0
#         ['space_corridor','space_stairway','space_toilet'], #1
#         ['wall_external','wall_stairway'] #1
#         ]
# elif LEVEL_FAILURE_NEIGHBOR == 5:
#     EXCEPTION_LINK_SEQUENCE = [
#         ['wall_external','wall_stairway'],  #0
#         ['space_corridor','space_stairway','space_toilet'], #1
#         ['wall_external','wall_stairway','wall_corridor'], #1
#         ['space_corridor','space_stairway','space_toilet'], #2
#         ['wall_external','wall_stairway'], #2
#         ]
    
# if LEVEL_FAILURE_NEIGHBOR == 1:
#     EXCEPTION_LINK_SEQUENCE = [
#         'wall_external' #0
#         ]
# elif LEVEL_FAILURE_NEIGHBOR == 3:
#     EXCEPTION_LINK_SEQUENCE = [
#         'wall_external',  #0
#         ['space_stairway','space_toilet'], #1
#         ['wall_external','wall_stairway'] #1
#         ]
# elif LEVEL_FAILURE_NEIGHBOR == 5:
#     EXCEPTION_LINK_SEQUENCE = [
#         'wall_external',  #0
#         ['space_stairway','space_toilet'], #1
#         'wall_external', #1
#         ['space_stairway','space_toilet'], #2
#         ['wall_external','wall_stairway'], #2
#         ]
# sweep target
APPROACH_TARGET = 'IBC_selected'