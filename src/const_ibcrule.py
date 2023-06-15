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

# neighbor search parameters.
# exceptions during linking.
EXCEPTION_LINK_TYPES = {
    'wall_external': {
        'link_type': 'wall',
        'property_type': 'isexternal',
        'property_value': 1
    },
    'wall_stairway': {
        'link_type': 'wall',
        'property_type': 'isstairwaywall',
        'property_value': 1
    },
    'wall_corridor': {
        'link_type': 'wall',
        'property_type': 'iscorridorwall',
        'property_value': 1
    },
    'space_corridor': {
        'link_type': 'space',
        'property_type': 'name',
        'property_value': 'Corridor'
    },
    'space_stairway': {
        'link_type': 'space',
        'property_type': 'name',
        'property_value': 'Stairway'
    },
    'space_toilet': {
        'link_type': 'space',
        'property_type': 'name',
        'property_value': 'Toilet'
    }
}

LEVEL_FAILURE_NEIGHBOR = K_LEVEL_PARAMETER

if LEVEL_FAILURE_NEIGHBOR == 1:
    EXCEPTION_LINK_SEQUENCE = [
        ['wall_external','wall_stairway'] #0
        ]
elif LEVEL_FAILURE_NEIGHBOR == 3:
    EXCEPTION_LINK_SEQUENCE = [
        ['wall_external','wall_stairway'],  #0
        ['space_corridor','space_stairway','space_toilet'], #1
        ['wall_external','wall_stairway'] #1
        ]
elif LEVEL_FAILURE_NEIGHBOR == 5:
    EXCEPTION_LINK_SEQUENCE = [
        ['wall_external','wall_stairway'],  #0
        ['space_corridor','space_stairway','space_toilet'], #1
        ['wall_external','wall_stairway','wall_corridor'], #1
        ['space_corridor','space_stairway','space_toilet'], #2
        ['wall_external','wall_stairway'], #2
        ]

# sweep target
APPROACH_TARGET = 'IBC_selected'