import json
import pandas as pd
import numpy as np
from collections import namedtuple

def create_dictionary_key_mapping(dictionary):
    ini_keys = list(dictionary[0].keys())
    new_keys =  [key.replace(" ", "_") for key in ini_keys]
    lower_new_keys = [key.lower() for key in new_keys]
    door_type_parametername_map = dict(zip(ini_keys, lower_new_keys))
    return door_type_parametername_map

def map_dictionary_keys(dictionary,mapping):
    ini_keys = list(dictionary[0].keys())
    new_keys = [mapping[key] for key in ini_keys]
    
    for row in dictionary:
        # for each type
        for ini_key,new_key in zip(ini_keys, new_keys):
            row[new_key] = row.pop(ini_key)
    return dictionary

def read_dictionary_to_classobjects(dictionary,class_name='X'):

    objects = []
    for row in dictionary:
        # for each type
        row_in_string = json.dumps(row)
        object = json.loads(
            row_in_string,
            object_hook = lambda d : namedtuple(class_name, d.keys())(*d.values()))
        objects.append(object)
    
    return objects

def convert_revitdict_to_clsobjs(dict, class_name='X', string_mapping=True):
    if string_mapping:
        parametername_map = create_dictionary_key_mapping(dict)
        dictionary_mapped = map_dictionary_keys(dict, parametername_map)
    else:
        dictionary_mapped = dict
    class_objects = read_dictionary_to_classobjects(dictionary_mapped, class_name=class_name)

    return class_objects

def convert_clsobjs_into_df(cls_objs):

    # Get the class attributes.
    attributes = dir(cls_objs[0])
    
    # Clean unrelevant attributes.
    attributes = [att for att in attributes if not att.startswith('_')]
    attributes = [att for att in attributes if att !='index' and att !='count']
    
    # Convert to pandasDataFrame.
    df = pd.DataFrame([[getattr(obj, att) for att in attributes] for obj in cls_objs], columns = attributes)
    
    return df

def build_instance_df(
    cls_objs_instance, instance_type=[], final_index_name='ifcguid'):
    
    df_instances = convert_clsobjs_into_df(cls_objs_instance)
    
    if final_index_name:
        df = df_instances.set_index(final_index_name)
    else:
        df = df_instances
    
    # if one single classification.
    if instance_type:
        df['classification'] = instance_type

    # if there's a list of classification by default.
    else:
        df['classification'] = df.apply(lambda x: x['name'].rsplit('_', 1)[0], axis=1)
        df['classification'] = df.apply(lambda x: 'Space_'+ x['classification'], axis=1)
        
    return df

# Constants
# TYPE_ATTRIBUTES = ['Door','Window','Wall','Slab']
# INSTANCE_ATTRIBUTES = ['Door','Window','Wall','Slab','Space']

DICT_REVIT_RES = r'C:\dev\phd\jw\healing\data\healing2023\20_extract_data\res'
INSTANCE_NAME = "\collected_Instances_"

# Set JSON files

JS_FILE_INSTANCE_DOOR = DICT_REVIT_RES + INSTANCE_NAME + 'Door.json'
JS_FILE_INSTANCE_WINDOW = DICT_REVIT_RES + INSTANCE_NAME + 'Window.json'
JS_FILE_INSTANCE_WALL = DICT_REVIT_RES + INSTANCE_NAME + 'Wall.json'
JS_FILE_INSTANCE_SLAB = DICT_REVIT_RES + INSTANCE_NAME + 'Slab.json'
JS_FILE_INSTANCE_SPACE = DICT_REVIT_RES + INSTANCE_NAME + 'Space.json'
JS_FILE_GLOBAL_PARAMETER = DICT_REVIT_RES + "\collected_GPs.json"

# Open JSON files for instances
with open(JS_FILE_INSTANCE_DOOR) as json_file:
    revit_doorinstances = json.load(json_file)
with open(JS_FILE_INSTANCE_WINDOW) as json_file:
    revit_windowinstances = json.load(json_file)
with open(JS_FILE_INSTANCE_WALL) as json_file:
    revit_wallinstances = json.load(json_file)
with open(JS_FILE_INSTANCE_SLAB) as json_file:
    revit_slabinstances = json.load(json_file)
with open(JS_FILE_INSTANCE_SPACE) as json_file:
    revit_spaceinstances = json.load(json_file)
with open(JS_FILE_GLOBAL_PARAMETER) as json_file:
    revit_globalparameters = json.load(json_file)

# Convert the revit-ouput dictionary to class objects (Revit Instances)
cls_objs_doorinstances = convert_revitdict_to_clsobjs(revit_doorinstances, class_name='Door')
cls_objs_windowinstances = convert_revitdict_to_clsobjs(revit_windowinstances, class_name='Window')
cls_objs_wallinstances = convert_revitdict_to_clsobjs(revit_wallinstances, class_name='Wall')
cls_objs_slabinstances = convert_revitdict_to_clsobjs(revit_slabinstances, class_name='Slab')
cls_objs_spaceinstances = convert_revitdict_to_clsobjs(revit_spaceinstances, class_name='Space')
cls_objs_globalparameters = convert_revitdict_to_clsobjs(revit_globalparameters, class_name='Parameter')

# DataFrame as output
df_doorinstances =  build_instance_df(
    cls_objs_doorinstances, 'Element_Door')
df_windowinstances =  build_instance_df(
    cls_objs_windowinstances, 'Element_Window')
df_wallinstances =  build_instance_df(
    cls_objs_wallinstances, 'Element_Wall')
df_slabinstances =  build_instance_df(
    cls_objs_slabinstances, 'Element_Slab')
df_spaceinstances =  build_instance_df(
    cls_objs_spaceinstances, 'Space')
df_globalparameters =  build_instance_df(
    cls_objs_globalparameters, 'Parameter_Global', final_index_name='id')

# Collect Topology between GP and elements
# output:   collected_topology_GP_host.txt
#           collected_topology_GP_objects.txt
with open(DICT_REVIT_RES+'\collected_topology_GP_host.txt', 'w') as f:
    df_as_string = df_globalparameters['name'].to_string(header=False, index=False)
    df_as_string = df_as_string.replace(" ", "")
    f.write(df_as_string)

with open(DICT_REVIT_RES+'\collected_topology_GP_objects.txt', 'w') as f:
    df_as_string = df_globalparameters['elementifcguid'].to_string(header=False, index=False)
    f.write(df_as_string)

# Write to csv
df_doorinstances.to_csv(DICT_REVIT_RES+'\df_Doors.csv')
df_windowinstances.to_csv(DICT_REVIT_RES+'\df_Windows.csv')
df_wallinstances.to_csv(DICT_REVIT_RES+'\df_Walls.csv')
df_slabinstances.to_csv(DICT_REVIT_RES+'\df_Slabs.csv')
df_spaceinstances.to_csv(DICT_REVIT_RES+'\df_Spaces.csv')
df_globalparameters.to_csv(DICT_REVIT_RES+'\df_Parameters.csv')