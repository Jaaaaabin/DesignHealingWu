#
# topoCollection.py
#

# import modules
from const_project import DIRS_DATA_TOPO, NAME_INSTANCE_COLLECTION
from funct_topo import *

def topoCollect():

    # Set JSON files
    JS_FILE_INSTANCE_DOOR = DIRS_DATA_TOPO + NAME_INSTANCE_COLLECTION + r'Door.json'
    JS_FILE_INSTANCE_WINDOW = DIRS_DATA_TOPO + NAME_INSTANCE_COLLECTION + r'Window.json'
    JS_FILE_INSTANCE_WALL = DIRS_DATA_TOPO + NAME_INSTANCE_COLLECTION + r'Wall.json'
    JS_FILE_INSTANCE_SLAB = DIRS_DATA_TOPO + NAME_INSTANCE_COLLECTION + r'Slab.json'
    JS_FILE_INSTANCE_SPACE = DIRS_DATA_TOPO + NAME_INSTANCE_COLLECTION + r'Space.json'
    JS_FILE_GLOBAL_PARAMETER = DIRS_DATA_TOPO + r'\collected_GPs.json'

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

    # Collect Topology between GP and elements -> output:   collected_topology_GP_host.txt and collected_topology_GP_objects.txt.
    with open(DIRS_DATA_TOPO+'\collected_topology_GP_host.txt', 'w') as f:
        df_as_string = df_globalparameters['name'].to_string(header=False, index=False)
        df_as_string = df_as_string.replace(" ", "")
        f.write(df_as_string)

    with open(DIRS_DATA_TOPO+'\collected_topology_GP_objects.txt', 'w') as f:
        df_as_string = df_globalparameters['elementifcguid'].to_string(header=False, index=False)
        f.write(df_as_string)

    # Write to csv
    df_doorinstances.to_csv(DIRS_DATA_TOPO+'\df_Doors.csv')
    df_windowinstances.to_csv(DIRS_DATA_TOPO+'\df_Windows.csv')
    df_wallinstances.to_csv(DIRS_DATA_TOPO+'\df_Walls.csv')
    df_slabinstances.to_csv(DIRS_DATA_TOPO+'\df_Slabs.csv')
    df_spaceinstances.to_csv(DIRS_DATA_TOPO+'\df_Spaces.csv')
    df_globalparameters.to_csv(DIRS_DATA_TOPO+'\df_Parameters.csv')
