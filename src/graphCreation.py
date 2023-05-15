#
# graphCreation.py
#

# import modules
from const_project import FILE_INIT_RES, DIRS_DATA_TOPO
from const_project import NAME_TOPO_OBJECT, NAME_TOPO_SPACE, NAME_TOPO_PARAMETER
from const_project import FILE_LIST_GP, FILE_RELATED_GP_PERRULE, FILE_RELATED_FL_PERRULE, FILE_RELATED_NB_PERRULE, FILE_RELATED_GP_INI

from const_ibcrule import BUILDING_RULES
from const_ibcrule import LEVEL_FAILURE_NEIGHBOR, LABEL_FAILURE_LOCATION, LABEL_FAILURE_NEIGHBOR, LABLE_ASSOCIATED_GP

from funct_topo import *

def graphCreate():
    
    # object-based
    FILE_OBJECT_HOST = DIRS_DATA_TOPO + NAME_TOPO_OBJECT + 'host.txt'
    FILE_OBJECT_WALLS = DIRS_DATA_TOPO + NAME_TOPO_OBJECT + 'walls.txt'
    FILE_OBJECT_SLABS = DIRS_DATA_TOPO + NAME_TOPO_OBJECT + 'slabs.txt'
    FILE_OBJECT_INSERTS = DIRS_DATA_TOPO + NAME_TOPO_OBJECT + 'inserts.txt'

    guid_wall_host, guid_wall_inserts, guid_wall_walls, guid_wall_slabs = [],[],[],[]
    with open(FILE_OBJECT_HOST) as file:
        for line in file:
            guid_wall_host.append(line.rstrip())
    with open(FILE_OBJECT_WALLS) as file:
        for line in file:
            guid_wall_walls.append(line.rstrip())
    with open(FILE_OBJECT_SLABS) as file:
        for line in file:
            guid_wall_slabs.append(line.rstrip())
    with open(FILE_OBJECT_INSERTS) as file:
        for line in file:
            guid_wall_inserts.append(line.rstrip())
            
    guid_wall_inserts.append('') #to solve the ISSUE:miss the last line for guid_wall_inserts

    # space-based
    FILE_SPACE_HOST = DIRS_DATA_TOPO + NAME_TOPO_SPACE + 'host.txt'
    FILE_SPACE_WALLS = DIRS_DATA_TOPO + NAME_TOPO_SPACE + 'walls.txt'
    FILE_SPACE_DOORS = DIRS_DATA_TOPO + NAME_TOPO_SPACE + 'doors.txt'

    guid_space_host, guid_space_walls, guid_space_doors= [],[],[]
    with open(FILE_SPACE_HOST) as file:
        for line in file:
            guid_space_host.append(line.rstrip())
    with open(FILE_SPACE_WALLS) as file:
        for line in file:
            guid_space_walls.append(line.rstrip())
    with open(FILE_SPACE_DOORS) as file:
        for line in file:
            guid_space_doors.append(line.rstrip())

    # parameter-based
    FILE_PARAMETER_HOST = DIRS_DATA_TOPO + NAME_TOPO_PARAMETER + 'host.txt'
    FILE_PARAMETER_OBJECTS = DIRS_DATA_TOPO + NAME_TOPO_PARAMETER + 'objects.txt'

    guid_parameter_host, guid_parameter_objects = [],[]
    with open(FILE_PARAMETER_HOST) as file:
        for line in file:
            guid_parameter_host.append(line.rstrip())
    with open(FILE_PARAMETER_OBJECTS) as file:
        for line in file:
            guid_parameter_objects.append(line.rstrip())

    # Build networkx edges 
    # wall-based edges.
    guid_wall_host_indi = split_guids(guid_wall_host)
    guid_wall_walls_indi = split_guids(guid_wall_walls)
    guid_wall_inserts_indi = split_guids(guid_wall_inserts)
    guid_wall_slabs_indi = split_guids(guid_wall_slabs)

    edges_wall_h_walls = build_guid_edges(guid_wall_host_indi, guid_wall_walls_indi)
    edges_wall_h_inserts = build_guid_edges(guid_wall_host_indi, guid_wall_inserts_indi)
    edges_wall_h_slabs = build_guid_edges(guid_wall_host_indi, guid_wall_slabs_indi)

    df_edges_wall_h_walls = pd.DataFrame.from_records(edges_wall_h_walls, columns = ['host','target'])
    df_edges_wall_h_inserts = pd.DataFrame.from_records(edges_wall_h_inserts, columns = ['host','target'])
    df_edges_wall_h_slabs = pd.DataFrame.from_records(edges_wall_h_slabs, columns = ['host','target'])

    # space-based edges.
    guid_space_host_indi = split_guids(guid_space_host)
    guid_space_walls_indi = split_guids(guid_space_walls)
    guid_space_doors_indi = split_guids(guid_space_doors)

    edges_space_h_walls = build_guid_edges(guid_space_host_indi, guid_space_walls_indi)
    edges_space_h_doors = build_guid_edges(guid_space_host_indi, guid_space_doors_indi)

    df_edges_space_h_walls = pd.DataFrame.from_records(edges_space_h_walls, columns = ['host','target'])
    df_edges_space_h_doors = pd.DataFrame.from_records(edges_space_h_doors, columns = ['host','target'])

    # parameter-based edges.
    guid_parameter_host_indi = split_guids(guid_parameter_host)
    guid_parameter_objects_indi = split_guids(guid_parameter_objects)

    edges_parameter_h_objects = build_guid_edges(guid_parameter_host_indi, guid_parameter_objects_indi, set_sort=False)

    df_edges_parameter_h_objects = pd.DataFrame.from_records(edges_parameter_h_objects, columns = ['host','target'])

    # Build networkx attributes
    # object attributes
    df_doorinstances = pd.read_csv(DIRS_DATA_TOPO+'\df_Doors.csv', index_col ='ifcguid')
    df_windowinstances = pd.read_csv(DIRS_DATA_TOPO+'\df_Windows.csv', index_col ='ifcguid')
    df_wallinstances = pd.read_csv(DIRS_DATA_TOPO+'\df_Walls.csv', index_col ='ifcguid')
    df_slabinstances = pd.read_csv(DIRS_DATA_TOPO+'\df_Slabs.csv', index_col ='ifcguid')

    attrs_door = df_doorinstances.to_dict(orient = 'index')
    attrs_window = df_windowinstances.to_dict(orient = 'index')
    attrs_wall = df_wallinstances.to_dict(orient = 'index')
    attrs_slab = df_slabinstances.to_dict(orient = 'index')

    # space attributes
    df_spaceinstances = pd.read_csv(DIRS_DATA_TOPO+'\df_Spaces.csv', index_col ='ifcguid')
    attrs_space = df_spaceinstances.to_dict(orient = 'index')

    # parameter-based attibutes
    df_gp_instances = pd.read_csv(DIRS_DATA_TOPO+'\df_Parameters.csv', index_col ='name')
    attrs_gp = df_gp_instances.to_dict(orient = 'index')

    # include edges selectively.
    all_df_edges_object = [df_edges_wall_h_walls, df_edges_wall_h_slabs] # df_edges_wall_h_inserts
    all_df_edges_space = [df_edges_space_h_walls] #df_edges_space_h_doors
    all_df_edges_parameter = [df_edges_parameter_h_objects]
    all_df_edges = all_df_edges_object + all_df_edges_space + all_df_edges_parameter

    # all attributes selectively.
    all_dict_attrs = [attrs_door, attrs_window, attrs_wall, attrs_slab, attrs_space, attrs_gp] 
    G_all = build_networkx_graph(all_df_edges, all_dict_attrs)

    # - - - - - - - - - - - - - - 
    # Add failure information and search neighbors related to the failure.
    failuresIBC1020_2 = get_data_from_h5(FILE_INIT_RES, 'IBC1020_2')
    failuresIBC1207_1 = get_data_from_h5(FILE_INIT_RES, 'IBC1207_1')
    failuresIBC1207_3 = get_data_from_h5(FILE_INIT_RES, 'IBC1207_3')
    dictFailures = {
        'IBC1020_2': list(failuresIBC1020_2.loc[failuresIBC1020_2['checkCompliance'] == False, 'spaceIfcGUID'].iloc[:]),
        'IBC1207_1': list(failuresIBC1207_1.loc[failuresIBC1207_1['checkCompliance'] == False, 'spaceIfcGUID'].iloc[:]),
        'IBC1207_3': list(failuresIBC1207_3.loc[failuresIBC1207_3['checkCompliance'] == False, 'spaceIfcGUID'].iloc[:]),
    }

    # visualization of the failures.
    nodesize_map_by_object_type = {
        'Element_Door':50,
        'Element_Window':50,
        'Element_Wall':50,
        'Element_Slab':50,
        'Space':150,
        'Parameter_Global':30,
        LABEL_FAILURE_LOCATION:100,
        LABEL_FAILURE_NEIGHBOR:100,
        LABLE_ASSOCIATED_GP:30,
        }

    nodecolor_map_by_object_type = {
        'Element_Door':'green',
        'Element_Window':'skyblue',
        'Element_Wall':'darkorange',
        'Element_Slab':'yellow',
        'Space':'navy',
        'Parameter_Global':'grey',
        LABEL_FAILURE_LOCATION:'red',
        LABEL_FAILURE_NEIGHBOR:'brown',
        LABLE_ASSOCIATED_GP:'maroon',
        }

    dictGraphs = dict()
    dictFailureNeighbors = dict()
    dictAssociatedGPs = dict()

    for rule in BUILDING_RULES:
        
        # enrich the networkx with failure information.
        dictGraphs[rule], dictFailureNeighbors[rule], dictAssociatedGPs[rule] = locate_failures_per_rule(
            G_all, dictFailures, rule,
            LABLE_ASSOCIATED_GP, LABEL_FAILURE_NEIGHBOR, LABEL_FAILURE_LOCATION,
            level_neighbor=LEVEL_FAILURE_NEIGHBOR, set_exception=True)
        
        # plot the networkx.
        plot_networkx_per_rule(
            DIRS_DATA_TOPO, dictGraphs[rule], rule, nodesize_map_by_object_type, nodecolor_map_by_object_type)

    # write to csv for the failure information.
    dfInitialFailures = pd.DataFrame(dict(
        [(k,pd.Series(v)) for k,v in dictFailures.items()]))
    dfFailureNeighbors = pd.DataFrame(dict(
        [(k,pd.Series(v)) for k,v in dictFailureNeighbors.items()]))
    dfAssociatedGPs = pd.DataFrame(dict(
        [(k,pd.Series(v)) for k,v in dictAssociatedGPs.items()]))

    dfInitialFailures.to_csv(FILE_RELATED_FL_PERRULE)
    dfFailureNeighbors.to_csv(FILE_RELATED_NB_PERRULE)
    dfAssociatedGPs.to_csv(FILE_RELATED_GP_PERRULE)

    # import Global Parameter related data.
    df_gps = pd.read_csv(FILE_RELATED_GP_PERRULE, index_col = 0)                # list of related GPs.
    df_gp_values = pd.read_csv(FILE_LIST_GP, index_col = 'name')                # values of all GPs.
    df_gp_values = df_gp_values['value']

    dictGPValues = df_gp_values.to_dict()
    dictGPperRule = createDictGlobalParametersPerRule(BUILDING_RULES, df_gps)

    all_gps = list(set([y for x in list(dictGPperRule.values()) for y in x])) 
    all_gps_vals  = [round(dictGPValues[gp],3) for gp in all_gps]

    with open(FILE_RELATED_GP_INI, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(zip(all_gps, all_gps_vals))