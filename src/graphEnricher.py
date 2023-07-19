#
# graphEnricher.py
#

# import modules
from const_project import FILE_INIT_RES, DIRS_DATA_TOPO, FILE_INI_GRAPH
from const_project import FILE_LIST_GP, FILE_RELATED_GP_PERRULE, FILE_RELATED_FL_PERRULE, FILE_RELATED_NB_PERRULE, FILE_RELATED_GP_INI

from const_ibcrule import BUILDING_RULES, LABEL_FAILURE_LOCATION, LABEL_FAILURE_NEIGHBOR, LABLE_ASSOCIATED_GP
from const_ibcrule import LEVEL_FAILURE_NEIGHBOR, EXCEPTION_LINK_SEQUENCE, EXCEPTION_LINK_TYPES

from funct_data import get_data_from_h5
from funct_topo import *

def graphEnrich(plot_graph=False):

    # - - - - - - - - - - - - - - 
    # collect the built Graph data.
    with open(FILE_INI_GRAPH, 'rb') as f:
        G_all = pickle.load(f)


    # n_aj =  G_all.adj['3575064']
    # for k in n_aj:
    #     print (G_all.nodes[k]['classification'])

    # - - - - - - - - - - - - - - 
    # visualization settings.
    nodesize_map_by_object_type = {
        'door':50,
        'window':50,
        'wall':50,
        'slab':50,
        'separationline':50,
        'space':150,
        'parameter':30,
        LABEL_FAILURE_LOCATION:100,
        LABEL_FAILURE_NEIGHBOR:100,
        LABLE_ASSOCIATED_GP:30,
        }

    nodecolor_map_by_object_type = {
        'door':'green',
        'window':'skyblue',
        'wall':'darkorange',
        'slab':'yellow',
        'separationline':'cyan',
        'space':'navy',
        'parameter':'grey',
        LABEL_FAILURE_LOCATION:'red',
        LABEL_FAILURE_NEIGHBOR:'brown',
        LABLE_ASSOCIATED_GP:'maroon',
        }
    
    # - - - - - - - - - - - - - -
    # create a dictory covering a specific graph per rule.
    dictGraphs = dict()
    dictFailureNeighbors = dict()
    dictAssociatedGPs = dict()

    # Add failure information and search neighbors related to the failure.
    failuresIBC1020_2 = get_data_from_h5(FILE_INIT_RES, 'IBC1020_2')
    failuresIBC1207_1 = get_data_from_h5(FILE_INIT_RES, 'IBC1207_1')
    failuresIBC1207_3 = get_data_from_h5(FILE_INIT_RES, 'IBC1207_3')
    dictFailures = {
        'IBC1020_2': list(failuresIBC1020_2.loc[failuresIBC1020_2['checkCompliance'] == False, 'spaceIfcGUID'].iloc[:]),
        'IBC1207_1': list(failuresIBC1207_1.loc[failuresIBC1207_1['checkCompliance'] == False, 'spaceIfcGUID'].iloc[:]),
        'IBC1207_3': list(failuresIBC1207_3.loc[failuresIBC1207_3['checkCompliance'] == False, 'spaceIfcGUID'].iloc[:]),
    }

    # specify the propagration rule.

    exception_class_linkage, exception_name_linkage, exception_value_linkage  = build_link_constraints(
        EXCEPTION_LINK_TYPES, LEVEL_FAILURE_NEIGHBOR, EXCEPTION_LINK_SEQUENCE)

    # enrich per rule.
    for rule in BUILDING_RULES:
        
        # enrich the networkx with failure information.
        dictGraphs[rule], dictFailureNeighbors[rule], dictAssociatedGPs[rule] = locate_failures_per_rule(
            G_all,
            dictFailures,
            rule,
            LABLE_ASSOCIATED_GP,
            LABEL_FAILURE_NEIGHBOR,
            LABEL_FAILURE_LOCATION,
            exception_class_linkage,
            exception_name_linkage,
            exception_value_linkage,
            LEVEL_FAILURE_NEIGHBOR,
            )
        
        # plot the networkx.
        if plot_graph:
            plot_networkx_per_rule(
                DIRS_DATA_TOPO,
                rule,
                dictGraphs[rule],
                nodesize_map_by_object_type,
                nodecolor_map_by_object_type,
                )

    # - - - - - - - - - - - - - -
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

    # - - - - - - - - - - - - - -
    # to improve.
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

    filesToMove = [f for f in os.listdir(DIRS_DATA_TOPO) if 'res_ini_' in f or '.png' in f]
    for file in filesToMove:
        new_path = DIRS_DATA_TOPO + r'\neighbor_'+ str(LEVEL_FAILURE_NEIGHBOR) + '\\' + file
        shutil.move(os.path.join(DIRS_DATA_TOPO, file), new_path)