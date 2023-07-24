#
# graphEnricher.py
#

# import modules
from const_project import FILE_INIT_RES, DIRS_DATA_TOPO, FILE_INI_GRAPH, FILE_CONSTRAINTS, FILE_CONSTRAINTS_APPLY
from const_project import FILE_RELATED_EL_CONSTRAINT_PERRULE, FILE_RELATED_GP_CONSTRAINT_PERRULE

from const_project import FILE_LIST_GP, FILE_RELATED_GP_PERRULE, FILE_RELATED_FL_PERRULE, FILE_RELATED_EL_PERRULE, FILE_RELATED_GP_INI

from const_ibcrule import BUILDING_RULES, LABEL_FAILURE_LOCATION, LABEL_FAILURE_NEIGHBOR, LABLE_ASSOCIATED_GP
from const_ibcrule import LEVEL_FAILURE_NEIGHBOR, dictRestriction

from funct_data import get_data_from_h5, write_dict_tocsv
from funct_topo import *

from GraphNeighbor import GraphNeighbor

def graphEnrich(plot_graph=False):

    # - - - - - - - - - - - - - - 
    # collect the built Graph data.
    with open(FILE_INI_GRAPH, 'rb') as f:
        G_all = pickle.load(f)

    file_mapping = DIRS_DATA_TOPO+'\df_parameter.csv'

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
    dictFailureElements = dict()
    dictFailureGlobalParameters = dict()
    dictFailureElementsConstrained = dict()
    dictFailureGlobalParametersConstrained = dict()

    # Add failure information and search neighbors related to the failure.
    failuresIBC1020_2 = get_data_from_h5(FILE_INIT_RES, 'IBC1020_2')
    failuresIBC1207_1 = get_data_from_h5(FILE_INIT_RES, 'IBC1207_1')
    failuresIBC1207_3 = get_data_from_h5(FILE_INIT_RES, 'IBC1207_3')
    dictFailures = {
        'IBC1020_2': list(failuresIBC1020_2.loc[failuresIBC1020_2['checkCompliance'] == False, 'spaceIfcGUID'].iloc[:]),
        'IBC1207_1': list(failuresIBC1207_1.loc[failuresIBC1207_1['checkCompliance'] == False, 'spaceIfcGUID'].iloc[:]),
        'IBC1207_3': list(failuresIBC1207_3.loc[failuresIBC1207_3['checkCompliance'] == False, 'spaceIfcGUID'].iloc[:]),
    }

    dictConstraints = {}

    # enrich per rule.
    for rule in BUILDING_RULES:

        G = copy.deepcopy(G_all)
        graphNeighborhood = GraphNeighbor(G, rule)

        graphNeighborhood.__build__(dictFailures)
        graphNeighborhood.__mapping__(file_mapping)
        graphNeighborhood.__restrict__(dictRestriction, 'wall')

        graphNeighborhood.__maxconnection__(n_maxconnection=5)
        
        # search neighbor.
        graphNeighborhood.search_all_neighbors(level_neighbor=LEVEL_FAILURE_NEIGHBOR)

        # analyze constraints as .json
        # till now we use the results from IBC1020_2 to apply constraints (for all three rule checking parts via looping).
        if rule == BUILDING_RULES[0]:
            graphNeighborhood.analyze_constraints()
            graphNeighborhood.create_constraints(FILE_CONSTRAINTS)
        
        dictConstraints.update({rule: graphNeighborhood.apply_constraints(FILE_CONSTRAINTS)})
        graphNeighborhood.update_graph()

        dictGraphs[rule] = graphNeighborhood.graph
        dictFailureElements[rule] = graphNeighborhood.all_failure_elements
        dictFailureGlobalParameters[rule] = graphNeighborhood.all_associated_gps
        dictFailureElementsConstrained[rule] = graphNeighborhood.constrained_neighbor_elements
        dictFailureGlobalParametersConstrained[rule] = graphNeighborhood.constrained_neighbor_gps

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
    # write the applied constraints
    with open(FILE_CONSTRAINTS_APPLY, "w") as outfile:
        json.dump(dictConstraints, outfile)

    # - - - - - - - - - - - - - -
    # write to csv for the failure information.

    write_dict_tocsv(dictFailures, FILE_RELATED_FL_PERRULE)
    write_dict_tocsv(dictFailureElements, FILE_RELATED_EL_PERRULE)
    write_dict_tocsv(dictFailureGlobalParameters, FILE_RELATED_GP_PERRULE)
    write_dict_tocsv(dictFailureElementsConstrained, FILE_RELATED_EL_CONSTRAINT_PERRULE)
    write_dict_tocsv(dictFailureGlobalParametersConstrained, FILE_RELATED_GP_CONSTRAINT_PERRULE)

    # - - - - - - - - - - - - - -
    # import Global Parameter related data (to improve.)
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