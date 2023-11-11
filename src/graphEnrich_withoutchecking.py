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

def graphEnrich_withoutchecking(plot_graph=False):

    # - - - - - - - - - - - - - - 
    # collect the built Graph data.
    with open(FILE_INI_GRAPH, 'rb') as f:
        G_all = pickle.load(f)

    # - - - - - - - - - - - - - - 
    # visualization settings.
    nodesize_map_by_object_type = {
        'door':100,
        # 'window':100,
        # 'wall':200,
        # 'slab':200,
        'separationline':200,
        'space':250,
        }

    nodecolor_map_by_object_type = {
        'door':'green',
        # 'window':'skyblue',
        # 'wall':'darkorange',
        # 'slab':'yellow',
        'separationline':'darkorange',
        'space':'navy',
        }
    
    # - - - - - - - - - - - - - -
    # create a dictory covering a specific graph per rule.

    G = copy.deepcopy(G_all)
    
    # search neighbor.
    
    # plot the networkx.
    if plot_graph:
        plot_networkx_per_rule(
            DIRS_DATA_TOPO,
            G,
            nodesize_map_by_object_type,
            nodecolor_map_by_object_type,
            )

    # # - - - - - - - - - - - - - -
    # # import Global Parameter related data (to improve.)
    # df_gps = pd.read_csv(FILE_RELATED_GP_PERRULE, index_col = 0)                # list of related GPs.
    # df_gp_values = pd.read_csv(FILE_LIST_GP, index_col = 'name')                # values of all GPs.
    # df_gp_values = df_gp_values['value']

    # dictGPValues = df_gp_values.to_dict()
    # dictGPperRule = createDictGlobalParametersPerRule(BUILDING_RULES, df_gps)

    # all_gps = list(set([y for x in list(dictGPperRule.values()) for y in x]))
    # all_gps_vals  = [round(dictGPValues[gp],3) for gp in all_gps]

    # with open(FILE_RELATED_GP_INI, 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerows(zip(all_gps, all_gps_vals))

    # filesToMove = [f for f in os.listdir(DIRS_DATA_TOPO) if 'res_ini_' in f or '.png' in f]
    # for file in filesToMove:
    #     new_path = DIRS_DATA_TOPO + r'\neighbor_'+ str(LEVEL_FAILURE_NEIGHBOR) + '\\' + file
    #     shutil.move(os.path.join(DIRS_DATA_TOPO, file), new_path)