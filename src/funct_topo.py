#
# funct_plot.py
#

# import packages
from base_external_packages import *


def create_dictionary_key_mapping(dictionary):
    """
    """

    ini_keys = list(dictionary[0].keys())
    new_keys =  [key.replace(" ", "_") for key in ini_keys]
    lower_new_keys = [key.lower() for key in new_keys]
    door_type_parametername_map = dict(zip(ini_keys, lower_new_keys))
    return door_type_parametername_map


def map_dictionary_keys(dictionary,mapping):
    """
    """
    
    
    ini_keys = list(dictionary[0].keys())
    new_keys = [mapping[key] for key in ini_keys]
    
    for row in dictionary:
        # for each type
        for ini_key,new_key in zip(ini_keys, new_keys):
            row[new_key] = row.pop(ini_key)
    return dictionary


def read_dictionary_to_classobjects(dictionary,class_name='X'):
    """
    """
    
    

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
    """
    """
    
    
    if string_mapping:
        parametername_map = create_dictionary_key_mapping(dict)
        dictionary_mapped = map_dictionary_keys(dict, parametername_map)
    else:
        dictionary_mapped = dict
    class_objects = read_dictionary_to_classobjects(dictionary_mapped, class_name=class_name)

    return class_objects


def convert_clsobjs_into_df(cls_objs):
    """
    """
    
    

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
    """
    """
    
    
    
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


def get_data_from_h5(h5doc, key):
    """
    Collect data from .h5 file by specifying the store key.
    """

    allData = pd.HDFStore(h5doc, 'r')
    data = allData[key]
    return data

def flatten(list):
    return [item for sublist in list for item in sublist]


def split_guids(guids, separator=','):
    
    guid_multilist = copy.deepcopy(guids)
    for ii in range(len(guid_multilist)):
        if separator in guid_multilist[ii]:
            guid_multilist[ii] = guid_multilist[ii].split(separator)
        elif guid_multilist[ii]:
            guid_multilist[ii] = [guid_multilist[ii]]
        else:
            continue

    return guid_multilist


def build_guid_edges(lst_host,lst_targets,set_sort=True):
    
    all_edges = []
    if len(lst_host) != len(lst_targets):
        return all_edges
    else:
        for host,targets in zip(lst_host,lst_targets):
            edges_per_host = []
            actual_targets = [tt for tt in targets if (tt != host[0] and tt)]
            edges_per_host = [[host[0],target] for target in actual_targets]
            all_edges.append(edges_per_host)
    
    all_edges = flatten(all_edges)
    if set_sort:
        all_edges = [sorted(x, key = lambda x:x[0]) for x in all_edges]
    all_edges = [list(i) for i in set(map(tuple, all_edges))]
    return all_edges


def build_networkx_graph(all_df_edges, all_dict_attrs=[]):

    G_all = []
    for df in all_df_edges:
        G = nx.Graph()
        G = nx.from_pandas_edgelist(df, 'host', 'target')
        G_all.append(G)
    
    G_all = nx.compose_all(G_all)

    if all_dict_attrs:
        for dict_attrs in all_dict_attrs:
            nx.set_node_attributes(G_all, dict_attrs)

    return G_all

# to debug the sequential search.
def knbrs_subgraph(
        G, ini_starts, k=1, 
        classname = 'classification',
        class_target = 'Parameter_Global',
        class_link = ['Element_Wall'],
        set_link_exception = [False],
        link_exceptionname = ['isexternal'],
        link_exception_value = [1],
        ):
    """
    search neighbors of specified nodes within subgraphs of a graph.
    G: the whole Graph;
    ini_strats: starting points.
    k: level of neighborhood;
    classname: name of the attribute which is used as node class identity;
    class_target: the class of the final targets that we search for;
    set_link_exception:
    class_link: the class(es) of the linking components;
    link_exceptionname:
    link_exception_value:
    """

    def get_tempo_data(lst, k_track, k):
        """
        get the temporary data at k_track
        """
        k_lst = []
        if isinstance(lst, list):
            if len(lst) == k:
                if len(lst) > 1:
                    k_lst = lst[k_track]
                elif len(lst) == 1:
                    k_lst = lst[0]
        else:
            k_lst = lst
        return k_lst
    
    starts = ini_starts
    k_real = 1 # count the accumulated level.
    k_seg = 1 # control the propagation always as 1 for each neighbor search.
    
    # search Element neighbors (initial excluded).
    nbrs = []

    # track the linking class at current level of neighborhood.
    k_track = 0
    
    while k_real <= k:
        
        # if class_link includes a sequential data.
        k_class_link = get_tempo_data(class_link, k_track, k)
        k_set_link_exception = get_tempo_data(set_link_exception, k_track, k)
        k_link_exceptionname = get_tempo_data(link_exceptionname, k_track, k)
        k_link_exception_value = get_tempo_data(link_exception_value, k_track, k)
        k_track+=1

        if isinstance(starts, list):

            # if there are multiple input starting points for the current round:
            for start in starts:
                G_sub = nx.ego_graph(G, start, radius=k_seg, undirected=True)
                G_sub_nodes = G_sub.nodes()

                if k_set_link_exception:
                    nbr = [n for n in G_sub._node if G_sub_nodes[n][classname] == k_class_link and G_sub_nodes[n][k_link_exceptionname]!=k_link_exception_value]
                else:
                    nbr = [n for n in G_sub._node if G_sub_nodes[n][classname] == k_class_link]
                nbrs = nbrs + nbr
        else:
            
            # if there is only one single input start point for the current round:
            G_sub = nx.ego_graph(G, starts, radius=k_seg, undirected=True)
            G_sub_nodes = G_sub.nodes()

            if k_set_link_exception:
                nbr = [n for n in G_sub._node if G_sub_nodes[n][classname] == k_class_link and G_sub_nodes[n][k_link_exceptionname]!=k_link_exception_value]
            else:
                nbr = [n for n in G_sub._node if G_sub_nodes[n][classname] == k_class_link]
            nbrs = nbrs + nbr

        k_real += k_seg
        starts = nbrs

    # search associated Parameter neighbors.(initial included).
    gp_nbrs = []
    gp_starts = starts + [ini_starts]
    for start in gp_starts:
        G_sub = nx.ego_graph(G, start, radius=k_seg, undirected=True)
        G_sub_nodes = G_sub.nodes()
        gp_nbr = [n for n in G_sub._node if G_sub_nodes[n][classname]==class_target]
        gp_nbrs = gp_nbrs + gp_nbr

    return nbrs, gp_nbrs


def update_graph_nodes(G, nn, label):
    """
    update the networkx graph with additonal nodes.
    """

    df_update = pd.DataFrame(nn, columns = ['node_name'])
    df_update['classification'] = label
    df_update = df_update.set_index('node_name')
    df_update = df_update[~df_update.index.duplicated(keep='first')]
    attrs = df_update.to_dict(orient = 'index')
    nx.set_node_attributes(G, attrs)


def locate_failures_per_rule(
        G_ini, all_failures, rule,
        label_gps, label_neighbors, label_locations,
        level_neighbor=1,
        class_link = ['Element_Wall'],
        set_link_exception=[False],
        link_exceptionname = ['isexternal'],
        link_exception_value = [1],):
    """
    locate the 
    - failure locations;
    - failure neighbors;
    - related Global Parameters.
    """
    
    # duplicate the initial network.
    G = copy.deepcopy(G_ini)

    # selecte the failures per rule.
    all_failure_locations = all_failures[rule]

    # Search neighbors and associated GPs.
    all_failure_neighbors, all_associated_gps= [],[]
    for failed_ifcuid in all_failure_locations:
        failure_neighbors, associated_gps = knbrs_subgraph(
            G, failed_ifcuid, k=level_neighbor,
            classname= 'classification',
            class_target='Parameter_Global',
            class_link = class_link,
            set_link_exception = set_link_exception,
            link_exceptionname = link_exceptionname,
            link_exception_value = link_exception_value,
            )
    
        all_failure_neighbors += failure_neighbors
        all_associated_gps += associated_gps
    
    all_failure_neighbors = list(set(all_failure_neighbors))    # remove duplicated from list
    all_associated_gps = list(set(all_associated_gps))

    update_graph_nodes(G, all_associated_gps, label_gps)
    update_graph_nodes(G, all_failure_neighbors, label_neighbors) # add failure neighbors
    update_graph_nodes(G, all_failure_locations, label_locations) # add failure locations.
    
    return G, all_failure_neighbors, all_associated_gps

# - - - - - - - - - - - - - - 
# Extra plotting of the networkx graph
# read plotting networkx with bokeh
# https://docs.bokeh.org/en/latest/docs/user_guide/graph.html
# https://www.kaggle.com/code/anand0427/network-graph-with-at-t-data-using-plotly/notebook

def plot_networkx_per_rule(
        path, G, rule, nodesize_map, nodecolor_map):
    """
    plot the networkx graph with specified maps for node size and node color.
    """

    fig = plt.figure(figsize=(30, 18))
    ax = plt.axes((0.05, 0.05, 0.90, 0.90))
    G_nodes = G.nodes()
    G_nodes_sizes = [nodesize_map[G.nodes[n]['classification']] for n in G_nodes]
    G_nodes_colors = [nodecolor_map[G.nodes[n]['classification']] for n in G_nodes]

    nx.draw_networkx(
        G,
        # pos = nx.nx_agraph.graphviz_layout(G, prog="neato"), # doesnot work
        pos = nx.kamada_kawai_layout(G, scale=0.75),
        with_labels=False,
        node_size=G_nodes_sizes,
        node_shape="o",
        node_color=G_nodes_colors,
        linewidths=0.1,
        width=0.5,
        alpha=0.80,
        edge_color='black')
    ax.title.set_position([.5, 0.975])

    for kk in list(nodecolor_map.keys()):
        plt.scatter([],[], c=nodecolor_map[kk], label=kk)

    plt.legend()
    plt.savefig(path + '\\G_' + str(rule) + '.png', dpi=200)


def createDictGlobalParametersPerRule(rules, df_gps):
    """
    """
    
    dictGPperRule = dict()
    for rule in rules:
        lst_gps_per_rule = [gp for gp in df_gps[rule].tolist() if str(gp) != 'nan']
        dictGPperRule[rule] =  lst_gps_per_rule

    return dictGPperRule