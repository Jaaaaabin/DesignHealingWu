#
# funct_plot.py
#

# import packages
from base_external_packages import *

def create_dictionary_key_mapping(dictionary):
    """
    """

    ini_keys = list(dictionary[0].keys())
    new_keys = [key.replace(" ", "_") for key in ini_keys]
    lower_new_keys = [key.lower() for key in new_keys]
    door_type_parametername_map = dict(zip(ini_keys, lower_new_keys))
    return door_type_parametername_map


def map_dictionary_keys(dictionary, mapping):
    """
    """

    ini_keys = list(dictionary[0].keys())
    new_keys = [mapping[key] for key in ini_keys]

    for row in dictionary:
        # for each type
        for ini_key, new_key in zip(ini_keys, new_keys):
            row[new_key] = row.pop(ini_key)
    return dictionary


def read_dictionary_to_classobjects(dictionary, class_name='X'):
    """
    """

    objects = []
    for row in dictionary:
        # for each type
        row_in_string = json.dumps(row)
        object = json.loads(
            row_in_string,
            object_hook=lambda d: namedtuple(class_name, d.keys())(*d.values()))
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
    class_objects = read_dictionary_to_classobjects(
        dictionary_mapped, class_name=class_name)

    return class_objects


def convert_clsobjs_into_df(cls_objs):
    """
    """

    # Get the class attributes.
    attributes = dir(cls_objs[0])

    # Clean unrelevant attributes.
    attributes = [att for att in attributes if not att.startswith('_')]
    attributes = [att for att in attributes if att !=
                  'index' and att != 'count']

    # Convert to pandasDataFrame.
    df = pd.DataFrame([[getattr(obj, att) for att in attributes]
                      for obj in cls_objs], columns=attributes)

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
        df['classification'] = df.apply(
            lambda x: x['name'].rsplit('_', 1)[0], axis=1)
        df['classification'] = df.apply(
            lambda x: 'Space_' + x['classification'], axis=1)
    return df

def flatten(list):
    return [item for sublist in list for item in sublist]


def split_guids(guids, separator=',', remove_repeat=False):

    guid_multilist = copy.deepcopy(guids)
    for ii in range(len(guid_multilist)):
        if separator in guid_multilist[ii]:
            guid_multilist[ii] = guid_multilist[ii].split(separator)
        elif guid_multilist[ii]:
            guid_multilist[ii] = [guid_multilist[ii]]
        else:
            continue
    
    if remove_repeat:
        guid_multilist = [list(set(l)) for l in guid_multilist]
        
    return guid_multilist


def build_guid_edges(
        lst_host, lst_targets, set_sort=True):
    
    all_edges = []
    if len(lst_host) != len(lst_targets):
        return all_edges
    else:
        for host, targets in zip(lst_host, lst_targets):
            edges_per_host = []
            actual_targets = [tt for tt in targets if (tt != host[0] and tt)]
            edges_per_host = [[host[0], target] for target in actual_targets]
            all_edges.append(edges_per_host)

    all_edges = flatten(all_edges)
    if set_sort:
        all_edges = [sorted(x, key=lambda x:x[0]) for x in all_edges]
    all_edges = [list(i) for i in set(map(tuple, all_edges))]
    return all_edges


def build_networkx_graph(
        all_df_edges, all_dict_attrs=[]):

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


def get_tempo_data(
        lst, k_track, k):
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
            print('the input data doest fit the assigned link level.')
    else:
        k_lst = lst
    return k_lst


def build_link_constraints(
    link_dict,
    level=0,
    link_type_seq=[],
    ):
    """
    build link exceptions/constraints.
    """

    if len(link_type_seq) == level:
        class_linkage, exception_name_linkage, exceptionvalue_linkage = [], [], []

        for link_type in link_type_seq:

            if isinstance(link_type, list) and len(link_type) != 1:
                class_linkage.append(
                    [link_dict[sub_link_type]['link_type'] for sub_link_type in link_type])
                exception_name_linkage.append(
                    [link_dict[sub_link_type]['property_type'] for sub_link_type in link_type])
                exceptionvalue_linkage.append(
                    [link_dict[sub_link_type]['property_value'] for sub_link_type in link_type])
            else:
                class_linkage.append(link_dict[link_type]['link_type'])
                exception_name_linkage.append(
                    link_dict[link_type]['property_type'])
                exceptionvalue_linkage.append(
                    link_dict[link_type]['property_value'])

        return class_linkage, exception_name_linkage, exceptionvalue_linkage

    else:
        return [], [], []


def propa_with_constraints(
    G,
    start,
    k_class_link,
    k_link_exceptionname,
    k_link_exceptionvalue,
    k_seg=1,
    set_undirected=True,
    classname='classification',
    ):
    """
    """

    G_sub = nx.ego_graph(G, start, radius=k_seg, undirected=set_undirected)
    G_sub_nodes = G_sub.nodes()

    if isinstance(k_class_link, list) and len(k_class_link) != 1:

        # if
        # there are multiple link constraints:

        tempo_nbr_all = []
        for one_class, one_exceptionname, one_exceptionvalue in zip(k_class_link, k_link_exceptionname, k_link_exceptionvalue):
            tempo_nbr_single = [
                n for n in G_sub._node if G_sub_nodes[n][classname] == one_class and G_sub_nodes[n][one_exceptionname] != one_exceptionvalue]
            tempo_nbr_all.append(tempo_nbr_single)

        # find intersection among all constraint-based linkages.
        nbr = []
        for i in range(len(tempo_nbr_all)):
            nbr = list(set(nbr).intersection(tempo_nbr_all[i]) if nbr else tempo_nbr_all[i])
        
    else:
        # if
        # there is only one link constraint.

        nbr = [
            n for n in G_sub._node if G_sub_nodes[n][classname] == k_class_link and G_sub_nodes[n][k_link_exceptionname] != k_link_exceptionvalue]
    return nbr


def propa_connection_limit(
    G,
    start,
    connection_classification=None,
    connection_n=5,
    start_type='wall',):
    """
    """
    
    propa_decision = True

    if connection_classification is not None and G.nodes[start]['classification'] == start_type: 
        all_conns = G.adj[start]
        connections = [G.nodes[conn]['classification'] for conn in all_conns]

        if connections.count(connection_classification) >= connection_n:
            propa_decision = False

    return propa_decision


def knbrs_subgraph(
        G,
        ini_starts,
        class_link,
        link_exceptionname,
        link_exceptionvalue,
        classname='classification',
        class_target='parameter',
        k=1,
        ):
    """
    search neighbors of specified nodes within subgraphs of a graph.
    G: the whole Graph;
    ini_strats: starting points.
    set_link_exception:
    class_link: the class(es) of the linking components;
    link_exceptionname:
    link_exceptionvalue:
    classname: name of the attribute which is used as node class identity;
    class_target: the class of the final targets that we search for;
    k: level of neighborhood;
    """

    starts = ini_starts
    k_real = 1  # count the accumulated level.
    k_seg = 1  # control the propagation always as 1 for each neighbor search.

    # search neighbors (initial ifc.).
    nbrs = []

    # track the linking class at current level of neighborhood.
    k_track = 0

    while k_real <= k:

        # if class_link includes a sequential data.
        k_class_link = get_tempo_data(class_link, k_track, k)
        k_link_exceptionname = get_tempo_data(link_exceptionname, k_track, k)
        k_link_exceptionvalue = get_tempo_data(link_exceptionvalue, k_track, k)
        k_track += 1

        new_nbrs = []
        if isinstance(starts, list):
            
            # if
            # there are multiple starting points as input for the current search round:
            for start in starts:

                # use <start>
                # when k_real=2 no need to check.
                if propa_connection_limit(G, start, connection_classification='space') or k_real == 2:

                    nbr = propa_with_constraints(
                        G,
                        start,
                        k_class_link,
                        k_link_exceptionname,
                        k_link_exceptionvalue,
                        )
                    new_nbrs = new_nbrs + nbr

                # # -
                # G_sub = nx.ego_graph(G, start, radius=k_seg, undirected=True)
                # G_sub_nodes = G_sub.nodes()
                # nbr = [n for n in G_sub._node if G_sub_nodes[n][classname] ==
                #        k_class_link and G_sub_nodes[n][k_link_exceptionname] != k_link_exceptionvalue]
                # # -

        else:
            
            # if
            # there is only one single pointas input for the current search round:
            # use <starts>

            if propa_connection_limit(G, starts, connection_classification='space') or k_real == 2:
            
                nbr = propa_with_constraints(
                    G,
                    starts,
                    k_class_link,
                    k_link_exceptionname,
                    k_link_exceptionvalue,
                    )
                new_nbrs = nbr

            # # -
            # G_sub = nx.ego_graph(G, starts, radius=k_seg, undirected=True)
            # G_sub_nodes = G_sub.nodes()
            # nbr = [n for n in G_sub._node if G_sub_nodes[n][classname] ==
            #        k_class_link and G_sub_nodes[n][k_link_exceptionname] != k_link_exceptionvalue]
            # # -

        nbrs += new_nbrs    # all neighbors.
        starts = new_nbrs   # new starting points.
        k_real += k_seg

    # search associated Parameter neighbors.(initial included).
    gp_nbrs = []
    gp_starts = starts + [ini_starts]
    for start in gp_starts:
        G_sub = nx.ego_graph(G, start, radius=k_seg, undirected=True)
        G_sub_nodes = G_sub.nodes()
        gp_nbr = [n for n in G_sub._node if G_sub_nodes[n]
                  [classname] == class_target]
        gp_nbrs = gp_nbrs + gp_nbr

    return nbrs, gp_nbrs


def update_graph_nodes(G, nn, label):
    """
    update the networkx graph with additonal nodes.
    """

    df_update = pd.DataFrame(nn, columns=['node_name'])
    df_update['classification'] = label
    df_update = df_update.set_index('node_name')
    df_update = df_update[~df_update.index.duplicated(keep='first')]
    attrs = df_update.to_dict(orient='index')
    nx.set_node_attributes(G, attrs)


def locate_failures_per_rule(
    G_ini,
    all_failures,
    rule,
    label_gps,
    label_neighbors,
    label_locations,
    class_link,
    link_exceptionname,
    link_exceptionvalue,
    level_neighbor=1,):
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
    all_failure_neighbors, all_associated_gps = [], []
    for failed_ifcuid in all_failure_locations:
        failure_neighbors, associated_gps = knbrs_subgraph(
            G,
            failed_ifcuid,
            class_link,
            link_exceptionname,
            link_exceptionvalue,
            k=level_neighbor,
        )

        all_failure_neighbors += failure_neighbors
        all_associated_gps += associated_gps

    # remove duplicated from list
    all_failure_neighbors = list(set(all_failure_neighbors))
    all_associated_gps = list(set(all_associated_gps))

    update_graph_nodes(G, all_associated_gps, label_gps)
    update_graph_nodes(G, all_failure_neighbors,
                       label_neighbors)  # add failure neighbors
    # add failure locations.
    update_graph_nodes(G, all_failure_locations, label_locations)

    return G, all_failure_neighbors, all_associated_gps

# - - - - - - - - - - - - - -
# Extra plotting of the networkx graph
# read plotting networkx with bokeh
# https://docs.bokeh.org/en/latest/docs/user_guide/graph.html
# https://www.kaggle.com/code/anand0427/network-graph-with-at-t-data-using-plotly/notebook


def plot_networkx_per_rule(
        path,
        rule,
        G,
        nodesize_map,
        nodecolor_map):
    """
    plot the networkx graph with specified maps for node size and node color.
    """

    fig = plt.figure(figsize=(30, 18))
    ax = plt.axes((0.05, 0.05, 0.90, 0.90))
    G_nodes = G.nodes()
    G_nodes_sizes = [nodesize_map[G.nodes[n]['classification']]
                     for n in G_nodes]
    G_nodes_colors = [nodecolor_map[G.nodes[n]['classification']]
                      for n in G_nodes]

    nx.draw_networkx(
        G,
        pos=nx.kamada_kawai_layout(G, scale=0.75),
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
        plt.scatter([], [], c=nodecolor_map[kk], label=kk)

    plt.legend()
    plt.savefig(path + '\\res_G_' + str(rule) + '.png', dpi=200)


def createDictGlobalParametersPerRule(rules, df_gps):
    """
    """

    dictGPperRule = dict()
    for rule in rules:
        lst_gps_per_rule = [
            gp for gp in df_gps[rule].tolist() if str(gp) != 'nan']
        dictGPperRule[rule] = lst_gps_per_rule

    return dictGPperRule
