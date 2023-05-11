import json
import pandas as pd
import numpy as np
from collections import namedtuple
import os
import copy 
import networkx as nx
import matplotlib.pyplot as plt
import pygraphviz

file_host = r'C:\dev\phd\jw\healing\data\onebuilding\res\collected_topology_1_host.txt'
file_follower = r'C:\dev\phd\jw\healing\data\onebuilding\res\collected_topology_2_follower.txt'
file_connector = r'C:\dev\phd\jw\healing\data\onebuilding\res\collected_topology_3_connector.txt'
file_storey = r'C:\dev\phd\jw\healing\data\onebuilding\res\collected_topology_4_storeys.txt'

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

def build_guid_edges(lst_host,lst_targets):
    
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
    all_edges = [sorted(x, key = lambda x:x[0]) for x in all_edges]
    all_edges = [list(i) for i in set(map(tuple, all_edges))]
    return all_edges

def build_networkx_graph(all_df_edges, all_dict_attrs):

    G_all = []
    for df in all_df_edges:
        G = nx.Graph()
        G = nx.from_pandas_edgelist(df, 'host', 'target')
        G_all.append(G)
    
    G_all = nx.compose_all(G_all)

    for dict_attrs in all_dict_attrs:
        nx.set_node_attributes(G_all, dict_attrs)

    return G_all

# https://stackoverflow.com/questions/18393842/k-th-order-neighbors-in-graph-python-networkx
def knbrs(G, start, k):
    nbrs = set([start])
    for l in range(k):
        nbrs = set((nbr for n in nbrs for nbr in G[n]))
    return list(nbrs)

guid_host, guid_follower, guid_connector, guid_storey = [],[],[],[]

with open(file_host) as file:
    for line in file:
        guid_host.append(line.rstrip())
with open(file_connector) as file:
    for line in file:
        guid_connector.append(line.rstrip())
with open(file_follower) as file:
    for line in file:
        guid_follower.append(line.rstrip())
with open(file_storey) as file:
    for line in file:
        guid_storey.append(line.rstrip())

# Build networkx edges 
guid_host_indi = split_guids(guid_host)
guid_connector_indi = split_guids(guid_connector)
guid_follower_indi = split_guids(guid_follower)
guid_storey_indi = split_guids(guid_storey)

edges_h_c = build_guid_edges(guid_host_indi, guid_connector_indi)
edges_h_f = build_guid_edges(guid_host_indi, guid_follower_indi)
edges_h_s = build_guid_edges(guid_host_indi, guid_storey_indi)

df_edges_h_c = pd.DataFrame.from_records(edges_h_c, columns = ['host','target'])
df_edges_h_f = pd.DataFrame.from_records(edges_h_f, columns = ['host','target'])
df_edges_h_s = pd.DataFrame.from_records(edges_h_s, columns = ['host','target'])

with open(r'C:\dev\phd\jw\healing\data\onebuilding\res\collected_revitinstance_and_gp.txt') as file:
    edges_instance_gp = json.load(file)

df_edges_instance_gp = pd.DataFrame.from_records(edges_instance_gp, columns = ['host','target'])

# Build networkx attributes
df_doorinstances = pd.read_csv(r'C:\dev\phd\jw\healing\data\onebuilding\res\df_door.csv', index_col ='ifcguid')
df_windowinstances = pd.read_csv(r'C:\dev\phd\jw\healing\data\onebuilding\res\df_window.csv', index_col ='ifcguid')
df_wallinstances = pd.read_csv(r'C:\dev\phd\jw\healing\data\onebuilding\res\df_wall.csv', index_col ='ifcguid')
df_floorinstances = pd.read_csv(r'C:\dev\phd\jw\healing\data\onebuilding\res\df_floor.csv', index_col ='ifcguid')

attrs_door = df_doorinstances.to_dict(orient = 'index')
attrs_window = df_windowinstances.to_dict(orient = 'index')
attrs_wall = df_wallinstances.to_dict(orient = 'index')
attrs_floor = df_floorinstances.to_dict(orient = 'index')

df_gp_instances = pd.read_csv(r'C:\dev\phd\jw\healing\data\onebuilding\res\df_gp.csv', index_col ='gp_name')
attrs_gp = df_gp_instances.to_dict(orient = 'index')

all_df_edges = [df_edges_h_c, df_edges_h_f, df_edges_h_s, df_edges_instance_gp, df_edges_instance_gp]
all_dict_attrs = [attrs_door, attrs_window, attrs_wall, attrs_floor, attrs_gp]
G_all = build_networkx_graph(all_df_edges, all_dict_attrs)

# Search neighbors related to the failure
searchlevel_neighbors = 1
failed_label = 'FailureRelatedObject'
failed_ifcuids = ['1W6Thknzf2qup40FGN1NC9','1W6Thknzf2qup40FGN1NC9','2_414Bjl90PPXdZA2Pm_Xn']
all_failure_neighbors = []

for failed_ifcuid in failed_ifcuids:
    failure_neighbors = knbrs(G_all, failed_ifcuid, k=searchlevel_neighbors)
    all_failure_neighbors.append(failure_neighbors)
all_failure_neighbors = flatten(all_failure_neighbors)

df_failureinstances = pd.DataFrame(all_failure_neighbors, columns = ['node_name'])
df_failureinstances['object_type'] = failed_label
df_failureinstances = df_failureinstances.set_index('node_name')
df_failureinstances = df_failureinstances[~df_failureinstances.index.duplicated(keep='first')]
attrs_failure = df_failureinstances.to_dict(orient = 'index')

# Add failure information
nx.set_node_attributes(G_all, attrs_failure)

# Plotting of the networkx graph
# read plotting networkx with bokeh
# https://docs.bokeh.org/en/latest/docs/user_guide/graph.html
# https://www.kaggle.com/code/anand0427/network-graph-with-at-t-data-using-plotly/notebook

nodesize_map_by_object_type = {
    'DoorObject':120,
    'WindowObject':120,
    'WallObject':120,
    'FloorObject':120,
    'GlobalParameterObject':80,
    failed_label:150,
    }

nodecolor_map_by_object_type = {
    'DoorObject': 'darkorange',
    'WindowObject': 'navy',
    'WallObject': 'darkgreen',
    'FloorObject':'gold',
    'GlobalParameterObject':'grey',
    failed_label:'red',
    }

fig = plt.figure(figsize=(25, 15))
ax = plt.axes((0.05, 0.05, 0.90, 0.90))
nx.draw_networkx(
    G_all,
    # pos = nx.nx_agraph.graphviz_layout(G_all, prog="neato"), # doesnot work
    pos = nx.kamada_kawai_layout(G_all, scale=0.75),
    with_labels=False,
    node_size = [nodesize_map_by_object_type[node[1]['object_type']] 
                    for node in G_all.nodes(data=True)],
    node_shape="o",
    node_color = [nodecolor_map_by_object_type[node[1]['object_type']] 
                    for node in G_all.nodes(data=True)],
    linewidths=0.1,
    width=0.5,
    alpha=0.80,
    edge_color='black')
ax.title.set_position([.5, 0.975])

for kk in list(nodecolor_map_by_object_type.keys()):
    plt.scatter([],[], c=nodecolor_map_by_object_type[kk], label=kk)

plt.legend()
plt.savefig(r'C:\dev\phd\jw\healing\data\onebuilding\fig\test_G_all.png', dpi=150)

# def visualization_settings(
#     composed_graph,
#     space_size=2000,
#     wall_size=1000,
#     door_size=500,
#     window_size=500):
#     """
#     Graph visualization settings of the graph
#     """

#     # color, node size
#     color_map, nodesize_map = [], []

#     for node in composed_graph:

#         # spaces
#         if node in sema_Space['spaceid'].values:

#             # room spaces
#             if 'Room' in sema_Space.loc[sema_Space['spaceid'] == node, 'spacename'].iloc[0]:
#                 nodesz = sema_Space.loc[sema_Space['spaceid']
#                                         == node, 'spacearea'].iloc[0]
#                 color_map.append('cyan')
#                 nodesize_map.append(space_size*nodesz/2)

#             # corridor spaces
#             elif 'Corridor' in sema_Space.loc[sema_Space['spaceid'] == node, 'spacename'].iloc[0]:
#                 nodesz = sema_Space.loc[sema_Space['spaceid']
#                                         == node, 'spacearea'].iloc[0]
#                 color_map.append('lightskyblue')
#                 nodesize_map.append(space_size*nodesz/2)

#         # boundary separation lines between corridor spaces
#         elif node in topo_SpaceSeparation['boundaryseplineid'].values:
#             color_map.append('navy')
#             nodesize_map.append(2000)

#         # boundary walls between spaces
#         elif node in topo_SpaceBoundary['boundarywallid'].values:
#             color_map.append('grey')
#             nodesize_map.append(wall_size)

#         # doors
#         elif node in topo_SpaceDoor['doorid'].values:
#             color_map.append('goldenrod')
#             nodesize_map.append(door_size)

#         # windows
#         elif node in topo_WallWindow['windowid'].values:
#             color_map.append('lime')
#             nodesize_map.append(window_size)

#         # others
#         else:
#             color_map.append('white')
#             nodesize_map.append(0)

#     return color_map, nodesize_map

# def visualization_compliance_settings(
#     composed_graph,
#     ck_df,
#     space_size=2000,
#     wall_size=1000,
#     door_size=500,
#     window_size=500):
#     """
#     Graph visualization settings related to the compliance checking
#     """

#     # color, node size
#     color_map, nodesize_map = [], []

#     for node in composed_graph:

#         # spaces
#         if node in sema_Space['spaceid'].values:

#             # room spaces
#             if 'Room' in sema_Space.loc[sema_Space['spaceid'] == node, 'spacename'].iloc[0]:
#                 nodesz = sema_Space.loc[sema_Space['spaceid']
#                                         == node, 'spacearea'].iloc[0]
#                 nodesize_map.append(space_size*nodesz/2)
#                 if node in ck_df['id'].values:
#                     color_map.append('r')
#                 else:
#                     color_map.append('cyan')

#         # corridor spaces
#             elif 'Corridor' in sema_Space.loc[sema_Space['spaceid'] == node, 'spacename'].iloc[0]:
#                 nodesz = sema_Space.loc[sema_Space['spaceid']
#                                         == node, 'spacearea'].iloc[0]
#                 nodesize_map.append(space_size*nodesz/2)
#                 if node in ck_df['id'].values:
#                     color_map.append('r')
#                 else:
#                     color_map.append('lightskyblue')

#         # boundary separation lines between corridor spaces
#         elif node in topo_SpaceSeparation['boundaryseplineid'].values:
#             color_map.append('navy')
#             nodesize_map.append(2000)

#         # boundary walls between spaces
#         elif node in topo_SpaceBoundary['boundarywallid'].values:
#             color_map.append('grey')
#             nodesize_map.append(wall_size)

#         # doors
#         elif node in topo_SpaceDoor['doorid'].values:
#             color_map.append('goldenrod')
#             nodesize_map.append(door_size)

#         # windows
#         elif node in topo_WallWindow['windowid'].values:
#             color_map.append('lime')
#             nodesize_map.append(window_size)

#         # others
#         else:
#             color_map.append('white')
#             nodesize_map.append(0)

#     return color_map, nodesize_map

# def displayComplianceChecking(G_topo, nr_model, ruleset):
#     """
#     Display the topology with compliance checking results
#     """

#     G_topo_SpaceDoor, G_topo_SpaceSeparation = G_topo

#     test_withchecking = nx.compose_all(
#         [G_topo_SpaceDoor, G_topo_SpaceSeparation])

#     fig = plt.figure(figsize=(30, 20))
#     ax = plt.axes((0.1, 0.1, 0.9, 0.9))
#     c_map, nz_map = visualization_compliance_settings(
#         test_withchecking, ruleset)
# #     c_map, nz_map = visualization_compliance_settings(
# #         test_withchecking, IBC1207_3)

#     nx.draw_networkx(test_withchecking, pos=nx.kamada_kawai_layout(test_withchecking),
#                      with_labels=True, node_size=nz_map, node_color=c_map,
#                      node_shape="o", linewidths=0.2, width=2, alpha=0.98, edge_color='black')

#     ax.title.set_position([.5, 0.975])
#     ax.set_title(str(nr_model) + '-' +
#                  ruleset['checkrule'].iloc[0], fontsize=20)
#     plt.savefig('figures/' + ruleset['checkrule'].iloc[0] + '/' + str(nr_model) + '-' +
#                 ruleset['checkrule'].iloc[0] + '.png', dpi=150)
# #     fig.show()