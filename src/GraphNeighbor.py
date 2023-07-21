#
# extractModel.py
#

#
# GraphNeighbor.py
#

# import packages
from base_external_packages import *


class GraphNeighbor():
    """
    Main class: GraphNeighbor class
    """


    def __init__(self, graph, rule):
        
        self.graph = graph                              # the whole graph.
        self.rule = rule                                # rule.
        self.failure = []                               # failures.
        self.all_failure_neighbors = []
        self.all_associated_gps = []

        self.label_search_belonging = ['wall','separationline']
        self.label_search_propagating = ['space']
        self.label_search_targeting = ['parameter']

    def __build__(self, failure_info: dict):
        
        self.all_failure_locations = failure_info[self.rule]
    
    def __restrict__(self, restriction_info: dict):
        
        self.label_search_restricting = restriction_info

    def __maxconnection__(self, n_maxconnection: int):
        
        self.limit_connection = n_maxconnection

    @property
    def strLabelGp(self):
        return 'parameter_associated'

    @property
    def strLabelFailureNeighbor(self):
        return 'failure_neighbor'

    @property
    def strLabelFailureLocation(self):
        return 'failure'
    
    @property
    def strLabelTypeAttribute(self):
        return 'classification'

    @property
    def lengthLimitation(self):
        return 12
    
    def search_belonging(self, starts):
        
        all_belongs = []
        
        if isinstance(starts, list):
            
            for start in starts:
                G_sub = nx.ego_graph(self.graph, start, radius=1, undirected=True)
                G_sub_nodes = G_sub.nodes()
                belongs = [n for n in G_sub._node if 
                        G_sub_nodes[n][self.strLabelTypeAttribute] in self.label_search_belonging]
                all_belongs += belongs
                
        else:
            start = starts
            G_sub = nx.ego_graph(self.graph, start, radius=1, undirected=True)
            G_sub_nodes = G_sub.nodes()
            belongs = [n for n in G_sub._node if 
                    G_sub_nodes[n][self.strLabelTypeAttribute] in self.label_search_belonging]
            all_belongs = belongs
        return all_belongs
    

    def search_restricting_byconnection(self, belongs):

        new_belongs =[]

        for n in belongs:
            
            all_conns = self.graph.adj[n] # get the connection map.
            all_conns_nodes = list(set(all_conns.keys())) # get the connected nodes.
            all_conns_nodes_new = [node for node in all_conns_nodes if node not in belongs] # remove the repeated nodes.
            
            type_connections = [self.graph.nodes[node]['classification'] for node in all_conns_nodes_new]
            connection_amount = sum([type_connections.count(item) for item in self.label_search_belonging])
            if connection_amount > self.limit_connection:
                continue
            else:
                new_belongs.append(n)

        return new_belongs

    def search_restricting_bylength(self, belongs):

        new_belongs =[]

        for node in belongs:
            if self.graph.nodes[node]['length'] > self.lengthLimitation:
                continue
            else:
                new_belongs.append(node)

        return new_belongs
    
    def property_restriction(self, nn, restriction):

        decision = False

        if nn[self.strLabelTypeAttribute] == restriction['link_type']:

            if isinstance(restriction['property_value'], list):
                if nn[restriction['property_type']] in restriction['property_value']:
                    decision = True
            else:
                if nn[restriction['property_type']] == restriction['property_value']:
                    decision = True
        
        return decision


    def search_restricting_property(self, belongs):

        new_belongs =[]
        graph_nodes = self.graph.nodes()
        restriction = self.label_search_restricting['belonging']
        
        for n in belongs:

            if self.property_restriction(graph_nodes[n], restriction): # if there's any potential restriction when querying properties.
                continue
            else:
                new_belongs.append(n)

        return new_belongs
        

    def search_propagating(self, belongs):
        
        new_belongs = []
        all_starts = []
        restriction = self.label_search_restricting['propagating']

        if isinstance(belongs, list):
                    
            for belong in belongs:
                G_sub = nx.ego_graph(self.graph, belong, radius=1, undirected=True)
                G_sub_nodes = G_sub.nodes()
                starts = [n for n in G_sub._node if 
                        G_sub_nodes[n][self.strLabelTypeAttribute] in self.label_search_propagating]
                
                restriction_starts = [self.property_restriction(G_sub_nodes[start], restriction) for start in starts]

                if not any(restriction_starts): # if there's any potential restriction when propagating.
                    new_belongs.append(belong)
                    all_starts += starts
                else:
                    continue
                    
        else:

            belong = belongs
            G_sub = nx.ego_graph(self.graph, belong, radius=1, undirected=True)
            G_sub_nodes = G_sub.nodes()
            starts = [n for n in G_sub._node if 
                    G_sub_nodes[n][self.strLabelTypeAttribute] in self.label_search_propagating]
            
            restriction_starts = [self.property_restriction(G_sub_nodes[start], restriction) for start in starts]

            if not any(restriction_starts): # if there's any potential restriction when propagating.
                new_belongs.append(belong)
                all_starts += starts    
    
        return new_belongs, all_starts


    def search_neighbors(self, ini_failed_id: str, level_neighbor: int):
        """
        search neighbors from an initial failure location.(space ifcguid)
        """
        
        # components
        starts = ini_failed_id
        k_real = 1  # count the accumulated level.
        k_seg = 1  # control the propagation always as 1 for each neighbor search.

        nbrs = [] # search neighbors (initial ifc.).

        while k_real <= level_neighbor:

            # search space belongings.
            belongs = self.search_belonging(starts)
            # tempo_space = starts

            # restrict the space belongings.
            belongs = self.search_restricting_bylength(belongs) # via length value.
            # belongs = self.search_restricting_byconnection(belongs) # via connection amount.
            belongs = self.search_restricting_property(belongs) # via object property.

            # propagate to new spaces and refresh the previous belongings.
            new_belongs, new_starts = self.search_propagating(belongs)
            # tempo_wall = belongs

            starts = [st for st in new_starts if st not in starts] # all propagating neighbors.
            nbrs += new_belongs    # all belonging neighbors.
            k_real += k_seg
        
        # global parameters.
        gp_nbrs = []
        gp_starts = nbrs + [ini_failed_id]

        for start in gp_starts:
            G_sub = nx.ego_graph(self.graph, start, radius=1, undirected=True)
            G_sub_nodes = G_sub.nodes()
            gp_nbr = [n for n in G_sub._node if G_sub_nodes[n]
                    [self.strLabelTypeAttribute] in self.label_search_targeting]
            gp_nbrs = gp_nbrs + gp_nbr

        return nbrs, gp_nbrs

    def search_all_neighbors(self, level_neighbor: int):
        """
        search all neighbors for all initial all_failure_locations
        """

        for failed_id in self.all_failure_locations:

            nbrs, gp_nbrs = self.search_neighbors(failed_id, level_neighbor)
            self.all_failure_neighbors += nbrs
            self.all_associated_gps += gp_nbrs
            
        self.all_failure_neighbors = list(set(self.all_failure_neighbors))
        self.all_associated_gps = list(set(self.all_associated_gps))

        # additional step: only display walls but not seplines.
        self.all_failure_neighbors = [item for item in self.all_failure_neighbors \
        if self.graph.nodes[item][self.strLabelTypeAttribute]=='wall']
   
    def update_graph_nodes(self, nn, label:str):

        """
        update the networkx graph with additonal nodes.
        """

        df_update = pd.DataFrame(nn, columns=['node_name'])
        df_update['classification'] = label
        df_update = df_update.set_index('node_name')
        df_update = df_update[~df_update.index.duplicated(keep='first')]
        attrs = df_update.to_dict(orient='index')
        nx.set_node_attributes(self.graph, attrs)
    
    def update_graph(self):

        self.update_graph_nodes(self.all_associated_gps, self.strLabelGp) # add associated global parameters.
        self.update_graph_nodes(self.all_failure_neighbors, self.strLabelFailureNeighbor)# add failure neighbors.
        self.update_graph_nodes(self.all_failure_locations, self.strLabelFailureLocation)# add failure locations.