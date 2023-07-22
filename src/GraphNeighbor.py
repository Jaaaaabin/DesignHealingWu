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

        # self.label_search_belonging = ['wall','separationline'] # much faster.
        self.label_search_belonging = ['wall']          # search step - beloning
        self.label_search_propagating = ['space']       # search step - propagating
        self.label_search_targeting = ['parameter']     # search step - targeting

        self.constraints = dict()


    def __build__(self, failure_info: dict):
        
        self.all_failure_locations = failure_info[self.rule]

    def __mapping__(self, file_mapping):

        self.df_map_from_element = pd.read_csv(file_mapping, index_col ='elementifcguid')
        self.df_map_from_gp = pd.read_csv(file_mapping, index_col ='name')
    
    def __maxconnection__(self, n_maxconnection: int):
        
        self.limit_connection = n_maxconnection


    def __restrict__(self, restriction_info: dict, buildingelement_type: str):
        
        self.search_restriction = restriction_info
        
        graph_nodes = self.graph.nodes()

        for restriction_key in restriction_info.keys():

            restriction  = self.search_restriction[restriction_key]
            constrained_nodes =[]
            
            # search constrained nodes.
            for node in graph_nodes:

                if graph_nodes[node][self.strLabelTypeAttribute] == restriction['constraint_object']:
                    if isinstance(restriction['property_value'], list):
                        if graph_nodes[node][restriction['property_type']] in restriction['property_value']:
                            constrained_nodes.append(node)
                    else:
                        if graph_nodes[node][restriction['property_type']] == restriction['property_value']:
                            constrained_nodes.append(node)
            
            # stop or find target building element.
            if restriction['constraint_object'] == buildingelement_type:
                target_constrained_nodes = constrained_nodes

            else:
                
                target_constrained_nodes = []

                for node in constrained_nodes:
                    
                    G_sub = nx.ego_graph(self.graph, node, radius=1, undirected=True)
                    G_sub_nodes = G_sub.nodes()
                    targets = [n for n in G_sub._node if G_sub_nodes[n][self.strLabelTypeAttribute] == buildingelement_type]
                    target_constrained_nodes.append(targets)

            self.search_restriction[restriction_key].update({'constrained_nodes':target_constrained_nodes})

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
    
    # search step - beloning
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
    
    # search step - beloning
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
        """
        this function could be removed since we already have all info about the constrained nodes.
        will influence:
        search_restricting_property()
        search_propagating()
        """

        decision = False

        if nn[self.strLabelTypeAttribute] == restriction['constraint_object']:

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
        restriction = self.search_restriction['belonging']
        
        for n in belongs:

            if self.property_restriction(graph_nodes[n], restriction): # if there's any potential restriction when querying properties.
                continue
            else:
                new_belongs.append(n)

        return new_belongs
        

    def search_propagating(self, belongs):
        
        new_belongs = []
        all_starts = []
        restriction = self.search_restriction['propagating']

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
        problem : very slow after level>=5.
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


    def apply_constraints(self, restriction_key: str, target_object='wall'):

        graph_nodes = self.graph.nodes()
        restriction  = self.search_restriction[restriction_key]
        
        constrained_nodes =[]

        for node in graph_nodes:

            if graph_nodes[node][self.strLabelTypeAttribute] == restriction['constraint_object']:
                if isinstance(restriction['property_value'], list):
                    if graph_nodes[node][restriction['property_type']] in restriction['property_value']:
                        constrained_nodes.append(node)
                else:
                    if graph_nodes[node][restriction['property_type']] == restriction['property_value']:
                        constrained_nodes.append(node)
        

        if restriction['constraint_object'] == target_object:
            target_constrained_nodes = constrained_nodes

        else:
            
            target_constrained_nodes = []

            for node in constrained_nodes:
                
                G_sub = nx.ego_graph(self.graph, node, radius=1, undirected=True)
                G_sub_nodes = G_sub.nodes()
                targets = [n for n in G_sub._node if G_sub_nodes[n][self.strLabelTypeAttribute] == target_object]
                target_constrained_nodes.append(targets)

            # target_constrained_nodes = list(set(target_constrained_nodes))

        self.search_restriction[restriction_key].update({'constrained_nodes':target_constrained_nodes})
    

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


    def analyze_constraints(self, param_num_precision:int=3):
        
        def flatten(list):
            return [item for sublist in list for item in sublist]
        

        # from building elements find global parameters.
        for ct in self.search_restriction.keys():
            constrained_nodes = self.search_restriction[ct]['constrained_nodes']

            all_cts_map = []

            for nodes in constrained_nodes:
                cts_map = []
                if isinstance(nodes,list):
                    [cts_map.append(self.df_map_from_element.loc[node,'name']) for node in nodes]
                else:
                    cts_map = self.df_map_from_element.loc[nodes,'name']
                all_cts_map.append(cts_map)

            self.search_restriction[ct].update({'constrained_parameters':all_cts_map})

        # from global parameters find values.
        for ct in self.search_restriction.keys():

            # case 1 (per wall): analyze and apply constraints individually to walls.
            if self.search_restriction[ct]['constraint_type'] ==  ('location', 'fix'):

                dictConstraints = dict()
                all_params = self.search_restriction[ct]['constrained_parameters']
                
                # flatten
                if isinstance(all_params[0],list) and len(all_params[0])>1:
                    all_params = flatten(all_params)

                for param in all_params:
                    param_value = self.df_map_from_gp.loc[param,'value']
                    dictConstraints.update({param: param + "=" + str (round(param_value, param_num_precision))})

                self.search_restriction[ct].update({'constraint_application':dictConstraints})
            
            # case 2 (per space): analyze and apply constraints individually to spaces.
            elif self.search_restriction[ct]['constraint_type'] ==  ('boundary','align'):

                dictConstraints = dict()
                all_group_params = self.search_restriction[ct]['constrained_parameters']

                for group_params in all_group_params:

                    group_param_values = [self.df_map_from_gp.loc[param, 'value'] for param in group_params]
                    collision = set([v for v in group_param_values if group_param_values.count(v) > 1])
                    
                    for v in collision:
                        collision_idx = [i for i, x in enumerate(group_param_values) if x == v]
                        
                        if collision_idx:

                            # idx0 in, idx1 not in.    
                            if group_params[collision_idx[0]] in self.all_associated_gps and \
                            group_params[collision_idx[1]] not in self.all_associated_gps:
                                active_id, passive_id = 1, 0
                                
                            # idx0 not in, idx1 in.
                            elif group_params[collision_idx[0]] not in self.all_associated_gps and \
                            group_params[collision_idx[1]] in self.all_associated_gps:
                                active_id, passive_id = 0, 1
                                
                            # both of idx0,idx1 in, no action.
                            elif group_params[collision_idx[0]] in self.all_associated_gps and \
                            group_params[collision_idx[1]] in self.all_associated_gps:
                                continue
                            
                            # none of idx0,idx1 in.
                            else:
                                active_id, passive_id = 1, 0

                            # constraint-wise.
                            active_param, passive_param = group_params[collision_idx[active_id]],group_params[collision_idx[passive_id]]
                            active_passive_value = self.df_map_from_gp.loc[passive_param,'value']
                            dictConstraints.update({active_param: active_param + "=" + str (round(active_passive_value, param_num_precision))})
                            dictConstraints.update({passive_param: passive_param + "=" + active_param })

                            # variation-wise.
                            if passive_param in self.all_associated_gps:
                                self.all_associated_gps.remove(passive_param)
                                self.all_failure_neighbors.remove(self.df_map_from_gp.loc[passive_param,'elementifcguid'])
                
                print ("stp")
                self.search_restriction[ct].update({'constraint_application':dictConstraints})
                
            # case 3 (per space type / group of space): analyze and apply constraints for a group of spaces.
            else:
                continue 

    def ouput_constraints(self, file_json):
        """
        """

        # write all affected information.
        with open(file_json, "w") as outfile:
            json.dump(self.search_restriction, outfile)