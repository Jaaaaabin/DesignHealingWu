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
        self.all_failure_elements = []
        self.all_associated_gps = []

        # self.label_search_belonging = ['wall','separationline'] # much faster.
        self.label_search_belonging = ['wall']          # search step - beloning
        self.label_search_propagating = ['space']       # search step - propagating
        self.label_search_targeting = ['parameter']     # search step - targeting

        self.constrained_neighbor_elements = [] # for case 1, 2, 3.(to improve together)
        self.constrained_neighbor_gps = [] # for case 1, 2, 3.(to improve together)
        self.constraints = {}

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

# constants-------------------------------------------------------------------------------------------
    @property
    def strLabelGp(self):
        return 'parameter_associated'

    @property
    def strLabelFailureElement(self):
        return 'failure_element'

    @property
    def strLabelFailureLocation(self):
        return 'failure'
    
    @property
    def strLabelTypeAttribute(self):
        return 'classification'

    @property
    def lengthLimitation(self):
        return 12
    
# searching-------------------------------------------------------------------------------------------
    # search step 1 - beloning
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
    

    # search step 2 - restricting by connection amount.
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


    # search step 2 - restricting by length amount.
    def search_restricting_bylength(self, belongs):

        new_belongs =[]

        for node in belongs:
            if self.graph.nodes[node]['length'] > self.lengthLimitation:
                continue
            else:
                new_belongs.append(node)

        return new_belongs
    

    # search step 2 - restricting by properties.
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
        

    # search step 3 - propagating.
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

            # 1 search space belongings.
            belongs = self.search_belonging(starts)
            # tempo_space = starts

            # 2 restrict the space belongings.
            belongs = self.search_restricting_bylength(belongs) # via length value.

            # belongs = self.search_restricting_byconnection(belongs) # via connection amount.
            
            belongs = self.search_restricting_property(belongs) # via object property.

            # 3 propagate to new spaces and refresh the previous belongings.
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
            self.all_failure_elements += nbrs
            self.all_associated_gps += gp_nbrs
            
        self.all_failure_elements = list(set(self.all_failure_elements))
        self.all_associated_gps = list(set(self.all_associated_gps))

        # additional step: only display walls but not seplines.
        self.all_failure_elements = [item for item in self.all_failure_elements \
        if self.graph.nodes[item][self.strLabelTypeAttribute]=='wall']


# constraints-------------------------------------------------------------------------------------------
    def analyze_constraints(self, param_num_precision:int=3):
        
        def flatten(list):
            return [item for sublist in list for item in sublist]
        
        def get_close_value_idx(mylist, threshold=0.05):

            idx_mylist = range(len(mylist))
            combs = set(combinations(idx_mylist, 2))

            return set(filter(lambda x: abs(mylist[x[0]] - mylist[x[1]]) < threshold, combs))

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
            
            dictConstraints = dict()
            all_params = self.search_restriction[ct]['constrained_parameters']

            # case 1 (per wall): analyze and apply constraints individually to walls.
            if self.search_restriction[ct]['constraint_type'] ==  ('location', 'fix'):
            
                # flatten
                if isinstance(all_params[0],list) and len(all_params[0])>1:
                    all_params = flatten(all_params)

                for param in all_params:
                    param_value = self.df_map_from_gp.loc[param,'value']
                    dictConstraints.update({param: param + "=" + str (round(param_value, param_num_precision))})
            
            # case 2 (per space): analyze and apply constraints individually to spaces.
            elif self.search_restriction[ct]['constraint_type'] ==  ('boundary','align'):

                for group_params in all_params:

                    group_param_values = [self.df_map_from_gp.loc[param, 'value'] for param in group_params]

                    all_collision = get_close_value_idx(group_param_values)
                    # collision = set([v for v in group_param_values if group_param_values.count(v) > 1])
                    
                    for collision_idx in all_collision:

                        if collision_idx:

                            # idx0 in, idx1 not in the planned variation.
                            if group_params[collision_idx[0]] in self.all_associated_gps and \
                            group_params[collision_idx[1]] not in self.all_associated_gps:
                                active_id, passive_id = 1, 0
                                
                            # idx0 not in, idx1 in the planned variation.
                            elif group_params[collision_idx[0]] not in self.all_associated_gps and \
                            group_params[collision_idx[1]] in self.all_associated_gps:
                                active_id, passive_id = 0, 1
                                
                            # both of idx0,idx1 in the planned variation., no action.
                            elif group_params[collision_idx[0]] in self.all_associated_gps and \
                            group_params[collision_idx[1]] in self.all_associated_gps:
                                continue
                            
                            # none of idx0,idx1 in the planned variation.
                            else:
                                active_id, passive_id = 1, 0

                            # constraint-wise.
                            active_param, passive_param = group_params[collision_idx[active_id]],group_params[collision_idx[passive_id]]
                            # active_passive_value = self.df_map_from_gp.loc[passive_param,'value']
                            # dictConstraints.update({active_param: active_param + "=" + str (round(active_passive_value, param_num_precision))})
                            dictConstraints.update({passive_param: passive_param + "=" + active_param })

                            # # variation-wise.(to improve together)
                            # if passive_param in self.all_associated_gps:
                            #     self.all_associated_gps.remove(passive_param)
                            #     self.all_failure_elements.remove(self.df_map_from_gp.loc[passive_param,'elementifcguid'])
                        
            # case 3 (per space type / group of space): analyze and apply constraints for a group of spaces.
            elif self.search_restriction[ct]['constraint_type'] ==  ('totalarea', 'fix'):
                
                if isinstance(all_params[0],list) and len(all_params[0])>1:
                    all_params = flatten(all_params)

                floor_keys = ['U1', 'E0', 'E1']
                dirction_keys = ['sn', 'ew']

                for flr_key in floor_keys:
                    for drt_key in dirction_keys:
                        flr_drt_params = [param for param in all_params if flr_key in param and drt_key in param]
                        flr_drt_params = list(set(flr_drt_params))
                        flr_drt_params_v = [self.df_map_from_gp.loc[param, 'value'] for param in flr_drt_params]
                        
                        # tempo function
                        # filter the horizontal walls.

                        if(set(flr_drt_params).issubset(set(self.all_associated_gps))): 
                            
                            sorted_flr_drt_params = [x for _, x in sorted(zip(flr_drt_params_v, flr_drt_params))] # sort.

                            # constraint-wise.
                            active_param, passive_param = sorted_flr_drt_params[-1], sorted_flr_drt_params[0]
                            active_value, passive_value = self.df_map_from_gp.loc[active_param,'value'], self.df_map_from_gp.loc[passive_param,'value']
                            dictConstraints.update({passive_param: passive_param + "=" + active_param + "-" + \
                                                     str (round(active_value-passive_value, param_num_precision)) })
                            
                            # # variation-wise..(to improve together)
                            # if passive_param in self.all_associated_gps:
                            #     self.all_associated_gps.remove(passive_param)
                            #     self.all_failure_elements.remove(self.df_map_from_gp.loc[passive_param,'elementifcguid'])

                        else:
                            continue
                
            self.search_restriction[ct].update({'constraint_application':dictConstraints})


    def create_constraints(self, file_json):
        """
        """

        # write all affected information.

        with open(file_json, "w") as outfile:
            json.dump(self.search_restriction, outfile)


    def apply_constraints(self, file_json):
        """
        """

        # read the constaints.
        with open(file_json) as json_file:
            dictConstraints = json.load(json_file)
        
        # merge together.
        dictConstraints_gps = {}
        [dictConstraints_gps.update(dictConstraints[ct]['constraint_application']) for ct in dictConstraints.keys()]
        
        # collect constrained_gps and constrained_elements around the neighborhood.
        self.constrained_neighbor_gps = [gp for gp in self.all_associated_gps if gp in dictConstraints_gps.keys()]
        self.constrained_neighbor_elements = [self.df_map_from_gp.loc[gp,'elementifcguid'] for gp in self.constrained_neighbor_gps]

        # remove constrained_gps            from    self.all_associated_gps
        # remove constrained_elements       from    self.all_failure_elements
        self.all_associated_gps = [v for v in self.all_associated_gps if v not in self.constrained_neighbor_gps]
        self.all_failure_elements = [v for v in self.all_failure_elements if v not in self.constrained_neighbor_elements]
        
        dictApplicationConstraints = {}
        [dictApplicationConstraints.update({gp: dictConstraints_gps[gp]}) for gp in self.constrained_neighbor_gps]

        print('stp')
        return dictApplicationConstraints


# updating-------------------------------------------------------------------------------------------
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
        self.update_graph_nodes(self.all_failure_elements, self.strLabelFailureElement)# add failure neighbors.
        self.update_graph_nodes(self.all_failure_locations, self.strLabelFailureLocation)# add failure locations.
