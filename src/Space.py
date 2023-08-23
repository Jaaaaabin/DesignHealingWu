#
# Space.py
#

# import packages
from base_external_packages import *

def flatten(L):
    for item in L:
        try:
            yield from flatten(item)
        except TypeError:
            yield item

class SolutionSpace():
    """
    Main class: Space class
    """


    def __init__(self, problem):
        
        problem = np.array(problem).T.tolist() if len(problem) > 1 else problem

        self.guids = []                                 # filtered guids. (including spaces and all other building elements)
        self.rule = problem                             # rule.
        self.ini_parameters = dict()                    # initial parameters and their values.
        self.ini_results = []                           # initial checking results.
        self.data_X_Y = pd.DataFrame()                  # major database.
        self.valid_idx = dict()                         # indices of the valid designs.
        self.sensitivity = dict()
        self.sensitivity_sign = dict()
        self.samples_by_skewnormal = pd.DataFrame()
        self.samples_by_lhs = pd.DataFrame()
        self.valid_set_x = dict()

    def ___setcenter__(self, iniDesign):

        self.ini_parameters = iniDesign.parameters

    def ___setguids__(self, guidlist):
        self.guids = guidlist

    @property
    def strQuant(self):
        return 'distance'

    @property
    def strQual(self):
        return 'compliance'
    

    def __str__(self):
      return f"""
      Solution Space:
          IfcGUID: {self.guids}
          Rule: {self.rule}
          Initial Parameters: {self.ini_parameters}
          Initial Results: {self.ini_results}
      """  
    
    # def __calc__(self, input, df):

    #     if isinstance(input, list):
    #         a = input[0]
    #         b = input[1]
        
    def _calculate_cosine_distance(self, a, b):
        cosine_distance = float(spatial.distance.cosine(list(a), list(b)))
        return cosine_distance


    def _skewnormal_sampling(self, param_name, delta_loc, alpha, random_seed, num_samples=100, plot_sampling=False, plot_dirs=[]):
        
        np.random.seed(random_seed)
        loc = self.ini_parameters[param_name]
        samples = skewnorm.rvs(alpha, loc=loc, scale=delta_loc/3, size=num_samples, random_state=random_seed)

        if plot_sampling:

            # # Plot a histogram of the generated samples
            fig = plt.figure(figsize=(10,5))  # unit of inch
            ax = plt.axes((0.15, 0.10, 0.80, 0.80))  # in range (0,1)

            plt.hist(samples, bins=25, density=True, color='g', label='Sampled Data')
            x = np.linspace(min(samples), max(samples), 100)
            pdf = skewnorm.pdf(x, alpha, loc=loc, scale=delta_loc/3)
            plt.plot(x, pdf, 'r', label='Skew Normal PDF')
            plt.axvline(x=loc-delta_loc,c='black')
            plt.axvline(x=loc+delta_loc,c='black')
            plt.axvline(x=loc,c='blue')
            plt.xlabel(str(param_name) + 'values')
            plt.ylabel('Probability Density')
            plt.title('Sampling from Skew Normal Distribution')
            plt.legend()
            plt.savefig(plot_dirs + '/sampling_skewnormal_delta_{}_alpha_{}_param_{}.png'.format(delta_loc, alpha, param_name), dpi=200)
    
        return samples


    def enrich_sensitivity(self, indicesSA=dict(), key_sign_rule=[], val_tol = 1e-3):

        # self.sensitivity
        for rl in self.rule:
            
            sensi_per_rule = dict()
            sensi_per_rule.update(
                {param: [param_mu, sigma] for (param, param_mu, sigma, param_mu_star) in \
                 zip(indicesSA[rl]['names'],indicesSA[rl]['mu'],indicesSA[rl]['sigma'],indicesSA[rl]['mu_star']) if param_mu_star > val_tol})
            
            self.sensitivity.update({rl: sensi_per_rule})
        
        # self.sensitivity_sign
        if key_sign_rule in self.rule:

            for k,v in self.sensitivity[key_sign_rule].items():
                if (v[0]+v[1])*(v[0]-v[1])>=0 or abs(abs(v[0])-abs(v[1])) <= val_tol:
                    sensi_sign = v[0]/abs(v[0])
                else:
                    sensi_sign = 0
                self.sensitivity_sign.update({k: sensi_sign})


    def explore_space_by_skewnormal(self, alpha_ratio=3, explore_range=0.3, num_samples=200, random_seed=1996, plot_dirs=[]):

        k_list , v_list= [],[]

        for k,v in self.sensitivity_sign.items():

            k_list.append(k)
            v_alpha = v*alpha_ratio

            np.random.seed(random_seed)
            new_v = self._skewnormal_sampling(
                param_name=k,
                delta_loc=explore_range,
                alpha=v_alpha,
                random_seed=random_seed,
                num_samples=num_samples,
                plot_sampling=True,
                plot_dirs=plot_dirs)
            
            np.random.shuffle(new_v)
            
            random_seed+=1
            v_list.append(new_v)

        self.samples_by_skewnormal = pd.DataFrame(np.array(v_list).T,columns=k_list).T


    def explore_space_by_lhs(self, explore_range, lhs_optimization, num_samples, random_seed=1996, plot_dirs=[]):

        np.random.seed(random_seed)

        explore_ranges_l, explore_ranges_u = [], []

        # if the exploration range is from a determined list.
        if isinstance(explore_range, str):

            with open(explore_range, 'rb') as handle:
                explore_range_dict = pickle.load(handle)
            for k in self.sensitivity_sign.keys():
                explore_ranges_l.append(explore_range_dict[k][0])
                explore_ranges_u.append(explore_range_dict[k][1])

        # if the exploration range is an assumed fixed value.
        else:

            for k,v in self.sensitivity_sign.items():
                if v==0:
                    explore_ranges_l.append(self.ini_parameters[k]-explore_range)
                    explore_ranges_u.append(self.ini_parameters[k]+explore_range)
                elif v>0:
                    explore_ranges_l.append(self.ini_parameters[k])
                    explore_ranges_u.append(self.ini_parameters[k]+explore_range)
                elif v<0:
                    explore_ranges_l.append(self.ini_parameters[k]-explore_range)
                    explore_ranges_u.append(self.ini_parameters[k])
        
        lh = qmc.LatinHypercube(d=len(explore_ranges_l), scramble=False, optimization=lhs_optimization, seed=random_seed)
        lhs_samples = lh.random(n=num_samples)
        samples = qmc.scale(lhs_samples, explore_ranges_l, explore_ranges_u)
        
        self.samples_by_lhs = pd.DataFrame(samples, columns=self.sensitivity_sign.keys()).T


    def form_space(self, iniDesign, newDesigns, diff_dims= True):

        columns_Y_quant, columns_Y_qual = [], []
        data_Y_quant, data_Y_qual = [], []
        data_X = []

        for i, design in enumerate(newDesigns):

            # x values

            # if the solution space covers different multi dimensional options
            if diff_dims:
            
                # if the dimensions are the same.
                if design.parameters.keys() == iniDesign.parameters.keys():

                    design_x = list(design.parameters.values())
                
                # if the dimensions are reduced.
                elif (all(x in iniDesign.parameters.keys() for x in design.parameters.keys())):

                    add_keys = list(set(iniDesign.parameters.keys()) - set(design.parameters.keys())) # add the unsensitive keys.

                    design.parameters.update({key:iniDesign.parameters[key] for key in add_keys}) # use the initial designs.

                    design.parameters = {k: design.parameters[k] for k in iniDesign.parameters.keys()} # reorder.

                    design_x = list(design.parameters.values())

            else:

                design_x = list(design.parameters.values())

            data_X.append(design_x)
            
            # y values
            
            data_y_quant, data_y_qual = [], []
            design_failure_dict = dict((guid, design.failures[guid]) for guid in design.failures.keys() if guid in self.guids)
            design_failure_clean = [sublist for sublist in list(design_failure_dict.values()) if any(sublist)]
            design_failure_clean = [item for sublist in design_failure_clean for item in sublist]

            for rl in self.rule:
                if rl in design.rules:
                    if rl not in design_failure_clean:
                        data_y_qual.append(True)
                    else:
                        data_y_qual.append(False)
                else:
                    continue
            data_Y_qual.append(data_y_qual)
        
        # columns_Y_quant += [self.strQuant+'-'+rl for rl in design.rules]
        columns_Y_qual += [self.strQual+'-'+rl for rl in design.rules]
        columns_Y_qual += [self.strQual]

        columns_X_Y = list(self.ini_parameters.keys()).copy()
        columns_X_Y += columns_Y_qual
        
        data_y_qual_sum = [all(data_y_qual) for data_y_qual in data_Y_qual]
        data_Y_qual = [i+[j] for i,j in zip(data_Y_qual, data_y_qual_sum)]
        list_X_Y = [[*X, *Y_qual] for X, Y_qual in zip(data_X, data_Y_qual)]

        self.data_X_Y = pd.DataFrame(list_X_Y, columns=columns_X_Y)
        for cl in columns_Y_qual:
            self.valid_idx.update({
                cl: self.data_X_Y.index[self.data_X_Y[cl]].tolist()})


    def __buildxy__(self, dir = [], build_valid_subset=False):

        # to rearrange---
        self.data_X_df = self.data_X_Y[list(self.ini_parameters.keys())]
        # self.data_X_np = self.data_X_Y[list(self.ini_parameters.keys())].to_numpy()

        self.data_Y_dict = dict()
        self.data_Y_dict.update({
            self.strQual: self.data_X_Y[self.strQual].to_numpy().astype(int)})
        
        tempo_cls = [cl for cl in self.data_X_Y.columns.values.tolist()if self.strQual in cl]
        for cl in tempo_cls:
            self.data_Y_dict.update({cl: self.data_X_Y[cl].to_numpy().astype(int)})
        # to rearrange---

        if build_valid_subset:

            for cl in tempo_cls:
                
                valid_subset = self.data_X_df.iloc[self.valid_idx[cl]]
                self.valid_set_x.update({cl: valid_subset})

                # summarize the validity ranges for the overall compliance.
                if cl == 'compliance':
                    
                    # Drop all columns that have constant value by the parameter names.
                    valid_subset_tempo = copy.deepcopy(valid_subset)
                    param_to_drop = valid_subset_tempo.columns[valid_subset_tempo.nunique() <= 1].tolist()
                    valid_subset_tempo = valid_subset_tempo.drop(param_to_drop, axis=1)
                    validity_dict = dict()

                    for param in valid_subset_tempo.columns.tolist():
                        validity_dict.update({param:[valid_subset[param].min(), valid_subset[param].max()]})

                    # save the validity ranges to an external dictionary.
                    filename_dict = dir + r'\compliance_valid_ranges.pickle'
                    with open(filename_dict, 'wb') as handle:
                        pickle.dump(validity_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

                # output along with the initial design values.
                valid_subset.loc[-1] = self.ini_parameters.values() # adding the initial desing.
                valid_subset.index = valid_subset.index + 1  # shifting index
                valid_subset.sort_index(inplace=True)
                
                valid_subset.to_csv(dir + r'\valid_subset_{}.csv'.format(cl), header=True)


    def __addini__(self):

        # add the initial design. (x and y)
        ini_X = [self.ini_parameters[variable] for variable in self.ini_parameters.keys()]  # adding a row
        ini_Y = ['Initial Design']*len(self.data_Y_dict.keys())
        ini_X_Y = ini_X + ini_Y

        # add to data_X_df
        self.data_X_df.loc[-1] = ini_X
        self.data_X_df.index = self.data_X_df.index + 1
        self.data_X_df = self.data_X_df.sort_index()

        # add to data_X_Y
        self.data_X_Y.loc[-1] = ini_X_Y 
        self.data_X_Y.index = self.data_X_Y.index + 1
        self.data_X_Y = self.data_X_Y.sort_index()

        # add to data_Y_dict
        for key in self.data_Y_dict.keys():
            lst =  list(self.data_Y_dict[key])
            lst.insert(0,0)
            self.data_Y_dict.update({
                key: lst
            })

        # need to also do for self.valid_idx, and valid_set_x.

    def remove_constant(self):
        
        constantVariable = self.data_X_df.columns[self.data_X_df.nunique() <= 1].tolist()
        # nonconstantVariable = [v for v in self.data_X_df.columns.tolist() if v not in constantVariable]

        for v in constantVariable:
            self.ini_parameters.pop(v)
            self.data_X_df = self.data_X_df.drop(columns = v) 
            self.data_X_Y = self.data_X_Y.drop(columns = v)

    def transfer_space_new(self):

        all_data = pd.DataFrame()
        
        all_data['sn1'] = self.data_X_df['U1_OK_d_wl_sn21']
        all_data['sn2'] = self.data_X_df['U1_OK_d_wl_sn10'] - self.data_X_df['U1_OK_d_wl_sn21']
        all_data['sn3'] = self.data_X_df['U1_OK_d_wl_sn26'] - self.data_X_df['U1_OK_d_wl_sn10']
        all_data['sn4'] = 11.095 - self.data_X_df['U1_OK_d_wl_sn26'] 

        all_data['ew1'] = self.data_X_df['U1_OK_d_wl_ew6']
        all_data['ew2'] = self.data_X_df['U1_OK_d_wl_ew35'] - self.data_X_df['U1_OK_d_wl_ew6']

        # all_data['sn-all'] =  all_data['sn1'] + all_data['sn2'] + all_data['sn3']
        # all_data['ew-all'] =  all_data['ew1'] + all_data['ew2']
        
        # all_data['all'] =  all_data['ew1'] * all_data['sn-all']
        
        self.data_X_df = all_data
    
        for cl in list(all_data.columns):
            self.data_X_Y[cl] = all_data[cl] 
        
    
    def transfer_space(self, inter_level=0):

        all_data = copy.deepcopy(self.data_X_df)
        level = 0
        l1 = []

        while level < inter_level:
            
            l2 = list(self.data_X_df.columns) # l2 always from the initial design.
            if not l1: # l1 is updated round by round
                l1=l2
            inter_pairs = [list(v) for v in product(l1, l2)]

            for (c1,c2) in inter_pairs:
                
                # full combination.
                # if ('sn' in c1 and 'sn' in c2) or ('ew' in c1 and 'ew' in c2):
                #     all_data[c1+'-'+c2] = all_data[c1] - self.data_X_df[c2]
                # elif ('sn' in c1 and 'ew' in c2) or ('ew' in c1 and 'sn' in c2):
                    # all_data[c1+'*'+c2] = all_data[c1] * self.data_X_df[c2]
                
                # full purpose-based.
                if ('sn' in c1 and 'sn' in c2) or ('ew' in c1 and 'ew' in c2):
                    if self.ini_parameters[c1]>=self.ini_parameters[c2]:
                        all_data[c1+'-'+c2] = all_data[c1] - self.data_X_df[c2]
                    else:
                        all_data[c2+'-'+c1] = all_data[c2] - self.data_X_df[c1]

            # clean the columns with constant values
            for v in all_data.columns[all_data.nunique() <= 1].tolist():
                all_data = all_data.drop(columns = v)

            # clean columns containing same values with another column
            all_data = all_data.T.drop_duplicates().T
            l1 = list(all_data.columns)
            level += 1
            
        self.data_X_df = all_data
    
        for cl in list(all_data.columns):
            self.data_X_Y[cl] = all_data[cl] 
        
    
#---------------------------------------------sweeping

    # def _sweeping_from_initotargets(self, sweep_density, ext_pad):
        
    #     # prepare the data for sweeping.
    #     v_init = np.array(list(self.ini_parameters.values()))
    #     v_targets = self.evolve_targets.values
        

    #     def sweeping(init_v, target_vs, sweep_density, ext_pad):
    #         """
    #         :init_v:        [0,                 n_parameter]
    #         :target_vs:     [number_of_targets, n_parameter], and will return
    #         samples         [number_of_targets * sweep_density, n_parameter]
    #         """
    #         def random_even_samp_vm2vn(vm, vn, amount, ext_pad): # np.random.uniform
    #             random.seed(1996)
    #             random_factors = np.random.uniform(
    #                 low=0-ext_pad, high=1+ext_pad, size=amount) 
    #             samples =  np.array([vm + random_factor*(vn-vm) for random_factor in random_factors])
    #             return samples

    #         all_samples = np.empty(
    #             shape=[target_vs.shape[0]*sweep_density,target_vs.shape[1]])
            
    #         for ii in range(target_vs.shape[0]):
    #             samples = np.empty(shape=[sweep_density,target_vs.shape[1]])
    #             for jj in range (target_vs.shape[1]):
    #                 tp_samples = random_even_samp_vm2vn(init_v[jj], target_vs[ii,jj], amount=sweep_density, ext_pad=ext_pad)
    #                 samples[:,jj] = tp_samples
    #             all_samples[ii*sweep_density:(ii+1)*sweep_density,:] = samples

    #         return all_samples

    #     # Sweeping Part1: sweeping from the initial design to target designs
    #     values_fill_gap = sweeping(v_init, v_targets, sweep_density=sweep_density, ext_pad=ext_pad*2)
        
    #     # Sweeping Part2: sweeping between target designs
    #     values_fill_region = np.array([]).reshape(0,v_init.shape[0])
    #     for i in range(v_targets.shape[0]-1):
    #         v_targets_init = v_targets[i]
    #         v_targets_targ = v_targets[i+1:]
    #         values_tempo = sweeping(v_targets_init, v_targets_targ, sweep_density=sweep_density, ext_pad=ext_pad)
    #         values_fill_region = np.concatenate([values_fill_region, values_tempo], axis=0)
        
    #     self.evolve_samples = pd.DataFrame(np.concatenate([values_fill_region, values_fill_gap], axis=0), columns=list(self.ini_parameters.keys()))
    

    # def _config_sweeping(self, set_sweep_density, set_sweep_ext_pad):
        
    #     if set_sweep_density:
    #         self.sweep_density = set_sweep_density
    #     if set_sweep_ext_pad:
    #         self.sweep_ext_pad = set_sweep_ext_pad
    
    # def evolve_space(self, evolve_aspects=[], vary_file=[]):
        
    #     # filter the evolve excitements
    #     if not evolve_aspects:
    #         evolve_aspects = list(self.valid_idx.keys())

    #     targets_to_approach = []
    #     for tgt in evolve_aspects:
    #         idx = self.valid_idx[tgt]
    #         evolve_dest = self.data_X_Y.iloc[idx] # use  self.data_X_Y to consider the distance as sweeping input later.
    #         targets_to_approach.append(evolve_dest)
        
    #     # get all evolve_targets to self.evolve_targets
    #     n = 0

    #     if len(evolve_aspects) == 1:
    #         all_evolve_targets = targets_to_approach[0]
    #     else:
    #         while n < len(targets_to_approach)-1:
    #             all_evolve_targets = pd.concat([targets_to_approach[n], targets_to_approach[n+1]], axis=0)
    #             targets_to_approach[n+1] = all_evolve_targets
    #             n+=1
    #         all_evolve_targets = all_evolve_targets.drop_duplicates()
    #     self.evolve_targets = all_evolve_targets[list(self.ini_parameters.keys())]
            
    #     # sweeping within the SolutionSpace.
    #     self._sweeping_from_initotargets(sweep_density = self.sweep_density, ext_pad = self.sweep_ext_pad)
    #     self.evolve_samples.T.to_csv(vary_file, header=False)