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

        self.guid = problem[0]                          # ifcguid.
        self.rule = problem[1]                          # rule.
        self.ini_parameters = dict()                    # initial parameters and their values.
        self.ini_results = []                           # initial checking results.
        self.data_X_Y = pd.DataFrame()                  # major database.
        self.valid_idx = dict()                         # indices of the valid designs.
        self.evolve_targets = pd.DataFrame()            # existing targets to approach.
        self.evolve_samples = pd.DataFrame()            # new samples within the evolvement.
        self.sweep_density = 2
        self.sweep_ext_pad = 0.1

        # self.data_X = np.empty(1, dtype=np.float32)
        # self.data_Y = np.empty(1, dtype=np.float32)

    @property
    def strQuant(self):
        return 'distance'

    @property
    def strQual(self):
        return 'compliance'
    

    def __str__(self):
      return f"""
      Solution Space:
          IfcGUID: {self.guid}
          Rule: {self.rule}
          Initial Parameters: {self.ini_parameters}
          Initial Results: {self.ini_results}
      """  
    

    def set_center(self, iniDesign):

        self.ini_parameters = iniDesign.parameters
        
        if not isinstance(self.guid, list) and not isinstance(self.rule, list):

            self.ini_results = iniDesign.data[self.guid][self.rule]
            self.ini_results = [self.guid, self.rule, self.ini_results]

        elif isinstance(self.guid, list) and isinstance(self.rule, list):
            if len(self.guid) == len(self.rule):
                self.ini_results = []
                for gd, rl in zip(self.guid, self.rule):
                    self.ini_results.append([gd, rl, iniDesign.data[gd][rl]])
            else:
                print ('number of IfcGUID and Rule should be equal')
    

    def form_space(self, newDesigns):

        columns_X = list(self.ini_parameters.keys())
        columns_Y_quant, columns_Y_qual = [], []

        for ini_result in self.ini_results:
            columns_Y_quant.append(
                self.strQuant + '-' + ini_result[0] + '-' + ini_result[1])
            columns_Y_qual.append(
                self.strQual + '-' + ini_result[0] + '-' + ini_result[1])

        columns_X_Y = columns_X.copy()
        columns_X_Y += columns_Y_quant
        columns_X_Y += columns_Y_qual
        
        data_X, data_Y_quant, data_Y_qual = [], [], []
        for i, design in enumerate(newDesigns):

            data_X.append(list(design.parameters.values()))
            data_y_quant, data_y_qual = [], []
            for gd, rl in zip(self.guid, self.rule):
                data_y_quant.append(design.data[gd][rl][self.strQuant])
                data_y_qual.append(design.data[gd][rl][self.strQual])
            data_Y_quant.append(data_y_quant)
            data_Y_qual.append(data_y_qual)
        
        data_Y_quant_sum = [sum(data_y_quant) for data_y_quant in data_Y_quant]
        data_Y_qual_sum = [all(data_y_qual) for data_y_qual in data_Y_qual]
        list_X_Y = [[*X, *Y_quant, *Y_qual] for X, Y_quant, Y_qual in zip(data_X, data_Y_quant, data_Y_qual)]
        
        self.data_X_Y = pd.DataFrame(list_X_Y, columns=columns_X_Y)
        self.data_X_Y[self.strQuant] = data_Y_quant_sum
        self.data_X_Y[self.strQual] = data_Y_qual_sum

        self.valid_idx.update({
            self.strQual: self.data_X_Y.index[self.data_X_Y[self.strQual]].tolist()})
        for cl in columns_Y_qual:
            self.valid_idx.update({
                cl: self.data_X_Y.index[self.data_X_Y[cl]].tolist()})


    def enrich_space(self, divide_label_x=[], divide_label_y=[]):

        if not divide_label_x and not divide_label_y:

            self.data_X_df = self.data_X_Y[list(self.ini_parameters.keys())]
            self.data_X_np = self.data_X_Y[list(self.ini_parameters.keys())].to_numpy()

            self.data_Y_dict = dict()
            self.data_Y_dict.update({
                self.strQual: self.data_X_Y[self.strQual].to_numpy().astype(int)})
            
            tempo_cls = [cl for cl in self.data_X_Y.columns.values.tolist()if self.strQual in cl]
            for cl in tempo_cls:
                self.data_Y_dict.update({
                    cl: self.data_X_Y[cl].to_numpy().astype(int)})


    def _sweeping_from_initotargets(self, sweep_density, ext_pad):
        
        # prepare the data for sweeping.
        v_init = np.array(list(self.ini_parameters.values()))
        v_targets = self.evolve_targets.values
        

        def sweeping(init_v, target_vs, sweep_density, ext_pad):
            """
            :init_v:        [0,                 n_parameter]
            :target_vs:     [number_of_targets, n_parameter], and will return
            samples         [number_of_targets * sweep_density, n_parameter]
            """
            def random_even_samp_vm2vn(vm, vn, amount, ext_pad): # np.random.uniform
                random.seed(2023)
                random_factors = np.random.uniform(
                    low=0-ext_pad, high=1+ext_pad, size=amount) 
                samples =  np.array([vm + random_factor*(vn-vm) for random_factor in random_factors])
                return samples

            all_samples = np.empty(
                shape=[target_vs.shape[0]*sweep_density,target_vs.shape[1]])
            
            for ii in range(target_vs.shape[0]):
                samples = np.empty(shape=[sweep_density,target_vs.shape[1]])
                for jj in range (target_vs.shape[1]):
                    tp_samples = random_even_samp_vm2vn(init_v[jj], target_vs[ii,jj], amount=sweep_density, ext_pad=ext_pad)
                    samples[:,jj] = tp_samples
                all_samples[ii*sweep_density:(ii+1)*sweep_density,:] = samples

            return all_samples

        # Sweeping Part1: sweeping from the initial design to target designs
        values_fill_gap = sweeping(v_init, v_targets, sweep_density=sweep_density, ext_pad=ext_pad*2)
        
        # Sweeping Part2: sweeping between target designs
        values_fill_region = np.array([]).reshape(0,v_init.shape[0])
        for i in range(v_targets.shape[0]-1):
            v_targets_init = v_targets[i]
            v_targets_targ = v_targets[i+1:]
            values_tempo = sweeping(v_targets_init, v_targets_targ, sweep_density=sweep_density, ext_pad=ext_pad)
            values_fill_region = np.concatenate([values_fill_region, values_tempo], axis=0)
        
        self.evolve_samples = pd.DataFrame(np.concatenate([values_fill_region, values_fill_gap], axis=0), columns=list(self.ini_parameters.keys()))


    def _config_sweeping(self, set_sweep_density, set_sweep_ext_pad):
        
        if set_sweep_density:
            self.sweep_density = set_sweep_density
        if set_sweep_ext_pad:
            self.sweep_ext_pad = set_sweep_ext_pad
    
    def evolve_space(self, evolve_aspects=[], vary_file=[]):
        
        # filter the evolve excitements
        if not evolve_aspects:
            evolve_aspects = list(self.valid_idx.keys())

        targets_to_approach = []
        for tgt in evolve_aspects:
            idx = self.valid_idx[tgt]
            evolve_dest = self.data_X_Y.iloc[idx] # use  self.data_X_Y to consider the distance as sweeping input later.
            targets_to_approach.append(evolve_dest)
        
        # get all evolve_targets to self.evolve_targets
        n = 0

        if len(evolve_aspects) == 1:
            all_evolve_targets = targets_to_approach[0]
        else:
            while n < len(targets_to_approach)-1:
                all_evolve_targets = pd.concat([targets_to_approach[n], targets_to_approach[n+1]], axis=0)
                targets_to_approach[n+1] = all_evolve_targets
                n+=1
            all_evolve_targets = all_evolve_targets.drop_duplicates()
        self.evolve_targets = all_evolve_targets[list(self.ini_parameters.keys())]
            
        # sweeping within the SolutionSpace.
        self._sweeping_from_initotargets(sweep_density = self.sweep_density, ext_pad = self.sweep_ext_pad)
        self.evolve_samples.T.to_csv(vary_file, header=False)

