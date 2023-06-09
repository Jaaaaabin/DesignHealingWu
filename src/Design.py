#
# Design.py
#

# import packages
from base_external_packages import *

class Design():
    """
    Main class: Design clas
    """

    def __init__(self, nr, rules):
        
        self.rules = rules          # rules.
        self.number = nr            # nr.
        self.parameters = dict()    # dict(parameter:value)
        self.results = dict()       # dict(rule: dict(target: ((distance: ), (compliance: )))) 
        self.data = dict()          # dict(target: dict(rule: ((distance: ), (compliance: )))) 
        self.failures = dict()      # dict(target: rule/None).

    def set_parameters(self, newdict):
        
        self.parameters = newdict   # add parameter names and values.
        
    def set_checkresults(self, newdict):

        # archive the 'initial' checking results.
        for rl in self.rules:
            self.results.update({rl: newdict[rl]})
            
        # transpose the first two levels of the initial checking results. and rewrite into self.data.
        tempo_dict = pd.DataFrame(self.results).transpose().to_dict()           
        for i in tempo_dict.keys():
            tempo_failures = []
            for j in tempo_dict[i].keys():
                if not isinstance(tempo_dict[i][j], dict) and np.isnan(tempo_dict[i][j]):
                    tempo_dict[i][j] = {'distance': None, 'compliance': None} 
                elif tempo_dict[i][j]['distance'] < 0 and tempo_dict[i][j]['compliance']== False:
                    tempo_failures.append(j)
            
            # archive the 'initial' failures.
            self.failures.update({i: tempo_failures})        
        
        # archive the 'target-oriented' data
        self.data = tempo_dict
        # # alternative option to write data in DataFrame.
        # self.data = pd.DataFrame.from_dict({(i,j): tempo_dict[i][j] 
        #                     for i in tempo_dict.keys() 
        #                     for j in tempo_dict[i].keys()},
        #                     orient='index')
        # self.data.index = pd.MultiIndex.from_tuples(tempo_dict.index)