"""
This is the principal module of the healing project.
here you put your main classes.
"""

# import packages
from base_external_packages import *
from base_functions import parameter_grouping, parameter_freezing, parameter_boundarying

# define base classes

class Design():
    """
    Main class: Design class
    
    """

    def __init__(self, number):

        self.number = number
        self.parameters = []
        self.compliance = []
        self.compliance_distance = []
        self.parameter_groups = []
        self.group_frozen = ""
        self.parameter_boundaries = []
    
    # add parameters
    def add_parameters(self, inputdata):
        keys = inputdata.columns.values.tolist()
        keys.sort() # reorder by ascending
        self.parameters = dict.fromkeys(keys)
        for key in keys:
            self.parameters[key] = inputdata[key].iloc[0]
    
    # add checking results, can automatically cover the initial data if there's any.
    def add_codecompliance(self, rules, outputdata):
        self.compliance = dict.fromkeys(rules)
        self.compliance_distance = dict.fromkeys(rules)

        for key in self.compliance.keys():
            idx = outputdata.index[outputdata['failedrules'] == key]
            amount_failed_component = outputdata.loc[idx,'failedcomponentnumbers'].values
            if amount_failed_component == 0:
                self.compliance[key] = True
                self.compliance_distance[key] = outputdata.loc[idx,'averagefailedpercent'].values
            else:
                self.compliance[key] = False
                self.compliance_distance[key] = outputdata.loc[idx, 'averagefailedpercent'].values * \
                    outputdata.loc[idx, 'failedcomponentnumbers'].values
    
    # add parameter groups
    def add_parametergroups(self, strategy):
        self.parameter_groups = parameter_grouping(self.parameters, strategy)
    
    # freeze parameter groups      
    def freeze_parametergroups(self, freezers, group_frozen):
        self.group_frozen = group_frozen
        groups_by_freezers = parameter_freezing(self.parameters, freezers)
        for param in self.parameters:
            self.parameter_groups[param] = self.group_frozen if groups_by_freezers[param] == self.group_frozen else self.parameter_groups[param]
    
    # add parameter boundaries
    def add_parameterboundaries(self, strategy, values):
        parameter_boundaries = parameter_boundarying(self.parameters, strategy, values)
        
        for param in self.parameters:
            if self.parameter_groups[param] == self.group_frozen:
                parameter_boundaries[param] = [-0,0]
            else:
                continue
        self.parameter_boundaries = parameter_boundaries