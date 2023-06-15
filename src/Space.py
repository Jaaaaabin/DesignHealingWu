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

    def __init__(self, ifcguid=[], rule=[]):
        
        self.guid = ifcguid         # ifcguid.
        self.rule = rule            # rule.
        
    def set_space_center(self, iniDesign):

        self.ini_parameters = iniDesign.parameters
        
        if self.guid and self.rule:
            self.ini_results = iniDesign.data[self.guid][self.rule]
        elif self.guid and not self.rule:
            self.ini_results = iniDesign.data[self.guid]
        elif not self.guid and self.rule:
            self.ini_results = iniDesign.results[self.rule]
    
    def form_space(self, newDesigns, label_y='distance'):
        
        data_columns = list(self.ini_parameters.keys())
        data_columns.append(label_y)
        
        data_X_y =[]
        for i, design in enumerate(newDesigns):
            data_X_y.append(
                list(flatten(
                [list(design.parameters.values()), # X
                design.data[self.guid][self.rule][label_y] # y
                ])))

        self.data = pd.DataFrame(data_X_y, columns=data_columns)


        # self.