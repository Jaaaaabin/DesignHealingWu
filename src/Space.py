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
        
    def set_center(self, iniDesign):

        self.ini_parameters = iniDesign.parameters
        
        if self.guid and self.rule:
            self.ini_results = iniDesign.data[self.guid][self.rule]
        elif self.guid and not self.rule:
            self.ini_results = iniDesign.data[self.guid]
        elif not self.guid and self.rule:
            self.ini_results = iniDesign.results[self.rule]
    
    def form_space(self, newDesigns, columns_Y=['distance','compliance']):
        
        columns_X = list(self.ini_parameters.keys())
        columns_X_Y = columns_X.copy()
        columns_X_Y += columns_Y
        
        data_X_Y =[]
        for i, design in enumerate(newDesigns):
            data_X_Y.append(
                list(flatten(
                [list(design.parameters.values()), # X
                [design.data[self.guid][self.rule][column_y] for column_y in columns_Y], # Y
                ])))

        self.data_X_Y = pd.DataFrame(data_X_Y, columns=columns_X_Y)
        self.data_X = self.data_X_Y[columns_X].to_numpy()
        self.data_Y = self.data_X_Y[columns_Y[1]].to_numpy().astype(int) # use true or false as y.

        # self.