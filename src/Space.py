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

        self.guid = problem[0]          # ifcguid.
        self.rule = problem[1]          # rule.
        self.ini_parameters = dict()
        self.ini_results = []
        self.data_X_Y = pd.DataFrame()
        self.data_X = np.empty(1, dtype=np.float32)
        self.data_Y = np.empty(1, dtype=np.float32)


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
          data X: {self.data_X}
          data Y: {self.data_Y}
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


    def subdivide_space(self, divide_label_x=[], divide_label_y=[]):

        if not divide_label_x and not divide_label_y:
            self.data_X = self.data_X_Y[list(self.ini_parameters.keys())].to_numpy()
            self.data_Y = self.data_X_Y[self.strQual].to_numpy().astype(int) # use true or false as y.