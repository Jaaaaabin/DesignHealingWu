#
# Design.py
#

# import packages
from base_external_packages import *

class Space():
    """
    Main class: Space clas
    """

    def __init__(self, ifcguid, rule):
        
        self.guid = ifcguid         # ifcguid
        self.rule = rule            # rule.
        self.parameters = dict()    # dict(parameter:value)
        
    def set_parameters(self, newdict):
        
        self.parameters = newdict   # add parameter names and values.