#
# __main__.py
#
from const_project import EXECUTION_NR
from topoCollection import topoCollect
from graphCreation import graphCreate
from graphEnricher import graphEnrich
from prepareVariation import prepareVariants
from testSensitivity import testSensi_sobol
# from prepareSolutionSpace import buildSpace

if __name__ == "__main__":
    
    # topoCollect()
    # graphCreate()
    # graphEnrich(plot_graph=False)

    print (EXECUTION_NR)
    prepareVariants(sa_type = 'morris', set_dup_rvt=False)
    # testSensi_sobol(build_design=False, calc_index=True, plot_index=True)




# to do
# 1. develop an automated Error/Warning handler.
# 1.1 too hard with Python for an automated handler. 
# 1.2 continue to test with test-handler-paramed.rvt and test-handler.dyn
# -> first conflict nr .38.
# 1.2 [first test] try with activating the workset during the generation
# it's slightly faster for generation.
# it's slower for checking.
# 1.3 [second test] remove windows/doors before the duplication. OK.
# desgin overwrite completed.
# how to automatically save??

# 2. add more advanced locational parameters on doors and windows.
# 2.1 ground floor -> [already succeed.]
# 2.2 upper floors -> [not successful yet.]
# solution 1 - > refer to different grids per floor.
# solution 2 - > find exact aligned dimension command.

# the next step.
# 3. focus on error on one floor maybe.. if not the paper doesnt move on