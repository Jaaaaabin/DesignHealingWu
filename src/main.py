#
# __main__.py
#

from const_project import EXECUTION_NR

from topoCollection import topoCollect
from graphCreation import graphCreate
from graphEnricher import graphEnrich
from prepareVariation import prepareSAVariants
from testSensitivity import buildDesigns, testSensi_sobol, testSensi_morris
from formSpace import formSolutionSpace

if __name__ == "__main__":

    #=================================#
    #                ini              #
    #=================================#
    # topoCollect()
    # graphCreate()
    # graphEnrich(plot_graph=False)
    
    #=================================#
    #                 sa              #
    #=================================#
    # EXECUTION_NR = 14, 54
    # prepareSAVariants(sa_type = 'morris', set_dup_rvt = True)
    # testSensi_morris(build_design=True, calc_index=True, plot_index=True)

    # EXECUTION_NR = 19, 59
    # prepareSAVariants(sa_type = 'sobol', set_dup_rvt = True)
    # testSensi_sobol(build_design=True, calc_index=True, plot_index=True)

    #=================================#
    #                 ss              #
    #=================================#
    # EXECUTION_NR = 114, 154, 119, 159
    formSolutionSpace(['\sa-14-0.3'], set_evolve_space = True, set_dup_rvt = True)
    # formSolutionSpace(['\sa-14-0.3', '\ss-114-0.3'])






# https://gsa-module.readthedocs.io/en/stable/implementation/morris_screening_method.html
# for levels in Morris: From Screening to Quantitative Sensitivity Analysis. A Unified Approach
# An effective screening design for sensitivity analysis of large models
# Choosing the appropriate sensitivity analysis method for building energy model-based investigations
# The number of grid levels.

# to do
# 1. develop an automated Error/Warning handler.
# 1.1 too hard with Python for an automated handler. 
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