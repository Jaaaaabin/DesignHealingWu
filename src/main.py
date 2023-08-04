#
# __main__.py
#

from const_project import EXECUTION_NR

from topoCollection import topoCollect
from graphCreation import graphCreate
from graphEnricher import graphEnrich
from prepareVariation import prepareSAVariants
from testSensitivity import testSensi_morris_weighted, testSensi_sobol_weighted
from formSpace import formSolutionSpace, buildDesignInSpace

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
    # # EXECUTION_NR = 34
    # prepareSAVariants(sa_type = 'morris', set_dup_rvt = True)
    testSensi_morris_weighted(build_design=False, calc_index=True, plot_index=True, pad_constant_sign=0.5)

    # EXECUTION_NR = 39
    # prepareSAVariants(sa_type = 'sobol', set_dup_rvt = True)
    # testSensi_sobol_weighted(build_design=True, calc_index=True, plot_index=True)

    #=================================#
    #                 ss              #
    #=================================#
    # EXECUTION_NR = 100 + (14, 54, 19, 59)
    
    # ITERATION-1
    # formSolutionSpace(['\sa-34-0.3'], set_evolve_space = True, set_dup_rvt = True, set_new_space = True)
    # buildDesignInSpace()

    # ITERATION-2
    # formSolutionSpace(['\sa-14-0.3', '\ss-114-1'], sweep_config = [4, 0.1], set_evolve_space = True, set_dup_rvt = True, set_new_space = True)
    # buildDesignInSpace()
    
    # ITERATION-3
    # formSolutionSpace(['\sa-14-0.3', '\ss-114-1', '\ss-114-2'], set_evolve_space = False, set_dup_rvt = False, set_new_space = True)
    # buildDesignInSpace()

    #=================================#
    #                cls              #
    #=================================#








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