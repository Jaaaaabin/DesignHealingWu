#
# __main__.py
#

from const_project import EXECUTION_NR
from const_solus import FILE_SS_VARY_LHS

from topoCollection import topoCollect
from graphCreation import graphCreate
from graphEnricher import graphEnrich
from prepareVariation import prepareSAVariants
from testSensitivity import testSensi_morris_weighted
from summarizeSensitivity import summarizeSensi
from formSpace import exploreLHS,  buildDesignInSpace, formSolutionSpace

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
    # EXECUTION_NR = 34
    # prepareSAVariants(sa_type = 'morris', set_dup_rvt = False)
    # for beta in [1, 0.5, 0]: # final figures are for the last element.
    #     testSensi_morris_weighted(build_design=False, calc_index=True, plot_index=True, beta_coef_reduction=beta)
    # summarizeSensi()
    
    #=================================#
    #                 ss              #
    #=================================#
    # EXECUTION_NR = 34
    # ITERATION_VALUES = 1
    # exploreLHS(
    #     '\sa-34-0.3',
    #     num_samples=1000,
    #     explore_range=0.6,
    #     lhs_optimization='random-cd',
    #     set_dup_rvt = False,
    #     set_new_space = True)
    # buildDesignInSpace(file_variation=FILE_SS_VARY_LHS)
    # formSolutionSpace(['\sa-34-0.3', '\ss-134-1'], set_new_space = False)
    
    #-================================#
    # ITERATION_VALUES = 2
    # exploreLHS(
    #     '\sa-34-0.3',
    #     num_samples=500,
    #     explore_range = r'C:\dev\phd\ModelHealer\data\ss-134-1\compliance_valid_ranges.pickle',
    #     lhs_optimization='lloyd',
    #     set_dup_rvt = True,
    #     set_new_space = True)
    buildDesignInSpace(file_variation=FILE_SS_VARY_LHS)
    formSolutionSpace(['\sa-34-0.3', '\ss-134-1', '\ss-134-2'], set_new_space = False)
    
    #=================================#
    #                 rs              #
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