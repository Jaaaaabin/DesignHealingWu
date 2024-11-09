#
# __main__.py
#

from const_project import EXECUTION_NR
from const_solus import FILE_SS_VARY_LHS

from topoCollection import topoCollect
from graphCreation import graphCreate
from graphEnricher import graphEnrich
from graphEnrich_withoutchecking import graphEnrich_withoutchecking
from prepareVariation import prepareSAVariants
from testSensitivity import testSensi_morris_weighted
from summarizeSensitivity import summarizeSensi
from formSpace import exploreLHS,  buildDesignInSpace, formSolutionSpace, reasonSolutionSpaceWeighted

from pairRooms import getRoomPairs

if __name__ == "__main__":

    # =================================#
    #                ini               #
    # =================================#
    # # initial preparation : collect the design topology.
    # topoCollect()
    
    # # initial preparation : build up the topological connectiviy graph from collected data from authoring tool.
    # graphCreate()

    # # graphEnrich_withoutchecking(plot_graph=True)
    # # getRoomPairs()

    # # initial preparation : enrich the created graph structure: embed the investigated desgin constraints and related design parameters.
    # graphEnrich(plot_graph=False)
    
    # # =================================#
    # #                 sa               #
    # # =================================#
    # # ececution of the sensitivity analysis.
    # EXECUTION_NR = 34       # nr=34 is one of the tested experiments that have the best search performance.
    # prepareSAVariants(sa_type = 'morris', set_dup_rvt = False)
    # for beta in [0, 0.5, 1]: # final figures are for different beta values.
    #     testSensi_morris_weighted(build_design=False, calc_index=True, plot_index=True, beta_coef_reduction=beta) # plot for convar bar figures.
    # summarizeSensi() #summarized visualization (mu and mu-star) for all the investigated building rules.

    # # =================================#
    # #                 ss              #
    # # =================================#
    # # ececution of the multi-step LHS-based design solution exploration.
    # # round 1
    EXECUTION_NR = 34
    ITERATION_VALUES = 0
    # exploreLHS(
    #     '\sa-34-0.3',
    #     num_samples=250,
    #     explore_range=0.3,
    #     lhs_optimization='random-cd',
    #     set_dup_rvt = True,
    #     set_new_space = True)
    # buildDesignInSpace(file_variation=FILE_SS_VARY_LHS)
    # formSolutionSpace(['\sa-34-0.3', '\ss-134-0'], set_new_space = False)

    # # ================================#
    # # round 2
    ITERATION_VALUES = 1
    # exploreLHS(
    #     '\sa-34-0.3',
    #     num_samples=1000,
    #     explore_range=0.6,
    #     lhs_optimization='random-cd',
    #     set_dup_rvt = False,
    #     set_new_space = True)
    # buildDesignInSpace(file_variation=FILE_SS_VARY_LHS)
    # formSolutionSpace(
    #     ['\sa-34-0.3', '\ss-134-1'],set_new_space = False)
    # reasonSolutionSpace(
    #     ['\sa-34-0.3', '\ss-134-1'], set_new_space = False, plot_space_pairwise = True)

    # formSolutionSpace(
    #     ['\sa-34-0.3', '\ss-134-0', '\ss-134-1'], set_new_space = False)
    # reasonSolutionSpace(
    #     ['\sa-34-0.3', '\ss-134-0', '\ss-134-1'], set_new_space = False, plot_space_pairwise = True)
    
    # # ================================#
    # # round 3
    ITERATION_VALUES = 2
    # exploreLHS(
    #     '\sa-34-0.3',
    #     num_samples=500,
    #     explore_range = r'C:\dev\phd\ModelHealer\data\ss-134-1\compliance_valid_ranges.pickle',
    #     lhs_optimization='lloyd',
    #     set_dup_rvt = True,
    #     set_new_space = True)
    # buildDesignInSpace(file_variation=FILE_SS_VARY_LHS)
    # formSolutionSpace(
    #     ['\sa-34-0.3', '\ss-134-0', '\ss-134-1', '\ss-134-2'], set_new_space = False)
    
    # # paper revision.
    formSolutionSpace(
        ['\sa-34-0.3', '\ss-134-0', '\ss-134-1', '\ss-134-2'], set_new_space = False)
    reasonSolutionSpaceWeighted(
        ['\sa-34-0.3', '\ss-134-0', '\ss-134-1', '\ss-134-2'],
        calc_distance = True)