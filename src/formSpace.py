#
# formSpace.py
#

# import modules

from base_external_packages import *

from Space import SolutionSpace
# from Design import Design

from funct_data import save_dict, load_dict, get_problems_from_paths, flatten, create_directory, duplicateRVT
from const_project import DIRS_DATA, FILE_INIT_SKL_RVT
from const_solus import DIRS_DATA_SS, DIRS_DATA_SS_DUP, DIRS_DATA_SS_RES, DIRS_DATA_SS_FIG, ITERATION_VALUES, FILE_SS_VARY_SKEWNORMAL, FILE_SS_VARY_LHS

from testSensitivity import buildDesigns


# new-------------------------------------------------------------------------------------------------------------------
def formSolutionSpaceNew(dataset=[], set_evolve_space=False, sweep_config = [[],[]], set_dup_rvt = False, set_new_space=False):

    if set_new_space:
        # check if the data directory exists.
        create_directory(DIRS_DATA_SS)

    pathIni = DIRS_DATA + dataset[0] + r'\DesignIni.pickle'
    pathRes = DIRS_DATA + dataset[0] + r'\res'
    problems =  get_problems_from_paths(pathRes)

    designIni = load_dict(pathIni)
    # del designIni.parameters["U1_OK_d_wl_sn25"]

    pathsNew = [DIRS_DATA + set + r'\DesignsNew.pickle' for set in dataset]
    designsNew = flatten([load_dict(path) for path in pathsNew])

    # form the initial Space.
    initialSpace = SolutionSpace(problems)
    initialSpace.set_center(designIni)
    
    # add another step: fill the space with the SA sampled points.

    # add sa results.
    indicesSA = load_dict(DIRS_DATA + dataset[0] + r'\sa_morris_indices_mu.pickle')
    indicesSA = {k:v[0] for k,v in indicesSA.items()}

    initialSpace.enrich_sensitivity(indicesSA,'IBC_all')
    # initialSpace.explore_space_by_skewnormal(
    #     explore_range=0.3,
    #     alpha_ratio=5,
    #     num_samples=200,
    #     random_seed=2008,
    #     plot_dirs=DIRS_DATA_SS_FIG,
    #     )

    initialSpace.explore_space_by_lhs(
        explore_range=0.6,
        num_samples=600,
        random_seed=1008,
    )
    
    initialSpace.samples_by_lhs.to_csv(FILE_SS_VARY_LHS, header=False)

    if set_dup_rvt:
                
        # duplicat the .rvts for variation.
        duplicateRVT(FILE_INIT_SKL_RVT, DIRS_DATA_SS_DUP, amount=initialSpace.samples_by_lhs.shape[1], clear_destination=True)

    # initialSpace.form_space(designsNew)
    # initialSpace.enrich_space()

    # # save the initial Space.
    # nameFileSpace = '_'.join([data.replace('\\','') for data in dataset])
    # save_dict(initialSpace, DIRS_DATA_SS + r'\Space_' + nameFileSpace + r'.pickle')

    # if set_dup_rvt:
                
    #     # duplicat the .rvts for variation.
    #     duplicateRVT(FILE_INIT_SKL_RVT, DIRS_DATA_SS_DUP, amount=initialSpace.evolve_samples.shape[0], clear_destination=True)


    # evolvement in the initial Space.
    # if set_evolve_space:
        
        
    #     # save the input for varyGP.dyn
    #     initialSpace._config_sweeping(
    #         set_sweep_density=sweep_config[0],
    #         set_sweep_ext_pad=sweep_config[1],
    #         )
        
    #     iter_evolve_aspect = [] if ITERATION_VALUES == 1 else ['compliance']

    #     initialSpace.evolve_space(
    #         evolve_aspects=iter_evolve_aspect,
    #         vary_file=FILE_SS_VARY_SWEEP
    #         )

    #     if set_dup_rvt:
            
    #         # duplicat the .rvts for variation.
    #         duplicateRVT(FILE_INIT_SKL_RVT, DIRS_DATA_SS_DUP, amount=initialSpace.evolve_samples.shape[0], clear_destination=True)



# old-------------------------------------------------------------------------------------------------------------------
# def formSolutionSpace(dataset=[], set_evolve_space=False, sweep_config = [[],[]], set_dup_rvt = False, set_new_space=False):

#     if set_new_space:
#         # check if the data directory exists.
#         create_directory(DIRS_DATA_SS)

#     pathIni = DIRS_DATA + dataset[0] + r'\DesignIni.pickle'
#     pathRes = DIRS_DATA + dataset[0] + r'\res'
#     problems =  get_problems_from_paths(pathRes)

#     designIni = load_dict(pathIni)
#     # del designIni.parameters["U1_OK_d_wl_sn25"]

#     pathsNew = [DIRS_DATA + set + r'\DesignsNew.pickle' for set in dataset]
#     designsNew = flatten([load_dict(path) for path in pathsNew])

#     # form the initial Space.
#     initialSpace = SolutionSpace(problems)
#     initialSpace.set_center(designIni)
#     initialSpace.form_space(designsNew)
#     initialSpace.enrich_space()

#     # save the initial Space.
#     nameFileSpace = '_'.join([data.replace('\\','') for data in dataset])
#     save_dict(initialSpace, DIRS_DATA_SS + r'\Space_' + nameFileSpace + r'.pickle')

#     # evolvement in the initial Space.
#     if set_evolve_space:
        
        
#         # save the input for varyGP.dyn
#         initialSpace._config_sweeping(
#             set_sweep_density=sweep_config[0],
#             set_sweep_ext_pad=sweep_config[1],
#             )
        
#         iter_evolve_aspect = [] if ITERATION_VALUES == 1 else ['compliance']

#         initialSpace.evolve_space(
#             evolve_aspects=iter_evolve_aspect,
#             vary_file=FILE_SS_VARY_SWEEP
#             )

#         if set_dup_rvt:
            
#             # duplicat the .rvts for variation.
#             duplicateRVT(FILE_INIT_SKL_RVT, DIRS_DATA_SS_DUP, amount=initialSpace.evolve_samples.shape[0], clear_destination=True)

def buildDesignInSpace():
    buildDesigns(
        FILE_SS_VARY_LHS,
        newdesigns_in_path = DIRS_DATA_SS_RES,
        newdesigns_out_path = DIRS_DATA_SS,
        build_ini=False,
        build_new=True)