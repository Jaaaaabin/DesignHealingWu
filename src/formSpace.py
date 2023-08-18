#
# formSpace.py
#

# import modules

from base_external_packages import *

from Space import SolutionSpace

# from Design import Design
from testSensitivity import buildDesigns

from funct_data import save_dict, load_dict, get_problems_from_paths, flatten, create_directory, duplicateRVT
from const_project import DIRS_DATA, FILE_INIT_SKL_RVT, DIRS_DATA_TOPO
from const_solus import DIRS_DATA_SS, DIRS_DATA_SS_DUP, DIRS_DATA_SS_RES, FILE_SS_VARY_LHS
from const_solus import DIRS_DATA_SS_FIG, ITERATION_VALUES, FILE_SS_VARY_SKEWNORMAL


def buildDesignInSpace(file_variation):
    buildDesigns(
        file_variation = file_variation,
        newdesigns_in_path=DIRS_DATA_SS_RES,
        newdesigns_out_path=DIRS_DATA_SS,
        build_ini=False,
        build_new=True)
    

def exploreLHS(
    dataset=[],
    num_samples=200,
    explore_range=0.6,
    lhs_optimization='random-cd',
    set_dup_rvt=False,
    set_new_space=False):
    
    if set_new_space:
        create_directory(DIRS_DATA_SS)

    pathIni = DIRS_DATA + dataset + r'\DesignIni.pickle'
    pathRes = DIRS_DATA + dataset + r'\res'
    problems =  get_problems_from_paths(pathRes)

    designIni = load_dict(pathIni)
    # del designIni.parameters["U1_OK_d_wl_sn25"]

    pathNew = DIRS_DATA + dataset + r'\DesignsNew.pickle'
    designsNew = load_dict(pathNew)

    # form the initial Space.
    initialSpace = SolutionSpace(problems)
    initialSpace.___setcenter__(designIni)
    
    # add another step: fill the space with the SA sampled points.

    # add sa results.
    indicesSA = load_dict(DIRS_DATA + dataset + r'\sa_morris_indices_mu.pickle')
    indicesSA = {k:v[0] for k,v in indicesSA.items()}

    initialSpace.enrich_sensitivity(indicesSA,'IBC_all')

    initialSpace.explore_space_by_lhs(
        num_samples=num_samples,
        explore_range=explore_range,
        lhs_optimization=lhs_optimization,
        random_seed=521,
    )

    initialSpace.samples_by_lhs.to_csv(FILE_SS_VARY_LHS, header=False)

    if set_dup_rvt:      
        # duplicat the .rvts for variation.
        duplicateRVT(FILE_INIT_SKL_RVT, DIRS_DATA_SS_DUP, amount=initialSpace.samples_by_lhs.shape[1], clear_destination=True)


def formSolutionSpace(
    dataset=[],
    set_new_space=False):

    if set_new_space:
        create_directory(DIRS_DATA_SS)

    filtered_ids = load_dict(DIRS_DATA_TOPO + "/filtered_id.pickle")

    pathIni = DIRS_DATA + dataset[0] + r'\DesignIni.pickle'
    pathRes = DIRS_DATA + dataset[0] + r'\res'
    problems =  get_problems_from_paths(pathRes)

    designIni = load_dict(pathIni)

    pathsNew = [DIRS_DATA + set + r'\DesignsNew.pickle' for set in dataset[1:]]
    designsNew = flatten([load_dict(path) for path in pathsNew])

    # form the initial Space.
    initialSpace = SolutionSpace(problems)
    initialSpace.___setguids__(filtered_ids)
    initialSpace.___setcenter__(designIni)

    initialSpace.form_space(designIni, designsNew)
    initialSpace.__buildxy__(dir=DIRS_DATA_SS, build_valid_subset=True)

    # save outcome Space.
    nameFileSpace = '_'.join([data.replace('\\','') for data in dataset])
    save_dict(initialSpace, DIRS_DATA_SS + r'\Space_' + nameFileSpace + r'.pickle')


# def reasonSolutionSpace(
#     dataset=[],
#     set_new_space=False):
    