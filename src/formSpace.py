#
# formSpace.py
#

# import modules

from base_external_packages import *

from Space import SolutionSpace
# from Design import Design

from funct_data import save_dict, load_dict, get_problems_from_paths, flatten, create_directory, duplicateRVT
from const_project import DIRS_DATA, FILE_INIT_SKL_RVT
from const_solus import DIRS_DATA_SS, DIRS_DATA_SS_DUP, DIRS_DATA_SS_RES, FILE_SS_VARY_SWEEP, ITERATION_VALUES

from testSensitivity import buildDesigns

def formSolutionSpace(dataset=[], set_evolve_space=False, sweep_config = [[],[]], set_dup_rvt = False, set_new_space=False):

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
    initialSpace.form_space(designsNew)
    initialSpace.enrich_space()

    # save the initial Space.
    nameFileSpace = '_'.join([data.replace('\\','') for data in dataset])
    save_dict(initialSpace, DIRS_DATA_SS + r'\Space_' + nameFileSpace + r'.pickle')

    # evolvement in the initial Space.
    if set_evolve_space:
        
        
        # save the input for varyGP.dyn
        initialSpace._config_sweeping(
            set_sweep_density=sweep_config[0],
            set_sweep_ext_pad=sweep_config[1],
            )
        
        iter_evolve_aspect = [] if ITERATION_VALUES == 1 else ['compliance']

        initialSpace.evolve_space(
            evolve_aspects=iter_evolve_aspect,
            vary_file=FILE_SS_VARY_SWEEP
            )

        if set_dup_rvt:
            
            # duplicat the .rvts for variation.
            duplicateRVT(FILE_INIT_SKL_RVT, DIRS_DATA_SS_DUP, amount=initialSpace.evolve_samples.shape[0], clear_destination=True)

def buildDesignInSpace():
    buildDesigns(
        FILE_SS_VARY_SWEEP,
        newdesigns_in_path = DIRS_DATA_SS_RES,
        newdesigns_out_path = DIRS_DATA_SS,
        build_ini=False,
        build_new=True)