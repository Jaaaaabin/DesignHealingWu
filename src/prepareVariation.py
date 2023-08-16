#
# prepareVariation.py
#

# import modules
from const_project import FILE_INIT_SKL_RVT, FILE_SA_PARAM_LIST, DIRS_DATA_TOPO

from const_sensi import DIRS_DATA_SA, DIRS_DATA_SA_DUP, FILE_SA_VARY_SOBOL, FILE_SA_VARY_MORRIS
from const_sensi import K_LEVEL_PARAMETER, NAME_FLOOR
from const_sensi import N_SMP_SOBOL, N_TRAJ_MORRIS, N_LEVEL_MORRIS, N_OPT_TRAJ_MORRIS
from const_sensi import SA_CALC_SECOND_ORDER, BOUNDARY_VALUES, SET_SA_DISTRIBUTION, SALTELLI_SKIP

from funct_data import  create_directory, duplicateRVT, checkSampleDuplication
from funct_sensi import *


def filterFloorLevel():
    # filter the building elements to a specific floor (for now only for space.)
    element_filter_bylevel = pd.read_csv(DIRS_DATA_TOPO + r'\df_space.csv', index_col ='ifcguid')
    element_filter_bylevel_ids = element_filter_bylevel.index[element_filter_bylevel['level'] == NAME_FLOOR].tolist()
    save_dict(element_filter_bylevel_ids, DIRS_DATA_TOPO + "/filtered_id.pickle")


def prepareSAVariants(
    sa_type=[], set_dup_rvt=False):
    
    # check if the data directory exists.
    create_directory(DIRS_DATA_SA)

    # 
    sa_init_parameter_names, sa_init_parameter_values, sa_init_parameter_num = collect_ini_sa_parameters(
        FILE_SA_PARAM_LIST, K_LEVEL_PARAMETER, set_floor = NAME_FLOOR)
    sa_init_parameter_bounds = np.array(
        [[v-BOUNDARY_VALUES, v+BOUNDARY_VALUES] for v in sa_init_parameter_values]).reshape((sa_init_parameter_num,2))

    # values via Saltelli’s extension of the Sobol’ sequence
    sa_problem = {
        'num_vars': sa_init_parameter_num,
        'names': sa_init_parameter_names,
        'bounds': sa_init_parameter_bounds,
        'dists': np.array([SET_SA_DISTRIBUTION] * sa_init_parameter_num),
        }

    save_dict(sa_problem, DIRS_DATA_SA+"/sa_problem.pickle")

    #=================================#
    #                                 #
    #              sobol              #
    #                                 #
    #=================================#
    if sa_type =='sobol':

        sa_values_sobol = sample_saltelli.sample(
            sa_problem,
            N_SMP_SOBOL,
            calc_second_order=SA_CALC_SECOND_ORDER,
            skip_values=SALTELLI_SKIP)
        df_sa_variation_sobol = pd.DataFrame(sa_values_sobol, columns=sa_init_parameter_names).T

        save_ndarray_2txt(sa_values_sobol, DIRS_DATA_SA+"/sa_values_sobol.txt")
        df_sa_variation_sobol.to_csv(FILE_SA_VARY_SOBOL, header=False)
        print ('duplication:',checkSampleDuplication(FILE_SA_VARY_SOBOL))

        if set_dup_rvt:
            duplicateRVT(FILE_INIT_SKL_RVT, DIRS_DATA_SA_DUP, amount=sa_values_sobol.shape[0], clear_destination=True)

    #=================================#
    #                                 #
    #             morris              #
    #                                 #
    #=================================#
    elif sa_type == 'morris':
            
        sa_values_morris = sample_morris.sample(
            sa_problem,
            N_TRAJ_MORRIS,
            num_levels=N_LEVEL_MORRIS,
            optimal_trajectories=N_OPT_TRAJ_MORRIS,
            seed = N_LEVEL_MORRIS,)
        
        df_sa_variation_morris = pd.DataFrame(sa_values_morris, columns=sa_init_parameter_names).T
        save_ndarray_2txt(sa_values_morris, DIRS_DATA_SA+"/sa_values_morris.txt")
        df_sa_variation_morris.to_csv(FILE_SA_VARY_MORRIS, header=False)
        print ('duplication:', checkSampleDuplication(FILE_SA_VARY_MORRIS))
        
        if set_dup_rvt:
            duplicateRVT(FILE_INIT_SKL_RVT, DIRS_DATA_SA_DUP, amount=sa_values_morris.shape[0], clear_destination=True)