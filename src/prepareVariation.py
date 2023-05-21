#
# prepareVariation.py
#

# import modules
from const_project import FILE_INIT_RVT, FILE_SA_PARAM_LIST, DIRS_DATA_SA, DIRS_DATA_SA_FIG, DIRS_DATA_SA_DUP, FILE_SA_VARY
from const_sensi import K_LEVEL_PARAMETER, N_SMP, SA_CALC_SECOND_ORDER, BOUNDARY_VALUES, SET_SA_DISTRIBUTION, SALTELLI_SKIP

from funct_sensi import *
# from funct_plot import plot_sa_parallel_parameters
# from base_classes import NewDesign

def prepareVariants(set_dup_rvt=False):

    sa_init_parameter_names, sa_init_parameter_values, sa_init_parameter_num = collect_ini_sa_parameters(
        FILE_SA_PARAM_LIST, K_LEVEL_PARAMETER)
    sa_init_parameter_bounds = np.array(
        [[v-BOUNDARY_VALUES, v+BOUNDARY_VALUES] for v in sa_init_parameter_values]).reshape((sa_init_parameter_num,2))

    # values via Saltelli’s extension of the Sobol’ sequence
    sa_problem = {
        'num_vars': sa_init_parameter_num,
        'names': sa_init_parameter_names,
        'bounds': sa_init_parameter_bounds,
        'dists': np.array([SET_SA_DISTRIBUTION] * sa_init_parameter_num),
        }

    sa_values = saltelli.sample(sa_problem, N_SMP, calc_second_order=SA_CALC_SECOND_ORDER, skip_values=SALTELLI_SKIP)
    save_ndarray_2txt(sa_values, DIRS_DATA_SA+"/sa_values.txt")
    save_dict(sa_problem, DIRS_DATA_SA+"/sa_problem.pickle")

    df_sa_variation = pd.DataFrame(sa_values, columns=sa_init_parameter_names).T
    df_sa_variation.to_csv(FILE_SA_VARY, header=False)

    if set_dup_rvt:
        duplicateRVT(FILE_INIT_RVT, DIRS_DATA_SA_DUP, amount=sa_values.shape[0], clear_destination=True)

