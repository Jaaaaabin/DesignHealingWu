#
# designVariation.py
#

# import modules
from const_project import DIRS_DATA_SA, DIRS_DATA_SA_DUP, FILE_SA_VARY
from const_project import FILE_INIT_RVT, DIRS_DATA_SA_DUP, FILE_SA_PARAM_LIST
from const_sensi import N_SMP, SA_CALC_SECOND_ORDER, BOUNDARY_VALUES, SET_SA_DISTRIBUTION, SALTELLI_SKIP

# from funct_sensi import *
from base_functions import *
from funct_sensi import duplicateRVT

sa_parameter_data = pd.read_csv(FILE_SA_PARAM_LIST, names=['names', 'values'], header=None)
sa_parameter_values = sa_parameter_data['values'].tolist()

sa_parameter_names = sa_parameter_data['names'].tolist()
sa_parameter_num = len(sa_parameter_names)
sa_parameter_bounds = np.array([[v-BOUNDARY_VALUES, v+BOUNDARY_VALUES] for v in sa_parameter_values]).reshape((sa_parameter_num,2))

# values via Saltelli’s extension of the Sobol’ sequence
sa_problem = {
    'num_vars': sa_parameter_num,
    'names': sa_parameter_names,
    'bounds': sa_parameter_bounds,
    # 'bounds': np.array(list([-BOUNDARY_VALUES, BOUNDARY_VALUES] * sa_parameter_num)),
    'dists': np.array([SET_SA_DISTRIBUTION] * sa_parameter_num),
    }

sa_values = saltelli.sample(sa_problem, N_SMP, calc_second_order=SA_CALC_SECOND_ORDER, skip_values=SALTELLI_SKIP)
save_ndarray_2txt(sa_values, DIRS_DATA_SA+"/sa_values.txt")
save_dict(sa_problem, DIRS_DATA_SA+"/sa_problem.pickle")

# build_variation_txt_csv(DIRS_BIM, project_name, FILE_SAMPLES_VARIATION, sa_samples_df)
sa_variation_names = np.array([sa_parameter_names])
df_sa_variation = pd.DataFrame(sa_values, columns=sa_parameter_names).T
df_sa_variation.to_csv(FILE_SA_VARY, header=False)

duplicateRVT(FILE_INIT_RVT, DIRS_DATA_SA_DUP, amount=sa_values.shape[0], clear_destination=True) # duplicate the .rvt