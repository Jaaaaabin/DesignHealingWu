#
# funct_sensi.py
#

# import packages
from base_external_packages import *

# import modules
from base_functions import *
# from base_classes import Design
from funct_plot import plot_sa_second_indices, plot_sa_S1ST_per_rule

# define sensitivity analysis functions
def set_sa_low_2_high(
    n_smp, bdry,
    index_smp=0.25,
    index_bdry=20,):
    """
    
    """
    n_smp *=index_smp
    bdry *=index_bdry

    return int(n_smp), bdry

def remove_frozen_group(dt_param, dt_param_bdry, dt_param_grp, group_frozen):
    """
    remove parameters that belong to the frozen group
    
    """
    keys = [key if dt_param_grp[key]== group_frozen else None for key in list(dt_param_grp.keys())]
    keys = [x for x in keys if x is not None]

    new_dt_param = copy.deepcopy(dt_param)
    new_dt_param_bdry = copy.deepcopy(dt_param_bdry)
    new_dt_param_grp = copy.deepcopy(dt_param_grp)
    
    for key in keys:
        new_dt_param.pop(key, None)
        new_dt_param_bdry.pop(key, None)
        new_dt_param_grp.pop(key, None)
    
    return new_dt_param, new_dt_param_bdry, new_dt_param_grp


def saltelli_sampling(
        ref_design,
        group_frozen,
        n_smp=4,
        set_SA_group=False,
        set_SA_distribution='unif',
        sa_calc_second_order=True,
        saltelli_skip=1024):
    """
    sampling based on saltelli ...

    :ref_design:                the referred design (initial design)
    :n_smp:                     number of samples
    :set_SA_group:              consider parameter groups or not 
    :set_SA_distribution:       the type of sampling distribution
        :"unif":                uniform distribution
        :"norm":                normal distribution

    """

    # do not consider those groups that are constant/frozen
    dt_param, dt_param_bdry, dt_param_grp = remove_frozen_group(
        ref_design.parameters, ref_design.parameter_boundaries, ref_design.parameter_groups, group_frozen)
    
    # prepare main components for the sensitivity analysis
    SA_param_names = list(dt_param.keys())
    SA_param_bounds = np.array(list(dt_param_bdry.values()))
    SA_param_num = int(len(SA_param_names))
    SA_param_dists = np.array([set_SA_distribution]*SA_param_num)

    problem = {
        'num_vars': SA_param_num,
        'names': SA_param_names,
        'bounds': SA_param_bounds,
        'dists': SA_param_dists
        }

    # Sampling via Saltelli’s extension of the Sobol’ sequence
    values = saltelli.sample(problem, n_smp, calc_second_order=sa_calc_second_order, skip_values=saltelli_skip)

    return values, problem


def build_samples(ref_design, sa_values, sa_problem):
    """
    build samples from the raw sampling results.

    """

    df_sa_values = pd.DataFrame(sa_values.copy(), columns=sa_problem['names']) 
    
    # create a list(samples) for all samples.
    # the initial design instance are also added here.
    samples = [ref_design]
    
    # build a DataFrame for all samples
    df_samples = pd.DataFrame(ref_design.parameters, index=[ref_design.number])

    nr_model = ref_design.number
    for nr_model in range(nr_model+1,(sa_values.shape[0])+nr_model+1,1):
        
        new_design = Design(nr_model)
        new_design.number = nr_model
        new_design.parameters = copy.deepcopy(ref_design.parameters)
        for param in sa_problem['names']:
            new_design.parameters[param] += df_sa_values.loc[nr_model-1, param]
        samples.append(new_design)
        
        df_tempo = pd.DataFrame(new_design.parameters, index=[new_design.number])
        df_samples = pd.concat([df_samples,df_tempo])

    return samples, df_samples


def execute_sa_sobol(dirs_res, dirs_fig, problem, target_rules, sa_calc_second_order=True, plot_res=False, plot_res_1_T=False, plot_res_2=False):
    """
    execute Sobol sensitivity analysis per checking rule.

    """

    all_total, all_first, all_second = [], [], []

    for single_rule in target_rules:
        
        single_rule_results = np.loadtxt(dirs_res + "/rule_results_" + str(single_rule) + ".txt", float)
        
        # remove the results of initial design.
        Y = single_rule_results[1:]
        
        # execute the sobol analysis.
        Si = sobol.analyze(problem, Y, calc_second_order=sa_calc_second_order, print_to_console=False)
        
        if sa_calc_second_order:
            total, first, second = Si.to_df()
            all_total.append(total)
            all_first.append(first)
            all_second.append(second)
        else:
            total, first = Si.to_df()
            all_total.append(total)
            all_first.append(first)
        
    # plot settings
    if plot_res:
        
        # 1st and total-order indices are anyway alwayes calculated.
        if plot_res_1_T:
            for rl, first_df, total_df in zip(target_rules, all_first, all_total):
                plot_sa_S1ST_per_rule(dirs_fig, rl, first_df, total_df)

        # only if 2nd sensitivity indices are calculated.
        if sa_calc_second_order:
            if plot_res_2:
                for ii in range(len(target_rules)):
                    plot_sa_second_indices(dirs_fig, target_rules[ii], all_second[ii])
    
    return all_total, all_first, all_second


# new functions.

def duplicateRVT(dir_ini, dir_dest, amount=0, clear_destination=True):
    """
    duplicate the initial .rvt file.
    """

    # clear all the previous *.rvt files the destination folder.
    if clear_destination:
        for f in os.listdir(dir_dest):
            if f.endswith(".rvt"):
                os.remove(os.path.join(dir_dest, f))
    if amount > 0 :
        nbs =  [item for item in range(1, amount+1)]
        pathnames = [dir_dest+'\\'+ str(nb) for nb in nbs]
        for pathname in pathnames:
            if os.path.isfile(dir_ini):
                shutil.copy(dir_ini, pathname+'.rvt')
    else:
        return 'Amount Error'

def getRVTFilename(file_dir, outpath, remove_ext = True):
    """
    write the .rvt files into a csv for controling.
    """

    # list to store files
    res = []

    # Iterate directory
    for path in os.listdir(file_dir):
        # check if current path is a file
        if os.path.isfile(os.path.join(file_dir, path)):
            if path.endswith('.rvt'):
                res.append(path)
    
    if remove_ext:
        res = [x.replace('.rvt', '') for x in res]

    df_rvtids = pd.DataFrame(res)
    df_rvtids.to_csv(outpath, header=False, index=False)
    print('Extraction of duplicated RVT filenames (with student IDs) Succeed.')


def collect_ini_sa_parameters(file_sa_parameter_list, k_level_parameter):
    """
    collect the initial parameters from a csv.
    """

    path_csv = file_sa_parameter_list.replace('tbd', str(k_level_parameter))
    data = pd.read_csv(path_csv, names=['names', 'values'], header=None)
    values = data['values'].tolist()
    names = data['names'].tolist()
    num = len(names)

    return names, values, num