#
# funct_sensi.py
#

# import packages
from base_external_packages import *

# import modules
from base_functions import *
# from base_classes import Design
from funct_plot import sobol_plot_sa_S1ST, sobol_plot_sa_S2, morris_sa_plot


# def getRVTFilename(file_dir, outpath, remove_ext = True):
#     """
#     write the .rvt files into a csv for controling.
#     """

#     # list to store files
#     res = []

#     # Iterate directory
#     for path in os.listdir(file_dir):
#         # check if current path is a file
#         if os.path.isfile(os.path.join(file_dir, path)):
#             if path.endswith('.rvt'):
#                 res.append(path)
    
#     if remove_ext:
#         res = [x.replace('.rvt', '') for x in res]

#     df_rvtids = pd.DataFrame(res)
#     df_rvtids.to_csv(outpath, header=False, index=False)
#     print('Extraction of duplicated RVT filenames (with student IDs) Succeed.')


def collect_ini_sa_parameters(
        file_sa_parameter_list,
        k_level_parameter,
        set_floor=[],
        exclude_gp=[]):
    """
    collect the initial parameters from a csv.
    """

    path_csv = file_sa_parameter_list.replace('tbd', str(k_level_parameter))
    data = pd.read_csv(path_csv, names=['names', 'values'], header=None)
    names = data['names'].tolist()
    values = data['values'].tolist()
    
    # including.
    if set_floor:
        idx = [id for id in range(len(names)) if set_floor in names[id]]
        names = [names[id] for id in idx]
        values = [values[id] for id in idx]
    
    # excluding.
    if exclude_gp:
        idx = [id for id in range(len(names)) if exclude_gp not in names[id]]
        names = [names[id] for id in idx]
        values = [values[id] for id in idx]   
    
    num = len(names)

    return names, values, num


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


def execute_sa_sobol(
        dirs_fig,
        problem,
        target,
        rule,
        y_result_txt,
        sa_calc_second_order=True,
        plot_res=False,
        plot_res_1_T=False,
        plot_res_2=False):
    
    """
    execute Sobol sensitivity analysis per checking target and checking rule.
    """

    total, first, second = [], [], []

    # execute the sobol analysis.
    Y = np.loadtxt(y_result_txt, float)
    Si = analyze_sobol.analyze(problem, Y, calc_second_order=sa_calc_second_order, print_to_console=False)
    
    if sa_calc_second_order:
        total, first, second = Si.to_df()
        total.append(total)
        first.append(first)
        second.append(second)
    else:
        total, first = Si.to_df()
        total.append(total)
        first.append(first)
        
    # plot settings
    if plot_res:
        
        # 1st and total-order indices are anyway alwayes calculated.
        if plot_res_1_T:
            sobol_plot_sa_S1ST(dirs_fig, target, rule, first, total)

        # only if 2nd sensitivity indices are calculated.
        if sa_calc_second_order:
            if plot_res_2:
                sobol_plot_sa_S2(dirs_fig, target, rule, second)
    
    return total, first, second

def execute_sa_morris(
    dirs_fig,
    problem,
    target,
    rule,
    input_x_txt,
    y_result_txt,
    num_levels=4,
    plot_res=False,
    ):

    X = np.loadtxt(input_x_txt)
    Y = np.loadtxt(y_result_txt, float)
    
    Si = analyze_morris.analyze(problem, X, Y, conf_level=0.95, print_to_console=False, num_levels=num_levels)
    # A dictionary of sensitivity indices containing the following entries.
    # `mu` - the mean of the distribution.
    # `mu_star` - the mean of the distribution of absolutevalues.
    # `sigma` - the standard deviation of the distribution.
    # `mu_star_conf` - the bootstrapped confidence interval
    # `names` - the names of the parameters

    if plot_res:
        morris_sa_plot(
            dirs_fig,
            target,
            rule,
            Si,
            input_sample=X,
            problem=problem,
            )
          
    return Si