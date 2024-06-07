#
# tempoCheck.py
#

# import modules

from base_external_packages import *
from const_project import EXECUTION_NR
from const_sensi import N_LEVEL_MORRIS, DIRS_DATA_SA, DIRS_DATA_SA_FIG
from funct_data import load_dict, save_dict, stnd_nrml
from const_ibcrule import BUILDING_RULES, BUILDING_RULES_ALL

# standarlization and normalization will not help.
# pd_X = pd.DataFrame(X)
# pd_X_stnl, X_α_β = stnd_nrml(pd_X)
# X_stnl = pd_X_stnl.to_numpy()


def _sort_Si(Si, key, sortby='mu_star'):
    return np.array([Si[key][x] for x in np.argsort(Si[sortby])])


def morris_horizontal_bar_plot(
    ax,
    Si,
    plotmu,
    bcolor,
    bheight,
    y_loc,
    lwidth=0.25,
    alpah=1,
    opts=None,
    unit='',
    x_max_abs=1.0):

    '''Updates a matplotlib axes instance with a horizontal bar plot
    of mu_star, with error bars representing mu_star_conf.
    '''

    sortby = plotmu
    assert sortby in ['mu_star', 'mu_star_conf', 'sigma', 'mu']

    if opts is None:
        opts = {}

    # Sort all the plotted elements by mu_star (or optionally another metric)
    names_sorted = _sort_Si(Si, 'names', sortby)

    mu_sorted = _sort_Si(Si, 'mu', sortby)
    sigma_sorted = _sort_Si(Si, 'sigma', sortby)

    mu_star_sorted = _sort_Si(Si, 'mu_star', sortby)
    mu_star_conf_sorted = _sort_Si(Si, 'mu_star_conf', sortby)

    if plotmu == 'mu_star':
        mean, varian = mu_star_sorted, mu_star_conf_sorted
    elif plotmu == 'mu':
        mean, varian = mu_sorted, sigma_sorted

    # Plot horizontal barchart
    y_pos = np.arange(len(mu_star_sorted))
    plot_names = names_sorted

    out = ax.barh(y_pos + y_loc,
                  mean,
                  xerr=varian,
                  color=bcolor,
                  height=bheight,
                  linewidth=lwidth,
                  alpha=alpah,
                  align='center',
                  ecolor='black',
                  edgecolor='black',
                  **opts)
    
    if plotmu == 'mu_star':
        ax.set_xlabel(r'$\mu^\star$' + unit, fontsize=20)
        ax.set_xlim(0.0,x_max_abs)
    elif plotmu == 'mu':
        ax.set_xlabel(r'$\mu$' + unit, fontsize=20)
        ax.set_xlim(-x_max_abs,x_max_abs)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_names)
    ax.set_ylim(min(y_pos)-1, max(y_pos)+1)

    ax.tick_params(axis='y', which='major', labelsize=18)
    ax.tick_params(axis='x', which='major', labelsize=18)

    return out


# calculation.
def summarizeSensi():

    # summarizing plot settings.
    sa_betas = [1, 0.5, 0]
    sa_bdry_colors = ['navy', 'royalblue', 'cyan'] # 'orange' #'darkgoldenrod'
    y_limit_c = 0.30
    y_height = (y_limit_c*2)/(len(sa_bdry_colors)-1)
    y_locs = np.linspace(-y_limit_c, y_limit_c, num=len(sa_bdry_colors)).tolist()

    # data extraction
    sa_problem = DIRS_DATA_SA + r'\sa_problem.pickle'
    X = np.loadtxt(DIRS_DATA_SA + r'\sa_values_morris.txt')
    Ys = [[DIRS_DATA_SA + r'\res\results_y_' +  rl  + '_beta_' + str(beta) + '.txt' for beta in sa_betas] for rl in BUILDING_RULES_ALL]
    sa_indices_all = dict()

    for plot_mu in ['mu', 'mu_star']:

        sa_indices_type = dict()
        
        for rl, ys in zip(BUILDING_RULES_ALL, Ys):
            
            fig = plt.figure(figsize=(16,8))  # unit of inch
            ax = plt.axes((0.15, 0.10, 0.80, 0.80))  # in range (0,1)
            sa_indices = dict()

            for (y, y_loc, beta, bdry_color) in zip(ys, y_locs, sa_betas, sa_bdry_colors):

                # load
                problem = load_dict(sa_problem)
                Y = np.loadtxt(y, float)

                # calculate Si
                Si = analyze_morris.analyze(problem, X, Y, conf_level=0.95, print_to_console=False, num_levels=N_LEVEL_MORRIS)

                # shorten the Si names.
                Si['names'] =  [name.replace("U1_OK_d_wl_","") for name in Si['names']]

                # default plot
                bar = morris_horizontal_bar_plot(
                    ax,
                    Si,
                    plot_mu, # plot by mu or mu_star
                    y_loc=y_loc,
                    bcolor=bdry_color,
                    bheight=y_height,
                    lwidth=0.5,
                    alpah=0.80,
                    )
                sa_indices.update({beta: Si})

                bar.set_label(r'$\beta = {}$'.format(beta))
                plt.legend(loc='lower right', prop={'size': 18})
                plt.savefig(DIRS_DATA_SA_FIG + r'\{nr}_{sort}_{failure}.png'.format(
                    nr=EXECUTION_NR,sort=plot_mu,failure=rl), dpi=400, bbox_inches='tight', pad_inches=0.05)
                # here to do. axis label size and also the other figure.

            sa_indices_type.update({rl: sa_indices})
            save_dict(sa_indices_type, DIRS_DATA_SA + r'\sa_morris_indices_' + str(plot_mu) + '.pickle')

        sa_indices_all.update({plot_mu: sa_indices_type})


# sa_sum_dict = dict()
# for bt in sa_betas:
#     test_dict = load_dict(DIRS_DATA_SA+r'\sa_morris_indices_beta_'+str(bt) + r'.pickle')
#     test_dict_bt = {test_dict['IBC1020_2']['names'][k]: test_dict['IBC1020_2']['mu'][k] +test_dict['IBC1207_1']['mu'][k] + test_dict['IBC1207_3']['mu'][k] for k in np.arange(len(test_dict['IBC1020_2']['names']))}
#     sa_sum_dict.update({bt: test_dict_bt})
