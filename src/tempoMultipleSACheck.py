#
# tempoCheck.py
#

# import modules

from base_external_packages import *
from const_sensi import N_LEVEL_MORRIS
from funct_data import load_dict


sa_morris_nr = 12
sa_morris_bdry = [0.02, 0.05, 0.1, 0.3]
sa_morris_bdry_labels = [str(bdry) for bdry in sa_morris_bdry]
sa_morris_bdry_colors = ['navy', 'royalblue', 'cyan', 'orange']
y_locs = [-0.3, -0.1, 0.1, 0.3]

check_failures = ['2SzsE5m8T4h9JlM6XpBSn3_IBC1020_2', '2SzsE5m8T4h9JlM6XpBSnd_IBC1207_1','2SzsE5m8T4h9JlM6XpBSnd_IBC1207_3']
sa_paths = [r'C:\dev\phd\ModelHealer\data\sa-'+str(sa_morris_nr)+'-'+str(bdry) for bdry in sa_morris_bdry]
sa_problems = [sa_path + r'\sa_problem.pickle' for sa_path in sa_paths]
input_x = [sa_path + r'\sa_values_morris.txt' for sa_path in sa_paths]
result_ys = [[sa_path + r'\res\results_y_' +  fl + '.txt' for sa_path in sa_paths] for fl in check_failures]

def _sort_Si(Si, key, sortby='mu_star'):
    return np.array([Si[key][x] for x in np.argsort(Si[sortby])])

def morris_horizontal_bar_plot(
    ax,
    Si,
    plotmu,
    bcolor,
    bheight,
    lwidth=0.25,
    y_loc=0,
    alpah=1,
    opts=None,
    sortby='mu_star',
    unit=''):

    '''Updates a matplotlib axes instance with a horizontal bar plot
    of mu_star, with error bars representing mu_star_conf.
    '''
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

    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_names)
    ax.set_xlabel(r'$\mu^\star$' + unit)

    ax.set_ylim(min(y_pos)-1, max(y_pos)+1)

    return out


for fl, ys in zip(check_failures, result_ys):
    
    fig = plt.figure(figsize=(10,5))  # unit of inch
    ax = plt.axes((0.15, 0.05, 0.80, 0.90))  # in range (0,1)
    
    for i, (sa_problem, x, y, y_loc, bdry_label, bdry_color) in enumerate(zip(
        sa_problems, input_x, ys, y_locs, sa_morris_bdry_labels, sa_morris_bdry_colors)):

        # widths = data[:, i]
        # starts = data_cum[:, i] - widths

        # load
        problem = load_dict(sa_problem)
        X = np.loadtxt(x)
        Y = np.loadtxt(y, float)

        # calculate Si
        Si = analyze_morris.analyze(problem, X, Y, conf_level=0.95, print_to_console=False, num_levels=N_LEVEL_MORRIS)
        
        # sort type.
        plot_mu = 'mu_star'
        
        # default plot
        bar = morris_horizontal_bar_plot(
            ax,
            Si,
            plot_mu, # plot by mu or mu_star
            bcolor=bdry_color,
            bheight=0.2,
            lwidth=0.5,
            y_loc=y_loc,
            alpah=0.80,
            )
        
        # refine plot
        bar.set_label(bdry_label)
        plt.legend(loc='lower right')
        plt.savefig(r'C:\dev\phd\ModelHealer\data\{nr}_{sort}_{failure}.png'.format(nr=sa_morris_nr,sort=plot_mu,failure=fl), dpi=200)