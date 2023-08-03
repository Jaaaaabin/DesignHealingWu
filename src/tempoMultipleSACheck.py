#
# tempoCheck.py
#

# import modules

from base_external_packages import *
from const_sensi import N_LEVEL_MORRIS, DIRS_DATA
from funct_data import load_dict, save_dict
from const_ibcrule import BUILDING_RULES


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


sa_morris_nr = 34
sa_pads = [0, 0.1, 0.25, 0.5]
sa_morris_bdry_colors = ['navy', 'royalblue', 'cyan', 'orange'] #'darkgoldenrod'
y_limit_c = 0.30
y_height = (y_limit_c*2)/(len(sa_morris_bdry_colors)-1)
y_locs = np.linspace(-y_limit_c, y_limit_c, num=len(sa_morris_bdry_colors)).tolist()
print (y_locs)

sa_path = r'C:\dev\phd\ModelHealer\data\sa-34-0.3'
sa_problem = sa_path + r'\sa_problem.pickle'
input_x = sa_path + r'\sa_values_morris.txt'
result_ys = [[sa_path + r'\res\results_y_' +  rl  + '_pad_' + str(pad) + '.txt' for pad in sa_pads] for rl in BUILDING_RULES]

X = np.loadtxt(input_x)

sa_indices_all = dict()

for rl, ys in zip(BUILDING_RULES, result_ys):
    
    fig = plt.figure(figsize=(10,5))  # unit of inch
    ax = plt.axes((0.15, 0.05, 0.80, 0.90))  # in range (0,1)
    sa_indices = dict()

    for i, (y, y_loc, pad, bdry_color) in enumerate(zip(ys, y_locs, sa_pads, sa_morris_bdry_colors)):

        # load
        problem = load_dict(sa_problem)
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
            y_loc=y_loc,
            bcolor=bdry_color,
            bheight=y_height,
            lwidth=0.5,
            alpah=0.80,
            )
        sa_indices.update({pad: Si})

        # refine plot
        bar.set_label(pad)
        plt.legend(loc='lower right')
        plt.savefig(r'C:\dev\phd\ModelHealer\data\{nr}_{sort}_{failure}.png'.format(nr=sa_morris_nr,sort=plot_mu,failure=rl), dpi=200)
    
    sa_indices_all.update({rl: sa_indices})
    save_dict(sa_indices_all, DIRS_DATA + r'\sa_morris_indices_' + str(plot_mu) + '.pickle')

    # next step is to compare
    # C:\dev\phd\ModelHealer\data\sa_morris_indices_mu.pickle
    # with
    # C:\dev\phd\ModelHealer\data\sa-34-0.3\sa_morris_indices_pad_0.pickle
    # C:\dev\phd\ModelHealer\data\sa-34-0.3\sa_morris_indices_pad_0.1.pickle
    # C:\dev\phd\ModelHealer\data\sa-34-0.3\sa_morris_indices_pad_0.25.pickle
    # C:\dev\phd\ModelHealer\data\sa-34-0.3\sa_morris_indices_pad_0.5.pickle