#
# tempoCheck.py
#

# import modules

from base_external_packages import *

from const_sensi import N_LEVEL_MORRIS
from funct_data import load_dict

sa_morris_nr = 12
sa_morris_bdry = [0.02, 0.1, 0.2, 0.25]
failure_list = ['2SzsE5m8T4h9JlM6XpBSn3_IBC1020_2', '2SzsE5m8T4h9JlM6XpBSnd_IBC1207_1','2SzsE5m8T4h9JlM6XpBSnd_IBC1207_3']
data_sa_paths = [r'C:\dev\phd\ModelHealer\data\sa-'+str(sa_morris_nr)+'-'+str(bdry) for bdry in sa_morris_bdry]
sa_problems = [data_sa_path + r'\sa_problem.pickle' for data_sa_path in data_sa_paths]
input_x = [data_sa_path + r'\sa_values_morris.txt' for data_sa_path in data_sa_paths]
result_ys = [[data_sa_path + r'\res\results_y' +  fl + '.txt' for data_sa_path in data_sa_paths] for fl in failure_list]

for ys in result_ys:
    
    fig = plt.figure(figsize=(12,10))  # unit of inch
    ax1 = fig.add_subplot(4,1,1)
    ax2 = fig.add_subplot(4,1,2, sharex = ax1)
    ax3 = fig.add_subplot(4,1,3, sharex = ax1)
    ax4 = fig.add_subplot(4,1,4, sharex = ax1)
    
    # ax1 = plt.axes((0.15, 0.05, 0.80, 0.20))  # in range (0,1)
    # ax2 = plt.axes((0.15, 0.275, 0.80, 0.20))  # in range (0,1)
    # ax3 = plt.axes((0.15, 0.50, 0.80, 0.20))  # in range (0,1)
    # ax4 = plt.axes((0.15, 0.725, 0.80, 0.20))  # in range (0,1)

    axes = [ax1, ax2, ax3, ax4]
    for sa_problem, x, y, ax in zip(sa_problems, input_x, ys, axes):
        problem = load_dict(sa_problem)
        X = np.loadtxt(x)
        Y = np.loadtxt(y, float)

        Si = analyze_morris.analyze(problem, X, Y, conf_level=0.95, print_to_console=False, num_levels=N_LEVEL_MORRIS)
        plot_morris.horizontal_bar_plot(ax, Si)
        plt.savefig(r'C:\dev\phd\ModelHealer\data\test.png',dpi=200)
        
print('end')