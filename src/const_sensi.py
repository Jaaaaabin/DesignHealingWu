#
# const_sensi.py
#
from const_project import EXECUTION_NR, DIRS_DATA


# SA boundary information
BOUNDARY_VALUES_LIST = [0.01, 0.05, 0.1, 0.3]
BOUNDARY_VALUES = BOUNDARY_VALUES_LIST[int(EXECUTION_NR%5-1)]
SET_SA_DISTRIBUTION ="unif"         # "unif", "norm"


NAME_FLOOR = 'U1_OK'
NAME_FAILURES = ['2SzsE5m8T4h9JlM6XpBSn3','2SzsE5m8T4h9JlM6XpBSnd']
EXCEPTION_GP = 'U1_OK_d_wl_sn25'


#=================================#
#              sobol              #
#=================================#
# amount = (N_p)
N_SMP_SOBOL = int(2**(EXECUTION_NR%10 - 4))    #= 2** (int
SA_CALC_SECOND_ORDER = True
SALTELLI_SKIP = 1024

#=================================#
#             morris              #
#=================================#
# amount = (N_p +1)* N_OPT_TRAJ_MORRIS
N_LEVEL_MORRIS = 6 # fix to 6 levels.
N_TRAJ_MORRIS = 500 # total trajs
N_OPT_TRAJ_MORRIS = 16 # 8 for 1..; 16 for 5..

# initial parameter selction via neighborhood levels.
K_LEVEL_PARAMETER = int(EXECUTION_NR/10) #= 1 , 3 , 5

DIRS_DATA_SA = DIRS_DATA + r'\sa-' + str(EXECUTION_NR) + '-' + str(BOUNDARY_VALUES)
FILE_SA_VARY_SOBOL = DIRS_DATA_SA + r'\sa_vary_sobol.csv'
FILE_SA_VARY_MORRIS = DIRS_DATA_SA + r'\sa_vary_morris.csv'
DIRS_DATA_SA_DUP = DIRS_DATA_SA + r'\dups'
DIRS_DATA_SA_VARY = DIRS_DATA_SA + r'\vary'
DIRS_DATA_SA_RES = DIRS_DATA_SA + r'\res'
DIRS_DATA_SA_FIG = DIRS_DATA_SA + r'\fig'
