#
# const_sensi.py
#
from const_project import EXECUTION_NR, DIRS_DATA

# SA boundary information
BOUNDARY_VALUES = 0.3
SET_SA_DISTRIBUTION ="unif"         # "unif", "norm"

NAME_FLOOR = 'U1_OK'
NAME_FAILURES = ['2SzsE5m8T4h9JlM6XpBSn3','2SzsE5m8T4h9JlM6XpBSnd']
EXCEPTION_GP = 'U1_OK_d_wl_sn25'

# SA sobol overall information
N_SMP_SOBOL = 2**(EXECUTION_NR%10)    #= 2** (int
SA_CALC_SECOND_ORDER = True
SALTELLI_SKIP = 1024

# SA morris overall information
# amount = (parameter amount +1)* traj
N_LEVEL_MORRIS = 4
N_TRAJ_MORRIS = 10 * (EXECUTION_NR%10)
N_OPT_TRAJ_MORRIS = int(N_TRAJ_MORRIS / 10)

# initial parameter selction via neighborhood levels.
K_LEVEL_PARAMETER = int(EXECUTION_NR/10) #= 1 , 3 , 5

DIRS_DATA_SA = DIRS_DATA + r'\sa-' + str(EXECUTION_NR) + '-' + str(BOUNDARY_VALUES)
FILE_SA_VARY_SOBOL = DIRS_DATA_SA + r'\sa_vary_sobol.csv'
FILE_SA_VARY_MORRIS = DIRS_DATA_SA + r'\sa_vary_morris.csv'
DIRS_DATA_SA_DUP = DIRS_DATA_SA + r'\dups'
DIRS_DATA_SA_VARY = DIRS_DATA_SA + r'\vary'
DIRS_DATA_SA_RES = DIRS_DATA_SA + r'\res'
DIRS_DATA_SA_FIG = DIRS_DATA_SA + r'\fig'
