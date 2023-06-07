#
# const_sensi.py
#
from const_project import EXECUTION_NR

NAME_FLOOR = 'U1_OK'
NAME_FAILURES = ['2SzsE5m8T4h9JlM6XpBSn3','2SzsE5m8T4h9JlM6XpBSnd']
EXCEPTION_GP = 'U1_OK_d_wl_sn25'

# SA overall information
N_SMP = 2**(EXECUTION_NR%10)    #= 2** (int
SA_CALC_SECOND_ORDER = True

# initial parameter selction
K_LEVEL_PARAMETER = int(EXECUTION_NR/10) #= 1 , 3 , 5

# sa boundary information
BOUNDARY_VALUES = 0.02
SET_SA_DISTRIBUTION ="unif"         # "unif", "norm"
SALTELLI_SKIP = 1024
