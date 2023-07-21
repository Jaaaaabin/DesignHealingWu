#
# const_solutions.py
#

from const_project import SOLUTION_NR, DIRS_DATA

#=================================#
#          SolutionSpace          #
#=================================#
ITERATION_VALUES = 2

DIRS_DATA_SS = DIRS_DATA + r'\ss-' + str(SOLUTION_NR) + '-' + str(ITERATION_VALUES)
FILE_SS_VARY_SWEEP = DIRS_DATA_SS + r'\ss_vary_sweep.csv'
DIRS_DATA_SS_DUP = DIRS_DATA_SS + r'\dups'
DIRS_DATA_SS_VARY = DIRS_DATA_SS + r'\vary'
DIRS_DATA_SS_RES = DIRS_DATA_SS + r'\res'
DIRS_DATA_SS_FIG = DIRS_DATA_SS + r'\fig'
