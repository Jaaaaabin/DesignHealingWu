#
# const_solutions.py
#

from const_project import SOLUTION_NR, DIRS_DATA

#=================================#
#          SolutionSpace          #
#=================================#
ITERATION_VALUES = 7

# 7 explore_space_by_lhs
# explore_range=0.6,
# num_samples=600,
# random_seed=1008,
# scramble=False
# optimization='random-cd', 


DIRS_DATA_SS = DIRS_DATA + r'\ss-' + str(SOLUTION_NR) + '-' + str(ITERATION_VALUES)

FILE_SS_VARY_SWEEP = DIRS_DATA_SS + r'\ss_vary_sweep.csv'
FILE_SS_VARY_SKEWNORMAL = DIRS_DATA_SS + r'\ss_vary_skewnormal.csv'
FILE_SS_VARY_LHS = DIRS_DATA_SS + r'\ss_vary_lhs.csv'

DIRS_DATA_SS_DUP = DIRS_DATA_SS + r'\dups'
DIRS_DATA_SS_VARY = DIRS_DATA_SS + r'\vary'
DIRS_DATA_SS_RES = DIRS_DATA_SS + r'\res'
DIRS_DATA_SS_FIG = DIRS_DATA_SS + r'\fig'
