#
# const_solutions.py
#

from const_project import SOLUTION_NR, DIRS_DATA

#=================================#
#          SolutionSpace          #
#=================================#
ITERATION_VALUES = 2

# 1 explore_space_by_lhs
# explore_range=0.6,
# optimization='random-cd', 
# num_samples=600,
# random_seed=2023,
# scramble=False
# - > no compliant options.

# 2 explore_space_by_lhs
# explore_range=0.6,
# optimization='random-cd', 
# num_samples=1000,
# random_seed=521,
# scramble=False

# 3 explore_space_by_lhs
# explore_range=by results,
# optimization='lloyd', 
# num_samples=500,
# random_seed=521,
# scramble=False

DIRS_DATA_SS = DIRS_DATA + r'\ss-' + str(SOLUTION_NR) + '-' + str(ITERATION_VALUES)

FILE_SS_VARY_SWEEP = DIRS_DATA_SS + r'\ss_vary_sweep.csv'

FILE_SS_VARY_SKEWNORMAL = DIRS_DATA_SS + r'\ss_vary_skewnormal.csv'

FILE_SS_VARY_LHS = DIRS_DATA_SS + r'\ss_vary_lhs.csv'

DIRS_DATA_SS_DUP = DIRS_DATA_SS + r'\dups'
DIRS_DATA_SS_VARY = DIRS_DATA_SS + r'\vary'
DIRS_DATA_SS_RES = DIRS_DATA_SS + r'\res'
DIRS_DATA_SS_FIG = DIRS_DATA_SS + r'\fig'
