#
# testSensitivity.py
#

from const_project import DIRS_INI_RES, DIRS_DATA_SA_RES
from const_ibcrule import BUILDING_RULES
from funct_h5 import analyze_h5s

# Add failure information and search neighbors related to the failure.

# dictionary: design - > rule -> target -> distance & compliance.
ini_dictCheckResult_h5s = analyze_h5s(DIRS_INI_RES, BUILDING_RULES)
dictCheckResult_h5s = analyze_h5s(DIRS_DATA_SA_RES, BUILDING_RULES)

print ('end')

