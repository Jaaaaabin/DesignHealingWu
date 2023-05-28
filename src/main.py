#
# __main__.py
# 

from topoCollection import topoCollect
from graphCreation import graphCreate
from graphEnricher import graphEnrich
from prepareVariation import prepareVariants
from testSensitivity import testSensi
# from prepareSolutionSpace import buildSpace

if __name__ == "__main__":
    
    # topoCollect()
    # graphCreate()
    graphEnrich()
    # prepareVariants(set_dup_rvt=True)
    # testSensi(build_design=True, calc_index=True, plot_index=True)


# to do 
# 1. develop a Erro/Warning ignoration command.
# 2. add parameters to doors and windows.
# 2.1 ground floor added and succeed.
# 2.2 upper floors not completed yet.