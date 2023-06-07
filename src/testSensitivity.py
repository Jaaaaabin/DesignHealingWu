#
# testSensitivity.py
#

from const_project import DIRS_INI_RES, DIRS_DATA_SA, DIRS_DATA_SA_RES, DIRS_DATA_SA_FIG
from const_project import FILE_SA_PARAM_LIST, FILE_SA_VARY_SOBOL, FILE_SA_VARY_MORRIS
from const_ibcrule import BUILDING_RULES
from const_sensi import K_LEVEL_PARAMETER, SA_CALC_SECOND_ORDER, NAME_FLOOR, NAME_FAILURES

from funct_data import analyze_h5s, save_dict, load_dict
from funct_sensi import *

from Design import Design

def buildDesigns():

    # dictionary: design - > rule -> target -> distance & compliance.
    ini_dictCheckResult_h5s = analyze_h5s(DIRS_INI_RES, BUILDING_RULES)
    dictCheckResult_h5s = analyze_h5s(DIRS_DATA_SA_RES, BUILDING_RULES)

    # create the initial design: "DesignIni"
    DesignIni = Design(list(ini_dictCheckResult_h5s.keys())[0], BUILDING_RULES)
    sa_ini_parameter_names, sa_ini_parameter_values, sa_ini_parameter_num = collect_ini_sa_parameters(
        FILE_SA_PARAM_LIST, K_LEVEL_PARAMETER, set_floor=NAME_FLOOR)
    DesignIni.set_parameters({k:v for k,v in zip(sa_ini_parameter_names,sa_ini_parameter_values)})
    DesignIni.set_checkresults(ini_dictCheckResult_h5s[0])

    # create the new designs: "DesignsNew"
    DesignsNew  = [Design(nr, BUILDING_RULES) for nr in list(dictCheckResult_h5s.keys())]
    sa_new_parameter_names = pd.read_csv(FILE_SA_VARY_SOBOL, index_col=0, header=None).T.columns.tolist()
    sa_new_parameter_values_all = pd.read_csv(FILE_SA_VARY_SOBOL, index_col=0, header=None).T.values.tolist()
    for newDesign, sa_new_parameter_values in zip(DesignsNew, sa_new_parameter_values_all):
        newDesign.set_parameters({k:v for k,v in zip(sa_new_parameter_names, sa_new_parameter_values)})
        newDesign.set_checkresults(dictCheckResult_h5s[newDesign.number])

    # save the initial design and the new design.
    save_dict(DesignIni, DIRS_DATA_SA + r'\DesignIni.pickle')
    save_dict(DesignsNew, DIRS_DATA_SA + r'\DesignsNew.pickle')


def calculateIndex_sobol(sa_problem, tgt, rl, plot_index=False):

    result_y_txt = DIRS_DATA_SA_RES + '/results_y' + tgt + '_' + rl + '.txt'
    # result_index = DIRS_DATA_SA_RES + '/results_index' + tgt + '_' + rl + '.txt'..

    total, first, second = execute_sa_sobol(
        DIRS_DATA_SA_FIG,
        sa_problem,
        tgt,
        rl,
        result_y_txt,
        sa_calc_second_order=SA_CALC_SECOND_ORDER,
        plot_res=plot_index,
        plot_res_1_T=True,
        plot_res_2=True)

    all_indices = {
        'first': first,
        'second': second,
        'total':total
    }

    return all_indices


def testSensi_sobol(build_design=False, calc_index=False, plot_index=False):
    
    if build_design:
        buildDesigns()
    
    DesignIni = load_dict(DIRS_DATA_SA + r'\DesignIni.pickle')
    DesignsNew = load_dict(DIRS_DATA_SA + r'\DesignsNew.pickle')

    sa_problem = load_dict(DIRS_DATA_SA+"/sa_problem.pickle")

    sa_indices_all = dict()
    for tgt in DesignIni.failures.keys():

        if tgt in NAME_FAILURES:

            sa_indices_one = dict()
            for rl in DesignIni.failures[tgt]:
                tempo_result = [design.data[tgt][rl]['distance'] for design in DesignsNew]
                result_y_txt = DIRS_DATA_SA_RES + '/results_y' + tgt + '_' + rl + '.txt'
                np.savetxt(result_y_txt,tempo_result)
                
                # for every pair of target & rule.
                if calc_index:
                    tempo_indices = calculateIndex_sobol(sa_problem, tgt, rl, plot_index)
                    sa_indices_one.update({rl: tempo_indices})

            sa_indices_all.update({tgt: sa_indices_one})
        
    save_dict(sa_indices_all, DIRS_DATA_SA + r'\sa_sobol_indices.pickle')


def calculateIndex_morris(sa_problem, tgt, rl, plot_index=False):

    input_x_txt = DIRS_DATA_SA + '/sa_values_morris.txt'
    result_y_txt = DIRS_DATA_SA_RES + '/results_y' + tgt + '_' + rl + '.txt'

    all_indices = execute_sa_morris(
        DIRS_DATA_SA_FIG,
        sa_problem,
        tgt,
        rl,
        input_x_txt,
        result_y_txt)

    return all_indices


def testSensi_morris(build_design=False, calc_index=False, plot_index=False):
    
    if build_design:
        buildDesigns()
    
    DesignIni = load_dict(DIRS_DATA_SA + r'\DesignIni.pickle')
    DesignsNew = load_dict(DIRS_DATA_SA + r'\DesignsNew.pickle')

    sa_problem = load_dict(DIRS_DATA_SA+"/sa_problem.pickle")

    sa_indices_all = dict()
    for tgt in DesignIni.failures.keys():

        if tgt in NAME_FAILURES:

            sa_indices_one = dict()
            for rl in DesignIni.failures[tgt]:
                tempo_result = [design.data[tgt][rl]['distance'] for design in DesignsNew]
                result_y_txt = DIRS_DATA_SA_RES + '/results_y' + tgt + '_' + rl + '.txt'
                np.savetxt(result_y_txt,tempo_result)
                
                # for every pair of target & rule.
                if calc_index:
                    tempo_indices = calculateIndex_morris(sa_problem, tgt, rl, plot_index)
                    sa_indices_one.update({rl: tempo_indices})

            sa_indices_all.update({tgt: sa_indices_one})
    
    save_dict(sa_indices_all, DIRS_DATA_SA + r'\sa_morris_indices.pickle')



