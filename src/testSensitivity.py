#
# testSensitivity.py
#

from Design import Design

from const_project import DIRS_INI_RES, FILE_SA_PARAM_LIST, DIRS_DATA_TOPO
from const_sensi import FILE_SA_VARY_SOBOL, FILE_SA_VARY_MORRIS, DIRS_DATA_SA, DIRS_DATA_SA_RES, DIRS_DATA_SA_FIG
from const_sensi import SA_CALC_SECOND_ORDER, DIRS_DATA_SA, N_LEVEL_MORRIS
from const_sensi import K_LEVEL_PARAMETER, NAME_FLOOR
from const_ibcrule import BUILDING_RULES

from funct_data import analyze_h5s, save_dict, load_dict
from funct_sensi import *


def buildDesigns(
    file_variation,
    newdesigns_in_path = [],
    newdesigns_out_path = [],
    inidesign_in_path = DIRS_INI_RES,
    build_ini=True,
    build_new=True):

    if build_ini:

        # dictionary: design - > rule -> target -> distance & compliance.
        ini_dictCheckResult_h5s = analyze_h5s(inidesign_in_path, BUILDING_RULES)

        # create the initial design: "DesignIni"
        DesignIni = Design(list(ini_dictCheckResult_h5s.keys())[0], BUILDING_RULES)
        ini_parameter_names, ini_parameter_values, ini_parameter_num = collect_ini_sa_parameters(
            FILE_SA_PARAM_LIST, K_LEVEL_PARAMETER, set_floor=NAME_FLOOR)
        DesignIni.set_parameters({k:v for k,v in zip(ini_parameter_names,ini_parameter_values)})
        DesignIni.set_checkresults(ini_dictCheckResult_h5s[0])
    
        # save the initial design
        save_dict(DesignIni, newdesigns_out_path + r'\DesignIni.pickle')   

    if build_new:
        
        # dictionary: design - > rule -> target -> distance & compliance.
        dictCheckResult_h5s = analyze_h5s(newdesigns_in_path, BUILDING_RULES)

        # create the new designs: "DesignsNew"
        DesignsNew  = [Design(nr, BUILDING_RULES) for nr in list(dictCheckResult_h5s.keys())]

        new_parameter_names = pd.read_csv(file_variation, index_col=0, header=None).T.columns.tolist()
        new_parameter_values_all = pd.read_csv(file_variation, index_col=0, header=None).T.values.tolist()
        for newDesign, new_parameter_values in zip(DesignsNew, new_parameter_values_all):
            newDesign.set_parameters({k:v for k,v in zip(new_parameter_names, new_parameter_values)})
            newDesign.set_checkresults(dictCheckResult_h5s[newDesign.number])

        # save the new design.
        save_dict(DesignsNew, newdesigns_out_path + r'\DesignsNew.pickle')


def calIndex_sobol(sa_problem, rl, plot_index=False):

    result_y_txt = DIRS_DATA_SA_RES + '/results_y_' + rl + '.txt'

    total, first, second = execute_sa_sobol(
        DIRS_DATA_SA_FIG,
        sa_problem,
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


# def testSensi_sobol(build_design=False, calc_index=False, plot_index=False):
    
#     if build_design:
#         buildDesigns(FILE_SA_VARY_SOBOL, DIRS_DATA_SA_RES, DIRS_DATA_SA)

#     DesignIni = load_dict(DIRS_DATA_SA + r'\DesignIni.pickle')
#     DesignsNew = load_dict(DIRS_DATA_SA + r'\DesignsNew.pickle')

#     sa_problem = load_dict(DIRS_DATA_SA+"/sa_problem.pickle")

#     sa_indices_all = dict()
#     for tgt in DesignIni.failures.keys():

#         if tgt in NAME_FAILURES:

#             sa_indices_one = dict()
#             for rl in DesignIni.failures[tgt]:
#                 tempo_result = [design.data[tgt][rl]['distance'] for design in DesignsNew]
#                 result_y_txt = DIRS_DATA_SA_RES + '/results_y_' + tgt + '_' + rl + '.txt'
#                 np.savetxt(result_y_txt,tempo_result)
                
#                 # for every pair of target & rule.
#                 if calc_index:
#                     tempo_indices = calIndex_sobol(sa_problem, tgt, rl, plot_index)
#                     sa_indices_one.update({rl: tempo_indices})

#             sa_indices_all.update({tgt: sa_indices_one})
        
#     save_dict(sa_indices_all, DIRS_DATA_SA + r'\sa_sobol_indices.pickle')


def calIndex_morris(sa_problem, rl, input_x_txt, result_y_txt, plot_index=False):

    all_indices = execute_sa_morris(
        DIRS_DATA_SA_FIG,
        sa_problem,
        rl,
        input_x_txt,
        result_y_txt,
        N_LEVEL_MORRIS,
        plot_index,
        )

    return all_indices


# def testSensi_morris(build_design=False, calc_index=False, plot_index=False):
    
#     if build_design:
#         buildDesigns(FILE_SA_VARY_MORRIS, DIRS_DATA_SA_RES, DIRS_DATA_SA)
    
#     DesignIni = load_dict(DIRS_DATA_SA + r'\DesignIni.pickle')
#     DesignsNew = load_dict(DIRS_DATA_SA + r'\DesignsNew.pickle')

#     sa_problem = load_dict(DIRS_DATA_SA+"/sa_problem.pickle")

#     sa_indices_all = dict()
#     for tgt in DesignIni.failures.keys():

#         if tgt in NAME_FAILURES:

#             sa_indices_one = dict()
#             for rl in DesignIni.failures[tgt]:
#                 tempo_result = [design.data[tgt][rl]['distance'] for design in DesignsNew]
#                 result_y_txt = DIRS_DATA_SA_RES + '/results_y_' + tgt + '_' + rl + '.txt'
#                 np.savetxt(result_y_txt,tempo_result)
                
#                 # for every pair of target & rule.
#                 if calc_index:
#                     tempo_indices = calIndex_morris(sa_problem, tgt, rl, plot_index)
#                     sa_indices_one.update({rl: tempo_indices})

#             sa_indices_all.update({tgt: sa_indices_one})
    
#     save_dict(sa_indices_all, DIRS_DATA_SA + r'\sa_morris_indices.pickle')


#new---------------------------------------------------------------------------------
def testSensi_sobol_weighted(build_design=False, calc_index=False, plot_index=False):
    
    if build_design:
        buildDesigns(FILE_SA_VARY_SOBOL, DIRS_DATA_SA_RES, DIRS_DATA_SA)
    
    DesignIni = load_dict(DIRS_DATA_SA + r'\DesignIni.pickle')
    DesignsNew = load_dict(DIRS_DATA_SA + r'\DesignsNew.pickle')

    sa_problem = load_dict(DIRS_DATA_SA+"/sa_problem.pickle")

    sa_indices_all = dict()

    for rl in DesignIni.rules:
        
        tuned_y_per_rule = []

        for new_design in DesignsNew:
            
            tuned_y_per_design = 0

            for tgt in DesignIni.results[rl].keys():
                
                if DesignIni.results[rl][tgt]['distance']>=0 and new_design.results[rl][tgt]['distance']>=0:
                    new_design.results[rl][tgt]['distance'] *= 0 # ignore the always positive part.
                elif DesignIni.results[rl][tgt]['distance'] * new_design.results[rl][tgt]['distance'] < 0:
                    new_design.results[rl][tgt]['distance'] *= 1 # 
                    
                tuned_y_per_design += new_design.results[rl][tgt]['distance']

            tuned_y_per_rule.append(tuned_y_per_design)

        result_y_txt = DIRS_DATA_SA_RES + '/results_y_' + rl + '.txt'
        np.savetxt(result_y_txt, tuned_y_per_rule)

        if calc_index:
            tempo_indices = calIndex_sobol(sa_problem, rl, plot_index)
            sa_indices_all.update({rl: tempo_indices})

    save_dict(sa_indices_all, DIRS_DATA_SA + r'\sa_sobol_indices.pickle')


def testSensi_morris_weighted(build_design=False, calc_index=False, plot_index=False, beta_coef_reduction=1):
    
    if build_design:
        buildDesigns(FILE_SA_VARY_MORRIS, DIRS_DATA_SA_RES, DIRS_DATA_SA)
    
    DesignIni = load_dict(DIRS_DATA_SA + r'\DesignIni.pickle')
    DesignsNew = load_dict(DIRS_DATA_SA + r'\DesignsNew.pickle')

    sa_problem = load_dict(DIRS_DATA_SA+"/sa_problem.pickle")

    sa_indices_all = dict()
    
    filtered_ids = load_dict(DIRS_DATA_TOPO + "/filtered_id.pickle")

    tuned_y_all_rules = []
    tuned_comp_all_rules = []

    for rl in DesignIni.rules:
        
        tuned_y_per_rule = []
        tuned_comp_per_rule = []

        for new_design in DesignsNew:
            
            tuned_y_per_design = 0
            tuned_comp_per_design = 1

            for tgt in DesignIni.results[rl].keys():
                
                # only for the specific floor.
                if tgt in filtered_ids:
                    
                    # tune the overall y value.
                    # ini True & new True.
                    if DesignIni.results[rl][tgt]['distance'] >= 0 and new_design.results[rl][tgt]['distance'] >= 0:
                        tuned_y = (new_design.results[rl][tgt]['distance']) * (beta_coef_reduction)

                    else:
                        tuned_y = (new_design.results[rl][tgt]['distance']) * 1

                    # # ini True & new False. or  ini False & new True. 
                    # elif DesignIni.results[rl][tgt]['distance'] * new_design.results[rl][tgt]['distance'] < 0:
                    #     tuned_y = (new_design.results[rl][tgt]['distance']) * 1

                    # # ini False & new False.
                    # elif DesignIni.results[rl][tgt]['distance'] < 0 and new_design.results[rl][tgt]['distance'] < 0:
                    #     tuned_y = (new_design.results[rl][tgt]['distance']) * (beta_coef_reduction)

                    tuned_y_per_design += tuned_y

                    # summarize the compliance results.
                    if new_design.results[rl][tgt]['compliance'] == True:
                        continue
                    else:
                        tuned_comp_per_design = 0

                else:
                    # not for other floors.
                    continue
            
            # sum for each rule. so each tgt is counted as equal weight.
            tuned_y_per_rule.append(tuned_y_per_design)
            tuned_comp_per_rule.append(tuned_comp_per_design) # compliance results:
        
        # the overall y value.
        input_x_txt = DIRS_DATA_SA + '/sa_values_morris.txt'
        result_y_txt = DIRS_DATA_SA_RES + '/results_y_' + rl + '_beta_' + str(beta_coef_reduction) + '.txt'
        np.savetxt(result_y_txt, tuned_y_per_rule)

        # the overall compliance results.
        result_comp_txt = DIRS_DATA_SA_RES + '/results_compliance_' + rl + '.txt'
        np.savetxt(result_comp_txt, tuned_comp_per_rule)

        # calculate the sensitivity indices per rules.
        if calc_index:
            tempo_indices = calIndex_morris(sa_problem, rl, input_x_txt, result_y_txt, plot_index)
            sa_indices_all.update({rl: tempo_indices})

        # append to all rules
        tuned_y_all_rules.append(tuned_y_per_rule)
        tuned_comp_all_rules.append(tuned_comp_per_rule)
    
    # calculated for all rules
    rl = 'IBC_all'
    tuned_y_all_rules_T = pd.DataFrame(tuned_y_all_rules).T.values.tolist()
    tuned_y_all_rules_T = [sum(sublist)/len(sublist) for sublist in tuned_y_all_rules_T]
    result_y_txt = DIRS_DATA_SA_RES + '/results_y_' + rl + '_beta_' + str(beta_coef_reduction) + '.txt'
    np.savetxt(result_y_txt, tuned_y_all_rules_T)
    
    tuned_comp_all_rules_T = pd.DataFrame(tuned_comp_all_rules).T.values.tolist()
    tuned_comp_all_rules_T = [int(all(sublist)) for sublist in tuned_comp_all_rules_T]
    result_comp_txt = DIRS_DATA_SA_RES + '/results_compliance_' + rl + '.txt'
    np.savetxt(result_comp_txt, tuned_comp_all_rules_T)

    save_dict(sa_indices_all, DIRS_DATA_SA + r'\sa_morris_indices_beta_' + str(beta_coef_reduction) + '.pickle')