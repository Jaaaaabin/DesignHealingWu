import pandas as pd
import csv

FILE_CHECK_RES = r'C:\dev\phd\jw\healing\data\healing2023\00_auto_check\res\0.h5'
DICT_REVIT_RES = r'C:\dev\phd\jw\healing\data\healing2023\20_extract_data\res'
DICT_ANALYSIS_RES = r'C:\dev\phd\jw\healing\data\healing2023\30_analyze_model\res'

IBC_RULES = ['IBC1020_2','IBC1207_1','IBC1207_3']

FILE_LIST_GP = DICT_REVIT_RES + r'\df_Parameters.csv'
FILE_RELATED_GP = DICT_ANALYSIS_RES + r'\df_AssociatedGPs.csv'
FILE_RELATED_GP_INI  =  DICT_ANALYSIS_RES + r'\ini_gps.csv'

def createDictGlobalParametersPerRule(rules, df_gps):
    
    dictGPperRule = dict()
    for rule in rules:
        lst_gps_per_rule = [gp for gp in df_gps[rule].tolist() if str(gp) != 'nan']
        dictGPperRule[rule] =  lst_gps_per_rule

    return dictGPperRule

# import Global Parameter related data.
df_gps = pd.read_csv(FILE_RELATED_GP, index_col = 0)                # list of related GPs.
df_gp_values = pd.read_csv(FILE_LIST_GP, index_col = 'name')        # values of all GPs.
df_gp_values = df_gp_values['value']

dictGPValues = df_gp_values.to_dict()
dictGPperRule = createDictGlobalParametersPerRule(IBC_RULES, df_gps)

all_gps = list(set([y for x in list(dictGPperRule.values()) for y in x])) 
all_gps_vals  = [round(dictGPValues[gp],3) for gp in all_gps]

with open(FILE_RELATED_GP_INI, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(zip(all_gps, all_gps_vals))