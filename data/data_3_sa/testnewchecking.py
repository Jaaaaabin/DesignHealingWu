import pandas as pd

FOLDER_CHECK_RES = r'C:\dev\phd\jw\healing\data\healing2023\40_sensitivity_analysis\res'
    

def get_data_from_h5(h5doc, key):
    """
    Collect data from .h5 file by specifying the store key.
    """

    allData = pd.HDFStore(h5doc, 'r')
    data = allData[key]
    return data

### this following script is for the detected elements and also for the failure percent.
# to dig 20230511.
 
 
 # Based on external information of the failures.
failuresIBC1020_2 = get_data_from_h5(FILE_CHECK_RES, 'IBC1020_2')
failuresIBC1207_1 = get_data_from_h5(FILE_CHECK_RES, 'IBC1207_1')
failuresIBC1207_3 = get_data_from_h5(FILE_CHECK_RES, 'IBC1207_3')
dictFailures = {
    'IBC1020_2': list(failuresIBC1020_2.loc[failuresIBC1020_2['checkCompliance'] == False, 'spaceIfcGUID'].iloc[:]),
    'IBC1207_1': list(failuresIBC1207_1.loc[failuresIBC1207_1['checkCompliance'] == False, 'spaceIfcGUID'].iloc[:]),
    'IBC1207_3': list(failuresIBC1207_3.loc[failuresIBC1207_3['checkCompliance'] == False, 'spaceIfcGUID'].iloc[:]),
}
# write to csv for the failure information.
dfInitialFailures = pd.DataFrame(dict(
    [(k,pd.Series(v)) for k,v in dictFailures.items()]))
dfInitialFailures.to_csv(DICT_ANALYSIS_RES+'\df_InitialFailures.csv')