import pandas as pd
import shutil
import csv
import os


def duplicateRVT(dir_ini, dir_dest, amount=0, clear_destination=True):
    """
    duplicate the initial .rvt file.
    """

    # clear all the previous *.rvt files the destination folder.
    if clear_destination:
        for f in os.listdir(dir_dest):
            if f.endswith(".rvt"):
                os.remove(os.path.join(dir_dest, f))
    if amount > 0 :
        nbs =  [item for item in range(1, amount+1)]
        pathnames = [dir_dest+'\\'+ str(nb) for nb in nbs]
        for pathname in pathnames:
            if os.path.isfile(dir_ini):
                shutil.copy(dir_ini, pathname+'.rvt')
    else:
        return 'Amount Error'

def getRVTFilename(file_dir, outpath, remove_ext = True):
    """
    write the .rvt files into a csv for controling.
    """

    # list to store files
    res = []

    # Iterate directory
    for path in os.listdir(file_dir):
        # check if current path is a file
        if os.path.isfile(os.path.join(file_dir, path)):
            if path.endswith('.rvt'):
                res.append(path)
    
    if remove_ext:
        res = [x.replace('.rvt', '') for x in res]

    df_rvtids = pd.DataFrame(res)
    df_rvtids.to_csv(outpath, header=False, index=False)
    print('Extraction of duplicated RVT filenames (with student IDs) Succeed.')

PATH_DUP = r'C:\dev\phd\jw\healing\data\healing2023\40_sensitivity_analysis\dups'
PATH_INI_RVT = r'C:\dev\phd\jw\healing\data\healing2023\ini-parametrized-testE1.rvt'
FILE_VARY_CSV = r'C:\dev\phd\jw\healing\data\healing2023\ini_gps_varytest.csv'

duplicateRVT(PATH_INI_RVT, PATH_DUP, amount=15, clear_destination=True) # duplicate the .rvt