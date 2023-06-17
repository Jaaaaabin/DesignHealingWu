#
# funct_sensi.py
#

# import packages
from base_external_packages import *


def flatten(list):
    return [item for sublist in list for item in sublist]


def get_problems_from_paths(
    paths, file_end='.txt', file_start = 'results_y_', file_sep = '_'):

    problems = []

    for file in os.listdir(paths):
        if file.endswith(file_end):
            problems.append(os.path.join(file))

    problems = [txt.replace(file_start,'') for txt in problems]
    problems = [txt.replace(file_end,'') for txt in problems]
    problems = [txt.split(file_sep, 1) for txt in problems]

    return problems

def sortStrListbyNumber(lst):
    """
    sort a list by number value inside.
    """
    
    sort_lst = natsorted(lst)
    return sort_lst


def get_data_from_h5(h5doc, key):
    """
    collect data from .h5 file by specifying the store key.
    """

    allData = pd.HDFStore(h5doc, 'r')
    data = allData[key]
    return data


def get_h5_from_directory(directory):
    """
    collect the list of .h5 from a directory.
    """

    full_paths = []
    local_paths = []
    for path in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, path)):
            if path.endswith('.h5'):
                full_paths.append(os.path.join(directory, path))
                local_paths.append(path)
    full_paths = sortStrListbyNumber(full_paths)
    local_paths = sortStrListbyNumber(local_paths)
    return full_paths, local_paths


def analyze_h5s(directory, rules):
    """
    analyze a group of .h5 documents containing code compliance checking results  
    """

    files_h5, names_h5 = get_h5_from_directory(directory)

    dictCheckResult_h5s = dict()
    for file_h5, name_h5 in zip(files_h5, names_h5):

        dictCheckResult_rules = dict()
        for rule in rules:
            
            # per rule.
            tempo = get_data_from_h5(file_h5, rule)
            tempo = tempo.set_index('spaceIfcGUID') #to improve

            dictCheckResult_targets = dict()
            for idx in tempo.index:
                
                # per target (of checking).
                dictCheckResult_target = dict()
                dictCheckResult_target.update({'distance': tempo.loc[idx, 'healDistanceScaled']}) #to improve 
                dictCheckResult_target.update({'compliance': tempo.loc[idx, 'checkCompliance']}) #to improve
                dictCheckResult_targets.update({idx: dictCheckResult_target})
            dictCheckResult_rules.update({rule: dictCheckResult_targets})

        name_h5_as_number = int(name_h5.replace(".h5",""))
        dictCheckResult_h5s.update({name_h5_as_number: dictCheckResult_rules})

    return dictCheckResult_h5s


def save_dict(dt, filename):
    """
    save dictionary / object

    """
    
    with open(filename, 'wb') as handle:
        pickle.dump(dt, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_dict(filename):
    """
    reload dictionary / object
    
    """
    
    with open(filename, 'rb') as handle:
        dt = pickle.load(handle)
    return dt