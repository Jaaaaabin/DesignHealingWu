#
# funct_sensi.py
#

# import packages
from base_external_packages import *


def flatten(list):
    return [item for sublist in list for item in sublist]


def save_ndarray_2txt(x, filename):
    """
    save multidimensional array into a txt file

    """

    np.savetxt(filename, x)


def load_ndarray_txt(filename):
    """
    reload the multidimensional array 

    """

    data = np.loadtxt(filename)
    return data


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


def create_directory(test_directory):
    """
    Create directories for tests

    """

    test_directory_archive = test_directory + r'\archive'
    test_directory_dup = test_directory + r'\dups'
    test_directory_fig = test_directory + r'\fig'
    test_directory_res = test_directory + r'\res'
    test_directory_vary = test_directory + r'\vary'

    try:

        os.makedirs(test_directory)
        os.makedirs(test_directory_archive)
        os.makedirs(test_directory_dup)
        os.makedirs(test_directory_vary)
        os.makedirs(test_directory_res)
        os.makedirs(test_directory_fig)

    except FileExistsError:

        print("Warning: the given data directory already exists: {}".format(test_directory))
        pass


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


def checkSampleDuplication(file_csv):
    """
    check if there's any repetative samples (rows in a Dataframe.)
    """

    test_dup = pd.read_csv(file_csv, index_col=0, header=None).T
    test_df_dup = test_dup.duplicated()

    return test_df_dup.value_counts()


def save_samples(dirs_data, data_file_name, samples, save_sample_object=False):
    """
    save the samples to a .dat
    
    """

    PIK = os.path.join(dirs_data, data_file_name + ".dat")

    if save_sample_object:
        with open(PIK, "wb") as f:
            pickle.dump(len(samples), f, protocol=pickle.HIGHEST_PROTOCOL)
            for sample in samples:
                pickle.dump(sample, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_samples(dirs_data, data_file_name, load_sample_object=False):
    """
    laod samples from to a .dat

    """

    PIK = os.path.join(dirs_data, data_file_name + ".dat")
    samples = []

    if load_sample_object:
        with open(PIK, "rb") as f:
            for _ in range(pickle.load(f)):
                samples.append(pickle.load(f))

    return samples


def is_dup_simple(arr):
    """
    find if there're duplicated rows in a 2D array and return the counts if any.
    
    """

    u, c = np.unique(arr, axis=0, return_counts=True)
    return (c>1).any(), c

# def map_global_y(elems, set_result_label_type='validity'):
    
#     v1,v2 = None, None

#     if set_result_label_type == 'to_1_0':
#         v1,v2 = 1, 0
#     elif set_result_label_type == 'to_bool':
#         v1,v2 = True, False
#     elif set_result_label_type == 'validity':
#         v1,v2 = 'valid','invalid'
#     else:
#         print("Summarizing error. Please choose correct Summarizing set_result_label_type!")

#     if all(elem == v1 for elem in elems):
#         return v1
#     else:
#         return v2