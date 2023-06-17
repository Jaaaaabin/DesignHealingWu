"""
This is the principal module of the healing project.
here you put your main functions.
"""

# import packages
from base_external_packages import *

# define base functions

# new
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

# old
def map_label_y(value, set_result_label_type='validity'):
    """
    mapping set_result_label_types:
    :'to_1_0':          distance to 1 or 0.
    :'to_bool':         distance to True or False.
    :'validity':        distance to string.
    
    """

    if set_result_label_type == 'to_1_0':
        return 1 if value >= 0 else 0
    elif set_result_label_type == 'to_bool':
        return True if value >= 0 else False
    elif set_result_label_type == 'validity':
        return 'valid' if value >= 0 else 'invalid'        
    else:
        print("Mapping error. Please choose correct mapping set_result_label_type!")


def map_global_y(elems, set_result_label_type='validity'):
    
    v1,v2 = None, None

    if set_result_label_type == 'to_1_0':
        v1,v2 = 1, 0
    elif set_result_label_type == 'to_bool':
        v1,v2 = True, False
    elif set_result_label_type == 'validity':
        v1,v2 = 'valid','invalid'
    else:
        print("Summarizing error. Please choose correct Summarizing set_result_label_type!")

    if all(elem == v1 for elem in elems):
        return v1
    else:
        return v2


def build_variation_txt_csv(dirs_bim, project_name, txt_filename, df):
    """
    build a .csv for design variations that will be employed for BIM.

    """

    # create a csv from df
    csv_file = os.path.join(dirs_bim, project_name + ".csv")

    # remove the first row (initial design)
    df_for_csv = df.tail(-1)
    df_for_csv.T.to_csv(csv_file, header=False)
    
    # create an file (*.txt) for instruction
    doc_variation = os.path.join(dirs_bim, txt_filename)
    text_file = open(doc_variation, "wt")
    n = text_file.write(csv_file)
    text_file.close()



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


def var_display(dirs_res, project_name):
    """
    write all constant variable values.
    
    """

    with open(dirs_res + '/log_'+project_name+'.txt', 'w') as file:
        for name, value in globals().copy().items():
            file.write("\n")
            file.write(f"{str(name):<30}{str(value):<50}")
            
    return dirs_res + '/log_'+project_name+'.txt'