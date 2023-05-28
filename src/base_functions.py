"""
This is the principal module of the healing project.
here you put your main functions.
"""

# import packages
from base_external_packages import *

# define base functions

#new
def create_directory(test_directory):
    """
    Create directories for tests

    """

    test_directory_dup = test_directory + r'\dups'
    test_directory_vary = test_directory + r'\vary'
    test_directory_res = test_directory + r'\res'
    test_directory_fig = test_directory + r'\fig'

    try:

        os.makedirs(test_directory)
        os.makedirs(test_directory_dup)
        os.makedirs(test_directory_vary)
        os.makedirs(test_directory_res)
        os.makedirs(test_directory_fig)

    except FileExistsError:
        print("Warning: the given data directory already exists: {}".format(test_directory))
        pass


def collect_bim_data(dirs_data, dirs_bim_res, nr_model=0):
    """
    Collect initial design information

    """

    input_file = os.path.join(dirs_bim_res, "inputdata_" + str(nr_model) + ".h5")
    output_file = os.path.join(dirs_bim_res, "outputdata_" + str(nr_model) + ".h5")
    
    if os.path.exists(input_file):
        shutil.copy2(input_file, dirs_data)
    else:
        print("There's no directory: \h",input_file)

    if os.path.exists(output_file):
        shutil.copy2(output_file, dirs_data)
    else:
        print("There's no directory: \h", output_file)


def get_data_from_h5(h5doc,datakey):
    """
    Collect the inputdata_"nr_model".h5

    """

    testData = pd.HDFStore(h5doc, 'r')
    allDataList = [item.replace('/', '') for item in testData.keys()]
    allDatapd = {k: pd.read_hdf(h5doc, k) for k in allDataList}
    data = allDatapd[datakey]
    return data



def analyze_h5_data(dirs_data, input_h5_key, output_h5_key, nr_model=0):
    """
    analyze the INITIAL data 

    :dirs_data:     directory of the copied data.
    :input_h5_key:  key of the input h5 file.
    :output_h5_key: key of the output h5 file.
    :nr_model:      nr_model = 0  for initial design

    """

    input = get_data_from_h5(dirs_data+'/inputdata_'+str(nr_model)+'.h5',input_h5_key)
    output = get_data_from_h5(dirs_data+'/outputdata_'+str(nr_model)+'.h5',output_h5_key)

    return input, output

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#st1_x	        st1_y	        st1_z
#st1_zo1_x	    st1_zo1_y
#st1_zo2_x	    st1_zo2_y
#st1_zo3_x	    st1_zo3_y
#st1_zo1_rm1_x
#st1_zo2_rm1_x	st1_zo2_rm2_x	st1_zo2_rm3_x
#st1_zo3_rm1_x	st1_zo3_rm2_x	st1_zo3_rm3_x	st1_zo3_rm4_x
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def parameter_grouping(parameter_list, strategy):
    """
    group the parameters according to different strategies
    
    :parameter_list:        the original list of parameters
    :strategy:              the strategy of grouping
        :"direction":       group the parameter by x, y, z direction
        :"hierarchy":       group the parameter by herarchical classes
        :"location":        group the parameter by locational distribution

    """

    def param_group_dirc(parameter_list):
        grp = dict.fromkeys(parameter_list)
        dirts = ["_z","_y","_x"]
        for param in parameter_list:
            if dirts[2] in param:
                grp[param]="group"+dirts[2]
            else:
                if dirts[1] in param:                
                    grp[param]="group"+dirts[1]
                else:
                    grp[param]="group"+dirts[0]
        return grp

    def param_group_hier(parameter_list):
        grp = dict.fromkeys(parameter_list)
        hiers = ["_st","_zo","_rm"]
        for param in parameter_list:
            if hiers[2] in param:
                grp[param]="group"+hiers[2]
            else:
                if hiers[1] in param:                
                    grp[param]="group"+hiers[1]
                else:
                    grp[param]="group"+hiers[0]
        return grp

    def param_group_loca(parameter_list):
        grp = dict.fromkeys(parameter_list)
        locts = ["_zo1","_zo2","_zo3"]
        for param in parameter_list:
            for key in locts:
                if key in param:
                    grp[param]="group"+key
            if grp[param] == None:
                grp[param]="group"+"_gb"
        return grp

    if strategy == "direction":
        parameter_groups = param_group_dirc(parameter_list)
    elif strategy == "hierarchy":
        parameter_groups = param_group_hier(parameter_list)
    elif strategy == "location":
        parameter_groups = param_group_loca(parameter_list)
    
    return parameter_groups

def parameter_freezing(parameter_list, freezers):
    """
    freeze the parameters according to key freezers
    :parameter_list:        the original list of parameters
    :freezers:              the key freezers

    """

    parameter_groups = dict.fromkeys(parameter_list)
    for param in parameter_list:
        for freezer in freezers:
            if freezer == param:
                parameter_groups[param] = "group"+"_frozen"
                break
        if parameter_groups[param] == None:
            parameter_groups[param]="group"+"_varying"
    
    return parameter_groups


def parameter_boundarying(parameter_list, strategy, values):
    """
    create variation boundary for all the parameters according to different strategies
    
    :parameter_list:        the original list of parameters
    :parameter_groups:      the parameter groups
    :strategy:              the strategy of setting boundaries
        :"percentage":      use relative percentage values for max-min boundaries
        :"absolute":        use relative absolute values for max-min boundaries

    """

    def percentage_boundarying(parameter_list, values):
        parameter_bdrys = dict.fromkeys(parameter_list)
        for param in parameter_bdrys:
            parameter_bdrys[param] = [-0.01*values*parameter_list[param],0.01*values*parameter_list[param]]
        return parameter_bdrys

    def absolute_boundarying(parameter_list, values):
        parameter_bdrys = dict.fromkeys(parameter_list)
        for param in parameter_bdrys:
            parameter_bdrys[param] = [-values,values]
        return parameter_bdrys

    if strategy == "percentage":
        parameter_boundaries = percentage_boundarying(parameter_list, values)
    elif strategy == "absolute":
        parameter_boundaries = absolute_boundarying(parameter_list, values)

    return parameter_boundaries


def arrange_bim_files(dirs_data, dirs_bim, dirs_bim_res, nr_model_max=0, collect_h5=True, clear_rvt=False, clear_h5=False):

    if collect_h5:
        if nr_model_max==0:
            collect_bim_data(dirs_data, dirs_bim_res)
        elif nr_model_max>0:
            for nr in range(nr_model_max+1):
                collect_bim_data(dirs_data, dirs_bim_res, nr)

    if clear_rvt:
        rvt_clear_list = os.listdir(dirs_bim)
        for item in rvt_clear_list:
            if item.endswith(".rvt") and item.startswith("test"):
                os.remove(os.path.join(dirs_bim, item))

    if clear_h5:
        h5_clear_list = os.listdir(dirs_bim_res)
        for item in h5_clear_list:
            if item.endswith("_0.h5"):
                continue
            else:
                os.remove(os.path.join(dirs_bim_res, item))


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

# def build_rule_results(dirs_res, samples, target_rules):
#     """
#     build rule results per checking rule.

#     """
    
#     rule_results_df = pd.DataFrame(data=None, columns=target_rules)
#     for rule in target_rules:
#         rule_results_df[rule] = [sample.compliance_distance[rule]
#                          for sample in samples]
#         np.savetxt(dirs_res + "/results_" + str(rule) + ".txt", rule_results_df[rule])
#     return rule_results_df

#new.

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