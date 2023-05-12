"""
This is the principal module of the healing project.
here you put your main functions.
"""

# import packages
from base_external_packages import *

# import modules
from base_functions import *


# define principal component analysis functions

def merge_project_samples(pj_names):
    """
    merge the samples in different vairation together for pca.

    """

    initial_design, all_sa_samples_df, all_rule_results_df = [],[],[]
    
    for pj_name in pj_names:

        tt_name = 'tests//' + pj_name

        # access the required directories.
        dirs_data, dirs_fig, dirs_res = tt_name+"/data", tt_name+"/fig", tt_name+"/res" #use existing ones

        # load the x and y (from single test) 
        sa_samples_df = load_dict(os.path.join(dirs_res, pj_name + "_x.pickle"))
        rule_results_df = load_dict(os.path.join(dirs_res, pj_name + "_y.pickle"))

        if pj_name == pj_names[0]:
            
            # collect data and the initial design from the frist iteration
            initial_design = load_dict(dirs_res+"/initial_design.obj")
            all_sa_samples_df = sa_samples_df
            all_rule_results_df = rule_results_df
        
        else:
            
            # later iteration(s)
            tempo_sa_samples_df =  pd.concat([all_sa_samples_df,sa_samples_df.iloc[1:]],axis=0,ignore_index=True)
            tempo_rule_results_df = pd.concat([all_rule_results_df,rule_results_df.iloc[1:]],axis=0,ignore_index=True)
            all_sa_samples_df = tempo_sa_samples_df
            all_rule_results_df = tempo_rule_results_df
            
    return initial_design, all_sa_samples_df, all_rule_results_df


def build_pca_data(target_rules, samples_df, results_df, dim, set_result_label_type='validity'):
    """
    prepare X and Y for PCA.
    
    """
    
    # prepare the X and Y for PCA
    # *** here might need to drop irrelevant parameters ***
    # X data
    data_for_x = samples_df

    # Y data
    results_df = results_df[target_rules]
    data_for_y = results_df.applymap(lambda x: map_label_y(x, set_result_label_type=set_result_label_type))

    # add a global compliance among all considered rules.
    data_for_y['IBC_selected'] = data_for_y.apply(lambda row: map_global_y(row.values.tolist(), set_result_label_type=set_result_label_type), axis=1)
 
    # execute the PCA for the initial X data
    X = samples_df.to_numpy()
    pca = PCA(n_components=dim)
    pca_components = pca.fit_transform(X)
    pca_loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    
    # create labels for the pc s
    labels = {
        str(i): f"PC {i+1} ({var:.1f}%)"
        for i, var in enumerate(pca.explained_variance_ratio_ * 100)
    }
    
    # collect all data into one df.
    pca_data_X_Y = data_for_x # raw x

    for label in labels:
        col_nb = int(label)
        pc_name = labels[str(label)]
        pca_s = pd.Series(pca_components[:,col_nb], name=pc_name)
        pca_data_X_Y = pd.concat([pca_data_X_Y,pca_s], axis=1) # x in pc
    
    pca_data_X_Y = pd.concat([pca_data_X_Y, data_for_y], axis=1) # raw y

    # label the initial design.
    type_s = pd.Series(np.array(['initial']+['adjusted']*(X.shape[0]-1)), name='type')
    pca_data_X_Y = pd.concat([pca_data_X_Y,type_s], axis=1) # model type (initial or adjusted)

    return pca_components, pca_loadings, pca_data_X_Y, labels