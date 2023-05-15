#
# funct_region.py
#


# import packages
from base_external_packages import *

# import modules
from base_classes import Design
from base_functions import *

# define region related functions

# calculate the outlier.
def distance_from_center(clusterer, X, cluster_labels):

    # calculate the distance from the cluster center.
    distance = []
    for ii in range(X.shape[0]):
        center =  clusterer.cluster_centers_[cluster_labels[ii]]
        center =  center[:-1]
        absolute_distance = np.absolute(np.array(center) - np.array(X[ii]))
        final_distance = np.sqrt(np.sum(absolute_distance**2))
        distance.append(final_distance)
    return np.round(distance, 4)


def find_outlier(cluster_distance, n_outlier=0):

    # find certain number of outliers
    distance = list(cluster_distance)
    sorted_idx = sorted(range(len(distance)), key=lambda k: distance[k], reverse=True)
    if n_outlier != 0:
        outlier_idx = sorted_idx[:n_outlier]
    else:
        outlier_idx = sorted_idx
    return outlier_idx


def remove_outlier(ini_idx, all_outliers_idx):
    
    # find all outlier_idx that exit in the ini_idx
    outliers_idx = [i for i in all_outliers_idx if i in ini_idx]

    # refine the ini_idx by outliers_idx
    for idx in outliers_idx:
        ini_idx.remove(idx)
    return ini_idx, outliers_idx


def KMeans_clusterings(
    dirs_fig,
    rl,
    X,
    y,
    X_pca,
    n_clus,):
    """
    plot the clusterings in PCs.
    
    """

    # combine X, X_pca data with y.
    X_y = np.concatenate((X, y.T),axis=1) # add y to the last column of X.
    X_pca_y = np.concatenate((X_pca, y.T),axis=1) # only for plotting

    # execute KMeans clustering.
    clusterer = KMeans(n_clusters=n_clus, init='random')

    # determine the cluster labels and distance (to the corresponding center) for each sample.
    cluster_labels = clusterer.fit_predict(X_y)
    cluster_distance = distance_from_center(clusterer, X, cluster_labels)
    outliers_idx = find_outlier(cluster_distance, n_outlier=20)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)
    ax1.set_xlim([-0.3, 1])
    ax1.set_ylim([0, len(X_pca) + (n_clus + 1) * 10])

    # The silhouette_score gives the average value for all the samples.
    silhouette_avg = silhouette_score(X, cluster_labels)
    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    # 1st plotting (left)
    ax1.set_xlabel("The silhouette coefficient values", color="black", fontsize=10)
    ax1.set_ylabel("Cluster labels (regarding {})".format(rl), color="black", fontsize=10)
    ax1.set_title("The silhouette plot for the identified clusters")

    # - - - - - - - - - - - - - - - - - - - - 
    # silhouette plotting
    y_lower = 10
    for i in range(n_clus):

        # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(i) / n_clus)
        
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )
        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples
    
    # - - - - - - - - - - - - - - - - - - - - 
    # 2nd plotting in PCA (right)
    ax2.set_title("The visualization of the clustered data in PCs")
    ax2.set_xlabel("PC1", color="black", fontsize=10)
    ax2.set_ylabel("PC2", color="black", fontsize=10)
    
    # set colors for clusters
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clus)

    # split according to valid and invalid indices.
    indices_invalid = [idx for idx, val in enumerate(list(X_pca_y)) if val[2] == 0]
    indices_valid = [idx for idx, val in enumerate(list(X_pca_y)) if val[2] == 1]
    
    # remove outliers 
    indices_invalid, indices_invalid_outlier = remove_outlier(indices_invalid, outliers_idx)
    indices_valid, indices_valid_outlier = remove_outlier(indices_valid, outliers_idx)
    
    # Clusters: each color represents a refined cluster (outliers removed).
    bad_X_pca_y_colors  = colors[indices_invalid]
    good_X_pca_y_colors = colors[indices_valid]

    bad_X_pca_y = X_pca_y[indices_invalid]
    good_X_pca_y = X_pca_y[indices_valid]    

    # Outliers: set a uniform color for outliers.
    c_outlier, c_outlier_edge = 'white', 'black'
    outlier_bad_X_pca_y = X_pca_y[indices_invalid_outlier]
    outlier_good_X_pca_y = X_pca_y[indices_valid_outlier]

    # plot bad clusters
    ax2.scatter(
        bad_X_pca_y[:,0], bad_X_pca_y[:,1],
        c=bad_X_pca_y_colors,
        marker=".", lw=0, s=20, alpha=0.75,
        label='clusters of invalid designs',
        )
    
    # plot good clusters
    ax2.scatter(
        good_X_pca_y[:,0], good_X_pca_y[:,1],
        c=good_X_pca_y_colors,
        marker="+", lw=1.5, s=20, alpha=0.75,
        label='clusters of valid designs',
        )# good points
    
    # plot outliers (invalid: ".") & (valid: "+")
    ax2.scatter(
        outlier_bad_X_pca_y[:,0], outlier_bad_X_pca_y[:,1],
        c=c_outlier, edgecolors=c_outlier_edge,
        marker="o", lw=0.5, s=20,
        label='outliers (invalid)',
        )
    ax2.scatter(
        outlier_good_X_pca_y[:,0], outlier_good_X_pca_y[:,1],
        c=c_outlier, edgecolors=c_outlier_edge,
        marker="s", lw=0.5, s=20,
        label='outliers (valid)',
        )

    # Labeling the clusters
    centers = np.empty(shape=[1,2])
    for nn in range(n_clus):
        indices_byclus = [idx for idx, label in enumerate(list(cluster_labels)) if label == nn]
        X_pca_byclus = X_pca[indices_byclus]
        X_pca_byclus_mean = np.mean([X_pca_byclus], axis=1)
        centers = np.concatenate((centers, X_pca_byclus_mean),axis=0)
    centers = centers[1:,:]

    # Draw white circles at cluster centers
    ax2.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="o",
        c="white",
        alpha=1,
        s=60,
        edgecolor="k")
    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=0.95, s=30, edgecolor="k")
    
    # Draw the position of the initial design
    ax2.plot([], [], ' ', label="colors represent different clusters")
    ax2.hlines(
        y=X_pca[0,1], xmin=-0.50, xmax=0.50,
        color='grey', linestyles='dashed', linewidths=0.5,
        label='locating lines of initial design')
    ax2.vlines(
        x=X_pca[0,0], ymin=-0.50, ymax=0.50,
        color='grey', linestyles='dashed', linewidths=0.5)
    ax2.legend(loc='upper right', fontsize ='small')

    # - - - - - - - - - - - - - - - - - - - - 
    # Save the picture
    plt.savefig(dirs_fig + "/Clusters_KMeans_" + rl + "_" + str(n_clus), dpi=200)
    
    return cluster_labels


def build_feasibility_regions(
    dirs_fig,
    target_rules,
    bim_parameters,
    pca_data_X_Y,
    pca_labels,
    n_clus=6,):
    """
    build feasible and infeasible regions for
    passed(p) and failed(f) samples respectively with a threshold of threshold_p and threshold_f for each rule.
    by K-means clustering.

    """

    # map the string 'valid/invalid' back to 1/0.
    def map_2_bool(v_string):
        return 1 if v_string == 'valid' else 0

    # create initial output file
    pca_data_X_Y_with_regions = copy.deepcopy(pca_data_X_Y)
    
    # create X and Y for region clustering.
    # cluster data in raw parameter values x, the clusters are directly connected to parameter values not PCs.
    cols_4_x = [bim_parameters[elem] for elem in list(range(len(bim_parameters)))]  # consider all the parameters' value by default.
    X_df = copy.deepcopy(pca_data_X_Y_with_regions[cols_4_x])
    
    # prepare y data per rule.
    cols_4_y = target_rules
    Y_df = copy.deepcopy(pca_data_X_Y_with_regions[cols_4_y])
    
    # use PCs to visualize the results in 2D.
    cols_4_plot = [pca_labels[elem] for elem in list(map(str, list(range(int(2)))))] # take PC1 and PC2 by default.
    data_in_pca_df = copy.deepcopy(pca_data_X_Y_with_regions[cols_4_plot])

    # for each single rule / the sum of all rule
    for rl in Y_df.columns.values:

        # prepare data for the K-means Clustering.
        y = Y_df[rl].values
        y = np.array([[map_2_bool(yy) for yy in y]]) # map the label back to 0 / 1 and add it to the x data X
        X = X_df.values
        X_pca = data_in_pca_df.values

        # Keep the system setting for KMeans memory leak on Windows.
        os.environ["OMP_NUM_THREADS"] = '12'
        np.random.seed(999)
        
        # Get the cluster labels.
        cluster_labels = KMeans_clusterings(
            dirs_fig,
            rl,
            X,
            y,
            X_pca,
            n_clus,)

        # pca_data_X_Y_with_region.columns: create region label columns with default value 'NAN'
        pca_data_X_Y_with_regions[rl+'_cluster'] = cluster_labels
    
    return pca_data_X_Y_with_regions


def detect_feasible_regions(
    target_rules,
    df,
    n_clus=2):
    """
    
    """
    
    regions_per_rule = dict.fromkeys(target_rules)
    for rl in target_rules:
        df_val = df.loc[df[rl] == 'valid']
        nr_clusters = df_val[rl+'_cluster'].value_counts()[:n_clus].index.tolist()
        regions_per_rule[rl] = nr_clusters
    return regions_per_rule


def extract_feasible_data(
    df,
    feasible_regions,
    target_rules,
    include_ini=True,):
    """
    
    """
    
    samples_with_regions_df = copy.deepcopy(df)
    feasible_data = dict.fromkeys(target_rules)
    
    # replace the clusters with preliminary_feasible_region_#
    for rl in target_rules:
        regions_per_rl = feasible_regions[rl]

        for ii in range(len(regions_per_rl)):
            
            samples_with_regions_df.loc[
                samples_with_regions_df[rl+'_cluster'] == regions_per_rl[ii],
                rl+'_cluster'] = f'preliminary_feasible_region_{(ii+1)}'
    
    # save feasible region data per rule
    for rl in target_rules:    
        tempo_all = pd.DataFrame(columns = samples_with_regions_df.columns.values)
        
        # add the initial design first. 
        if include_ini:
            tempo_ini =  samples_with_regions_df.loc[
                    samples_with_regions_df['type'] == 'initial']  
            tempo_all = pd.concat([tempo_all, tempo_ini], axis=0)

        for ii in range(len(regions_per_rl)):
            
            tempo_single = samples_with_regions_df.loc[
                samples_with_regions_df[rl+'_cluster'] == f'preliminary_feasible_region_{(ii+1)}']
            tempo_all = pd.concat([tempo_all, tempo_single], axis=0)

        feasible_data[rl] = tempo_all

    return samples_with_regions_df, feasible_data


def get_approach_parameters(df, tol_val_std = 1.0e-12):
    """
    
    """
    
    all_parameters = df.std().index.tolist()
    approach_params = [
        col if (df[col].std() > tol_val_std) and ('PC' not in col)
        else None for col in all_parameters]
    approach_params = [i for i in approach_params if i is not None]
    return approach_params


def sweeping_sampling(init_v, target_vs, sweep_density=10,):
    """
    :init_v:        [0,                             n_parameter]
    :target_vs:     [number of feasible samples,    n_parameter]

    return
    samples         [number of feasible samples * sweep_density, n_parameter]

    """

    def random_evenly_sampling_vm2vn(vm, vn, amount):

        # keep changing the generation results from "p.random.uniform"
        random.seed(100) 
        random_factors = np.random.uniform(low=0.1, high=0.80, size=amount)
        samples =  np.array([vm + random_factor*(vn-vm) for random_factor in random_factors])
        return samples

    all_samples = np.empty(shape=[target_vs.shape[0]*sweep_density,target_vs.shape[1]])

    for ii in range(target_vs.shape[0]):

        samples = np.empty(shape=[sweep_density,target_vs.shape[1]])
        for jj in range (target_vs.shape[1]):
            inter_samples = random_evenly_sampling_vm2vn(init_v[jj], target_vs[ii,jj], sweep_density)
            samples[:,jj] = inter_samples

        all_samples[ii*sweep_density:(ii+1)*sweep_density,:] = samples

    return all_samples


def build_sweeping_values(
    feasible_data_df,
    approach_params,
    sweep_density=5,):
    """

    """

    v_init = feasible_data_df.loc[0,approach_params].values
    v_targets = feasible_data_df.loc[1:,approach_params].values
    values_fill_region = np.array([]).reshape(0,v_init.shape[0])

    # sweeping between valid designs within the preliminary feasible region.
    for i in range(v_targets.shape[0]-1):
        v_targets_init = v_targets[i]
        v_targets_targ = v_targets[i+1:]
        values_tempo = sweeping_sampling(v_targets_init, v_targets_targ, sweep_density=int(sweep_density))
        values_fill_region = np.concatenate([values_fill_region, values_tempo], axis=0)
    
    # sweeping from the initial design to valid designs in the preliminary feasible region.
    values_fill_gap = sweeping_sampling(v_init, values_fill_region, sweep_density=(sweep_density*2))

    # add both.
    samples_values = np.concatenate([values_fill_region, values_fill_gap], axis=0)
    
    return samples_values
    

def build_sweeping_samples(
    ref_design,
    sweeping_values,
    approach_params):
    """

    """

    sweeping_values_df = pd.DataFrame(columns = approach_params)
    for i, param in enumerate(approach_params):
        sweeping_values_df[param] = sweeping_values[:,i].tolist()

    # create a list(samples) for all samples.
    # the initial design instance are also added here.
    samples = [ref_design]
    
    # build a DataFrame for all samples
    df_samples = pd.DataFrame(ref_design.parameters, index=[ref_design.number])


    nr_model = ref_design.number
    for nr_model in range(nr_model+1,len(sweeping_values_df)+nr_model+1,1):
        
        new_design = Design(nr_model)
        new_design.number = nr_model
        new_design.parameters = copy.deepcopy(ref_design.parameters)
        for param in approach_params:
            new_design.parameters[param] = sweeping_values_df.loc[nr_model-1, param]
        samples.append(new_design)
        
        df_tempo = pd.DataFrame(new_design.parameters, index=[new_design.number])
        df_samples = pd.concat([df_samples,df_tempo])

    return samples, df_samples

# discard

# def plot_clusterings(
#     X, labels,
#     n_clus,
#     title,
#     alpha=0.9):
#     """
#     plot the clusterings in PCs.
    
#     """

#     fig = plt.figure(figsize=(12, 8))  # unit of inch
#     ax = plt.axes((0.1, 0.1, 0.85, 0.85))  # in range (0,1)

#     # set marker types according to valid/invalid
#     map_markers = {
#         0: '1',     # 0 for invalid samples
#         1: ','}     # 1 for valid samples
#     X_markers = [map_markers[xx[2]] for xx in X]
    
#     # set marker colors according to Nr. cluster 
#     np.random.seed(999)
#     map_colors = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
#          for i in range(n_clus)]
#     X_colors = [map_colors[label] for label in labels]

#     for ii, (item, txt) in enumerate(zip(X,labels)):
#         plt.scatter(item[0], item[1], c=X_colors[ii], marker=X_markers[ii], s=20, alpha=alpha)
#         ax.annotate(txt, (item[0]+0.01, item[1]+0.01), c=X_colors[ii], fontsize='xx-small')
    
#     # to investigate
#     # plt.legend(loc='best') 
#     plt.savefig(title, dpi=100)


# def build_feasibility_regions(
#     dirs_fig,
#     target_rules,
#     bim_parameters,
#     pca_data_X_Y,
#     pca_labels,
#     n_clus=2,
#     linkage="ward",):
#     """
#     build feasible and infeasible regions for
#     passed(p) and failed(f) samples respectively with a threshold of threshold_p and threshold_f for each rule.
#     by AgglomerativeClustering.

#     """
    
#     # map the string 'valid/invalid' back to 1/0.
#     def map_2_bool(v_string):
#         return 1 if v_string == 'valid' else 0

#     # create initial output file
#     pca_data_X_Y_with_regions = copy.deepcopy(pca_data_X_Y)
    
#     # create X and Y for region clustering.
#     # cluster data in raw parameter values x, the clusters are directly connected to parameter values not PCs.
#     cols_4_x = [bim_parameters[elem] for elem in list(range(len(bim_parameters)))]  # consider all the parameters' value by default.
#     X_df = copy.deepcopy(pca_data_X_Y_with_regions[cols_4_x])
#     # prepare y data per rule.
#     cols_4_y = target_rules
#     Y_df = copy.deepcopy(pca_data_X_Y_with_regions[cols_4_y])
    
#     # use PCs to plot the results in "2" D
#     cols_4_plot = [pca_labels[elem] for elem in list(map(str, list(range(int(2)))))] # take PC1 and PC2 by default.
#     pca_df = copy.deepcopy(pca_data_X_Y_with_regions[cols_4_plot])

#     # for each single rule / the sum of all rule
#     for rl in Y_df.columns.values:

#         y = Y_df[rl].values
#         y = np.array([[map_2_bool(yy) for yy in y]]) # map the label back to 0 / 1 and add it to the x data X
#         X = X_df.values
#         X = np.concatenate((X, y.T),axis=1) # add y to the last column of X.
#         plot_pca = pca_df.values
#         plot_pca = np.concatenate((plot_pca, y.T),axis=1) # only for plotting

#         # the AgglomerativeClustering: to play with parameters
#         clustering = AgglomerativeClustering(
#             linkage=linkage,
#             n_clusters=n_clus, # try with more
#             compute_distances=True,
#             )
            
#         cluster_labels  = clustering.fit_predict(X)
#         plot_clusterings(
#             plot_pca, cluster_labels,
#             n_clus,
#             dirs_fig + "/Clusters_" + rl + "_" + linkage + "_" + str(n_clus))
    
#         # pca_data_X_Y_with_region.columns: create region label columns with default value 'NAN'
#         pca_data_X_Y_with_regions[rl+'_cluster'] = cluster_labels
    
#     # passed first then failed
    
#     return pca_data_X_Y_with_regions


