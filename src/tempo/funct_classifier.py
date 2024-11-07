#
# funct_region.py
#


# import packages
from base_external_packages import *

# import modules
from Design import Design
from funct_data import *

# define region related functions

# determine the outliers
def find_outlier_by_distance(clusterer, X, cluster_labels, sort_type='cluster', p_outlier=0):
    """
    sort_type = 'all': find the outliers by absolute longest distance among samples in all clusters.
    sort_type = 'cluster': find the outliers by longest distance within each cluster.
    p_outlier in (0,0.25] positive percent values.
    """

    # calculate the distance from the cluster center.
    all_distance = []
    cls_distance = [[] for _ in range(clusterer.n_clusters)]
    cls_distance_idx = [[] for _ in range(clusterer.n_clusters)]

    for ii in range(X.shape[0]):
        center =  clusterer.cluster_centers_[cluster_labels[ii]]
        center =  center[:-1]
        absolute_distance = np.absolute(np.array(center) - np.array(X)[ii])
        final_distance = np.sqrt(np.sum(absolute_distance**2))

        # write the distances per cluster.
        cls_distance[cluster_labels[ii]].append(final_distance)
        cls_distance_idx[cluster_labels[ii]].append(ii)

        # write all the distances together.
        all_distance.append(final_distance)

    all_distance = list(np.round(all_distance, 4))
    
    outlier_idx = []
    # by absolute longest distance among samples in all clusters. + outlier < 25%
    if sort_type =='all' and 0 < p_outlier:
        
        n_outlier = int(p_outlier*len(all_distance))
        sorted_idx = sorted(range(len(all_distance)), key=lambda k: all_distance[k], reverse=True)
        outlier_idx = sorted_idx[:n_outlier]
    
    # by longest distance within each cluster. + p_outlier < 25%
    elif sort_type =='cluster' and 0 < p_outlier:

        for cls_dist , cls_dist_idx in zip(cls_distance, cls_distance_idx):

            n_outlier = int(p_outlier*len(cls_dist))
            cls_sorted_idx = sorted(range(len(cls_dist)), key=lambda k: cls_dist[k], reverse=True)
            cls_sorted_idx = cls_sorted_idx[:n_outlier]
            sorted_idx = [cls_dist_idx[id] for id in cls_sorted_idx]
            outlier_idx.append(sorted_idx)
            
        outlier_idx = flatten(outlier_idx)

    return outlier_idx
    

def remove_outlier(ini_idx, all_outliers_idx):

    if all_outliers_idx!=[]:
        # find all outlier_idx that exit in the ini_idx
        outliers_idx = [i for i in all_outliers_idx if i in ini_idx]

        # refine the ini_idx by outliers_idx
        for idx in outliers_idx:
            ini_idx.remove(idx)

    else:
        outliers_idx = []

    return ini_idx, outliers_idx


def build_pca_data(data_for_x, data_for_y, dim,):
    """
    prepare X and Y for PCA.
    
    """
    # execute the PCA for the initial X data
    X = data_for_x.to_numpy()
    pca = PCA(n_components=dim)
    pca_components = pca.fit_transform(X)
    pca_loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    # create labels for all the PCs
    labels = {
        str(i): f"PC {i+1} ({var:.1f}%)"
        for i, var in enumerate(pca.explained_variance_ratio_ * 100)
    }
    
    # collect all data into one df.
    pca_data_X = data_for_x # raw x
    pca_data_Y = pd.Series(data_for_y, name='compliance')

    for label in labels:
        col_nb = int(label)
        pc_name = labels[str(label)]
        pca_s = pd.Series(pca_components[:,col_nb], name=pc_name)
        pca_data_X = pd.concat([pca_data_X, pca_s], axis=1) # x in pc
    
    pca_data_X_Y = pd.concat([pca_data_X, pca_data_Y], axis=1) # raw y

    return pca_data_X_Y


def KMeans_clusterings(
    dirs_fig,
    X,
    y,
    p_outlier,
    outlier_sort_type,
    n_clus,
    n_pca=2,
    ):

    """
    execute the KMeans-Clustering.
    plot the clusterings in PCs.
    """

    # pca in 2D.
    pca_data_X_Y = build_pca_data(X, y, dim=n_pca)

    # initial design
    clns_pca = [cl for cl in pca_data_X_Y.columns.values.tolist() if 'PC' in cl]
    pca_x_y = pca_data_X_Y[clns_pca].to_numpy()

    # real Data (real values)
    clns_X_y = [cl for cl in pca_data_X_Y.columns.values.tolist() if not 'PC' in cl]
    X_y = pca_data_X_Y[clns_X_y].to_numpy()
    
    # plot Data (PCA values)
    clns_pca_y = [cl for cl in pca_data_X_Y.columns.values.tolist() if 'PC' in cl or 'compliance' in cl]
    X_pca_y = pca_data_X_Y[clns_pca_y].to_numpy()

    # execute KMeans clustering.
    clusterer = KMeans(n_clusters=n_clus, init='random')

    # determine the cluster labels and distance (to the corresponding center) for each sample.
    cluster_labels = clusterer.fit_predict(X_y)
    outliers_idx = find_outlier_by_distance(clusterer, X, cluster_labels, sort_type=outlier_sort_type, p_outlier=p_outlier)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)
    ax1.set_xlim([-0.3, 1])
    ax1.set_ylim([0, len(X_pca_y) + (n_clus + 1 ) * 10])

    # The silhouette_score gives the average value for all the samples.
    silhouette_avg = silhouette_score(X, cluster_labels)
    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    # 1st plotting (left)
    ax1.set_xlabel("The silhouette coefficient values", color="black", fontsize=10)
    ax1.set_ylabel("Cluster labels ", color="black", fontsize=10)
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
    indices_invalid = [idx for idx, val in enumerate(X_pca_y.tolist()) if val[-1] == 0]
    indices_valid = [idx for idx, val in enumerate(X_pca_y.tolist()) if val[-1] == 1]
    
    # remove outliers 
    indices_invalid, indices_invalid_outlier = remove_outlier(indices_invalid, outliers_idx)
    indices_valid, indices_valid_outlier = remove_outlier(indices_valid, outliers_idx)
    
    # Clusters: each color represents a refined cluster (outliers removed).
    bad_X_pca_y_colors  = colors[indices_invalid]
    good_X_pca_y_colors = colors[indices_valid]

    bad_X_pca_y = X_pca_y[indices_invalid]
    good_X_pca_y = X_pca_y[indices_valid]    

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
    
    if outliers_idx!=[]:
        
        # Outliers: set a uniform color for outliers.
        c_outlier, c_outlier_edge = 'white', 'black'
        outlier_bad_X_pca_y = X_pca_y[indices_invalid_outlier]
        outlier_good_X_pca_y = X_pca_y[indices_valid_outlier]

        # plot outliers (invalid: ".") & (valid: "+")
        ax2.scatter(
            outlier_bad_X_pca_y[:,0], outlier_bad_X_pca_y[:,1],
            c=c_outlier, edgecolors=c_outlier_edge,
            marker="o", lw=0.25, s=8,
            label='outliers (invalid)',
            )
        ax2.scatter(
            outlier_good_X_pca_y[:,0], outlier_good_X_pca_y[:,1],
            c=c_outlier, edgecolors=c_outlier_edge,
            marker="s", lw=0.25, s=8,
            label='outliers (valid)',
            )

    # Labeling the clusters
    centers = []
    for nn in range(n_clus):
        indices_byclus = [idx for idx, label in enumerate(list(cluster_labels)) if label == nn]
        X_pca_byclus = X_pca_y[indices_byclus][:-1]
        X_pca_byclus_mean = np.mean(X_pca_byclus, axis=0)
        centers.append(X_pca_byclus_mean)
    centers = np.array(centers)

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
    
    # # Draw the position of the initial design
    pc1_min, pc2_min = pca_x_y.min(axis=0)
    pc1_max, pc2_max = pca_x_y.max(axis=0)
    
    ax2.plot([], [], ' ', label="colors represent different clusters")
    ax2.vlines(
        x=pca_x_y[0,0], ymin=pc2_min, ymax=pc2_max,
        color='grey', linestyles='dashed', linewidths=0.5)
    ax2.hlines(
        y=pca_x_y[0,1], xmin=pc1_min, xmax=pc1_max,
        color='grey', linestyles='dashed', linewidths=0.5,
        label='locating lines of initial design')
    
    ax2.legend(loc='upper right', fontsize ='small')

    # - - - - - - - - - - - - - - - - - - - - 
    # Save the picture
    plt.savefig(dirs_fig + "\KMeansClustering_{}_{}.png".format(n_clus, p_outlier), dpi=200)
    
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


