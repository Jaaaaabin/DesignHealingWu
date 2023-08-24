#
# formSpace.py
#

# import modules

from base_external_packages import *

from Space import SolutionSpace

# from Design import Design

from const_project import DIRS_DATA, FILE_INIT_SKL_RVT, DIRS_DATA_TOPO
from const_solus import DIRS_DATA_SS, DIRS_DATA_SS_DUP, DIRS_DATA_SS_RES, FILE_SS_VARY_LHS, DIRS_DATA_SS_FIG
from const_solus import ITERATION_VALUES, FILE_SS_VARY_SKEWNORMAL

from funct_data import save_dict, load_dict, get_problems_from_paths, flatten, create_directory, duplicateRVT
from funct_svm import displaySVCinPC, displaySVCin3PC

from testSensitivity import buildDesigns


def buildDesignInSpace(file_variation):
    buildDesigns(
        file_variation = file_variation,
        newdesigns_in_path=DIRS_DATA_SS_RES,
        newdesigns_out_path=DIRS_DATA_SS,
        build_ini=False,
        build_new=True)
    

def exploreLHS(
    dataset=[],
    num_samples=200,
    explore_range=0.3,
    lhs_optimization='random-cd',
    set_dup_rvt=False,
    set_new_space=False):
    
    if set_new_space:
        create_directory(DIRS_DATA_SS)

    pathIni = DIRS_DATA + dataset + r'\DesignIni.pickle'
    pathRes = DIRS_DATA + dataset + r'\res'
    problems =  get_problems_from_paths(pathRes)

    # form the initial Space.
    designIni = load_dict(pathIni)
    initialSpace = SolutionSpace(problems)
    initialSpace.___setcenter__(designIni)

    # add sa results.
    indicesSA = load_dict(DIRS_DATA + dataset + r'\sa_morris_indices_mu.pickle')
    indicesSA = {k:v[0] for k,v in indicesSA.items()}

    initialSpace.enrich_sensitivity(indicesSA,'IBC_all')

    initialSpace.explore_space_by_lhs(
        num_samples=num_samples,
        explore_range=explore_range,
        lhs_optimization=lhs_optimization,
        random_seed=521,
    )

    initialSpace.samples_by_lhs.to_csv(FILE_SS_VARY_LHS, header=False)

    if set_dup_rvt:      
        # duplicat the .rvts for variation.
        duplicateRVT(FILE_INIT_SKL_RVT, DIRS_DATA_SS_DUP, amount=initialSpace.samples_by_lhs.shape[1], clear_destination=True)


def formSolutionSpace(
    dataset=[],
    set_new_space=False):

    if set_new_space:
        create_directory(DIRS_DATA_SS)

    filtered_ids = load_dict(DIRS_DATA_TOPO + "/filtered_id.pickle")

    pathIni = DIRS_DATA + dataset[0] + r'\DesignIni.pickle'
    pathRes = DIRS_DATA + dataset[0] + r'\res'
    problems =  get_problems_from_paths(pathRes)

    designIni = load_dict(pathIni)

    pathsNew = [DIRS_DATA + set + r'\DesignsNew.pickle' for set in dataset[1:]]
    designsNew = flatten([load_dict(path) for path in pathsNew])

    # form the initial Space.
    initialSpace = SolutionSpace(problems)
    initialSpace.___setguids__(filtered_ids)
    initialSpace.___setcenter__(designIni)

    initialSpace.form_space(designIni, designsNew)
    initialSpace.__buildxy__(dir=DIRS_DATA_SS, build_valid_subset=True)

    # save outcome Space.
    sourceSpace = '_'.join([data.replace('\\','') for data in dataset])
    save_dict(initialSpace, DIRS_DATA_SS + r'\Space_' + sourceSpace + r'.pickle')


def reasonSolutionSpace(
    dataset=[],
    transfer_space=False,
    inter_level=0,
    plot_space_pairwise=False,
    plot_space_svm=False,
    plot_knc=False,
    random_seed=521,
    ):

    # load the space
    sourceSpace = '_'.join([data.replace('\\','') for data in dataset])
    file_solutionspace = DIRS_DATA_SS + r'\Space_' + sourceSpace + r'.pickle'
    
    currentSpace = load_dict(file_solutionspace)
    currentSpace.__addini__()
    currentSpace.remove_constant()

    if transfer_space:
        currentSpace.transfer_space_new()
        versionSpace = 'TSpace'
    else:
        versionSpace = 'Space'

    # extract X values.
    X = currentSpace.data_X_df
    
    # extract Y values.
    compliance_keys = list(currentSpace.data_Y_dict.keys())
    Y = currentSpace.data_Y_dict
    
    if plot_space_svm:
        
        # X: X_svm -> X_scaled -> X_scaled_reduced. 
        X_svm = copy.deepcopy(X)

        # pca = PCA(n_components=2)
        # X_pca = pca.fit_transform(X_svm)

        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_svm)
        
        for key in compliance_keys:
        
            # Y: y.
            y = Y[key]

            # split for train and test.
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=random_seed)

            # The standard approach is to use t-SNE to reduce the dimensionality of the data for visualization purposes.
            # Once you have reduced the data to two dimensions you can easily replicate the visualization in the scikit-learn tutorial
            # define the meshgrid

            # for svm_kernel in ["linear", "poly", "rbf" ,"sigmoid"]:
            v_gamma = 1

            # build and fit the svm.
            svm_classifer = svm.SVC(kernel="linear", C=2, random_state=random_seed)
            # svm_classifer = svm.SVC(kernel="rbf",gamma=v_gamma,random_state=random_seed)
            svm_classifer.fit(X_train, y_train)

            y_pred_test = svm_classifer.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred_test)
            print ("accuracy for ", key, "= ", accuracy)

            displaySVCinPC(X_scaled, y, path = DIRS_DATA_SS_FIG, rule_label = key)

            # print('w = ',svm_classifer.coef_)
            # print('b = ',svm_classifer.intercept_)
            # print('Indices of support vectors = ', svm_classifer.support_)
            # print('Support vectors = ', svm_classifer.support_vectors_)
            # print('Number of support vectors for each class = ', svm_classifer.n_support_)
            # print('Coefficients of the support vector in the decision function = ', np.abs(svm_classifer.dual_coef_))
            
            # displaySVCin3PC(X_scaled, y, path = DIRS_DATA_SS_FIG, rule_label = key)
    
    # plot
    if plot_space_pairwise:

        # INPUT Data distribution.
        # https://www.kaggle.com/code/sunaysawant/iris-pair-plot-palettes-all-170-visualisations

        for label_compliance in list(currentSpace.valid_idx.keys()):

            spaceVariable_label = list(X.columns)
            spaceVariable_label.append(label_compliance)
            df_data = currentSpace.data_X_Y[spaceVariable_label]

            # add the initial design. (x and y)
            # ini_plot = [currentSpace.ini_parameters[variable] for variable in spaceVariable_label]  # adding a row
            # ini_plot.append('Initial Design')

            # add the initial design and its label.
            # df_data.loc[-1] = ini_plot 
            # df_data.index = df_data.index + 1
            # df_data = df_data.sort_index()

            # sort the plot orders
            df_data = df_data.replace(True,'Valid')
            df_data = df_data.replace(False,'Invalid')
            df_data['sort'] = df_data[label_compliance].apply(lambda x: {'Initial Design': 2, 'Valid':1, 'Invalid':0}[x])
            df_sort = df_data.sort_values(by=['sort'])
            df_plot = df_sort.loc[:, df_sort.columns != 'sort']

            # plot
            fig = plt.figure(figsize=(10, 10))
            palette_3d = np.array([
                sns.color_palette("seismic")[-1],       # Invalid
                sns.color_palette("seismic")[0],        # Valid
                sns.color_palette("rocket")[0],         # Initial Design
                ])
    
            g = sns.pairplot(
                df_plot,
                hue=label_compliance,
                markers=["o", "s", "D"],
                plot_kws=dict(s=4, edgecolor="white", alpha=0.35),
                palette=palette_3d,
                corner=False,)

            for ax in g.axes.ravel():
                
                if (ax is None):

                    continue
                
                else:

                    param_x = ax.get_xlabel()
                    param_y = ax.get_ylabel()

                    if param_x and param_y:

                        param_x_v = df_plot[param_x].iloc[-1]
                        param_y_v = df_plot[param_y].iloc[-1]
                        ax.axvline(x=param_x_v, ls='-.', linewidth=0.5, alpha=0.80, c='black')
                        ax.axhline(y=param_y_v, ls='-.', linewidth=0.5, alpha=0.80, c='black', label='Initial Values')
                        g._update_legend_data(ax)

                        # pad = 0.01
                        # xmin = X[param_x].min() - pad
                        # xmax = X[param_x].max() + pad
                        # ymin = X[param_y].min() - pad
                        # ymax = X[param_y].max() + pad

                        xmin, xmax = ax.get_xlim()
                        ymin, ymax = ax.get_ylim()

                        total_line_segs = 50
                        lseg = int(total_line_segs/2)

                        x = np.linspace(xmin, xmax, total_line_segs)
                        y = np.linspace(ymin, ymax, total_line_segs)

                        shift = 0.05


                        # ---------------------------------
                        # 1020_2
                        txt_color = 'navy'
                        txt_fontsize = 6

                        # txt_bbox = dict(facecolor='none', edgecolor=txt_color, alpha=0.5)

                        if '1020_2' in label_compliance or label_compliance == 'compliance':
                            
                            ax_label = '$ {ew}_{6} ≤ {ew}_{35}-6.983 $'
                            
                            if param_x == 'U1_OK_d_wl_ew35' and param_y == 'U1_OK_d_wl_ew6':
                            
                                y_u = x - 6.983
                                ax.plot(x, y_u, ls='-', linewidth=1.0, alpha=0.95, c = txt_color, label="Analytical Boundaries")
                                ax.text(
                                    x[lseg], y_u[lseg] + 2*shift, ax_label, fontsize=txt_fontsize, c = txt_color, ha='center', va='center',
                                    rotation=35)
                                
                                # ax.fill_between(x, y_l, y_u, where=y_l<y_u, color='darkorange', alpha=0.5)
                                # legend for "Analytical boundaries"
                                g._update_legend_data(ax)

                        # ---------------------------------
                        # 1207_1
                        if '1207_1' in label_compliance or label_compliance == 'compliance':
                            
                            ax_label = '$ {sn}_{26} ≤ 8.861 $'

                            if param_x == 'U1_OK_d_wl_sn26' and (param_y == 'U1_OK_d_wl_ew6' or param_y == 'U1_OK_d_wl_sn21'):
                                x_l = y * 0 + 8.861
                                ax.plot(x_l, y, ls='-', linewidth=1.0, alpha=0.95, c = txt_color, label="Analytical Boundaries")
                                ax.text(
                                    x_l[lseg] - shift, y[lseg], ax_label, fontsize=txt_fontsize, c = txt_color, ha='center', va='center',
                                    rotation=90,)
                            if param_y == 'U1_OK_d_wl_sn26' and (param_x == 'U1_OK_d_wl_sn10' or param_x == 'U1_OK_d_wl_ew35'):
                                y_u = x * 0 + 8.861
                                ax.plot(x, y_u, ls='-', linewidth=1.0, alpha=0.95, c = txt_color, label="Analytical Boundaries")
                                ax.text(
                                    x[lseg], y_u[lseg] + shift, ax_label, fontsize=txt_fontsize, c = txt_color, ha='center', va='center',
                                    rotation=0,)

                            ax_label = '$ {sn}_{21} ≥ 2.309 $'

                            if param_y == 'U1_OK_d_wl_sn21' and param_x !='U1_OK_d_wl_ew6':
                                y_l = x * 0 + 2.309
                                ax.plot(x, y_l, ls='-', linewidth=1.0, alpha=0.95, c = txt_color, label="Analytical Boundaries")
                                ax.text(
                                    x[lseg], y_l[lseg] + shift, ax_label, fontsize=txt_fontsize, c = txt_color, ha='center', va='center',
                                    rotation=0,)
                            if param_x == 'U1_OK_d_wl_sn21' and param_y == 'U1_OK_d_wl_ew6':
                                x_u = y *0 + + 2.309
                                ax.plot(x_u, y, ls='-', linewidth=1.0, alpha=0.95, c = txt_color, label="Analytical Boundaries")
                                ax.text(
                                    x_u[lseg] - shift, y[lseg], ax_label, fontsize=txt_fontsize, c = txt_color, ha='center', va='center',
                                    rotation=90,)

                            ax_label = '$ {ew}_{6} ≥ 2.309 $'
                            
                            if param_y == 'U1_OK_d_wl_ew6':
                                y_u = x * 0 + 2.309
                                ax.plot(x, y_u, ls='-', linewidth=1.0, alpha=0.95, c = txt_color, label="Analytical Boundaries")
                                ax.text(
                                    x[lseg], y_u[lseg] + shift, ax_label, fontsize=txt_fontsize, c = txt_color, ha='center', va='center',
                                    rotation=0,)

                            ax_label = '$ {sn}_{26} ≥ {sn}_{10} + 2.234 $'

                            if param_x == 'U1_OK_d_wl_sn10' and param_y == 'U1_OK_d_wl_sn26':
                                y_l = x + 2.234
                                ax.plot(x, y_l, ls='-', linewidth=1.0, alpha=0.95, c = txt_color, label="Analytical Boundaries")
                                ax.text(
                                    x[lseg], y_l[lseg] - 3*shift, ax_label, fontsize=txt_fontsize, c = txt_color, ha='center', va='center',
                                    rotation=30,)

                            ax_label = '$ {sn}_{21} ≤ {sn}_{10} - 2.234 $'

                            if param_x == 'U1_OK_d_wl_sn10' and param_y == 'U1_OK_d_wl_sn21':
                                y_u = x - 2.234
                                ax.plot(x, y_u, ls='-', linewidth=1.0, alpha=0.95, c = txt_color, label="Analytical Boundaries")
                                ax.text(
                                    x[lseg], y_u[lseg] + 3*shift, ax_label, fontsize=txt_fontsize, c = txt_color, ha='center', va='center',
                                    rotation=35,)
                                
                                # legend for "Analytical Boundaries"
                                g._update_legend_data(ax)

                        # # ---------------------------------
                        # 1207_3
                        if '1207_3' in label_compliance or label_compliance == 'compliance':

                            ax_label = '$ (10.995 - {sn}_{26}) * ({ew}_{6} - 0.175) ≥ 6.5 $'

                            if param_y == 'U1_OK_d_wl_ew6' and param_x == 'U1_OK_d_wl_sn26':
                                y_l = 6.5 / (10.995-x) + 0.175
                                ax.plot(x, y_l, ls='-', linewidth=1.0, alpha=0.95, c = txt_color, label="Analytical Boundaries")
                                ax.text(
                                    x[lseg], y_l[lseg] - 2*shift, ax_label, fontsize=txt_fontsize, c = txt_color, ha='center', va='center',
                                    rotation=35,)

                            ax_label = '$ ({sn}_{21} - 0.125) * ({ew}_{6} - 0.175) ≥ 6.5 $'

                            if param_y == 'U1_OK_d_wl_ew6' and param_x == 'U1_OK_d_wl_sn21':
                                y_l = 6.5 / (x-0.125) + 0.175
                                ax.plot(x, y_l, ls='-', linewidth=1.0, alpha=0.95, c = txt_color, label="Analytical Boundaries")
                                ax.text(
                                    x[lseg], y_l[lseg] - shift, ax_label, fontsize=txt_fontsize, c = txt_color, ha='center', va='center',
                                    rotation=-35,)
                                
                                # legend for "Analytical Boundaries"
                                g._update_legend_data(ax)

                            txt_color = 'darkgreen'
                            txt_fontsize = 5

                            for ew6 in [2.309, 3]:
                                
                                ax_label = '$ ({sn}_{26} - {sn}_{10} - 0.125) * ({ew}_{6} - 0.175) ≥ 6.5, {ew}_{6}= $' + str(ew6)

                                if param_y == 'U1_OK_d_wl_sn26' and param_x == 'U1_OK_d_wl_sn10':
                                    
                                    y_l = x + 0.1 + 6.5 /(ew6 - 0.175)
                                    ax.plot(x, y_l, ls='--', linewidth=0.5, alpha=0.75, c=txt_color, label="Analytical Boundaries (dynamic)")
                                    ax.text(
                                        x[lseg], y_l[lseg] + 2*shift, ax_label, fontsize=txt_fontsize, c=txt_color, ha='center', va='center',
                                        rotation=0,)

                                ax_label = '$ ({sn}_{10} - {sn}_{21} - 0.125)  * ({ew}_{6} - 0.175) ≥ 6.5, {ew}_{6}= $' + str(ew6)

                                if param_y == 'U1_OK_d_wl_sn21' and param_x == 'U1_OK_d_wl_sn10':
                                    y_u = x - 0.1 - 6.5 /(ew6 - 0.175)
                                    ax.plot(x, y_u, ls='--', linewidth=0.5, alpha=0.75, c=txt_color, label="Analytical Boundaries (dynamic)")
                                    ax.text(
                                        x[lseg], y_u[lseg] + 2*shift, ax_label, fontsize=txt_fontsize, c=txt_color, ha='center', va='center',
                                        rotation=0,)
                                    
                                    # legend for "Analytical boundaries (dynamic)"
                                    g._update_legend_data(ax)

            # legend arrangement.
            g._legend.remove()
            g.add_legend(label_order=g._legend_data.keys(), title=None)                        
            sns.move_legend(g, "upper center", bbox_to_anchor=(.45, 1.005), ncol=len(g._legend_data.keys()), title=None)

            # https://programtalk.com/vs4/python/icaros-usc/dqd-rl/external/seaborn/seaborn/axisgrid.py/

            # g.map_lower(sns.kdeplot, levels=2, color=".8", linestyles= 'dashed', alpha=0.75, linewidths=1)

            plt.savefig(DIRS_DATA_SS_FIG + r'\{}_{}_pairwise_relationship_{}.png'.format(versionSpace, sourceSpace, label_compliance), dpi=400)
            

    if plot_knc:
        pca = PCA(n_components=3)
        X = X.to_numpy()
        X_pc = pca.fit_transform(X)
        X = X_pc[:, :2]
        n_neighbors = 5
    
        for key in compliance_keys:
            
            y = Y[key]

            # Create color maps 
            cmap_light = ListedColormap(["orange", "cornflowerblue"])
            cmap_bold = ["darkorange", "darkblue"]

            for weights in ["uniform", "distance"]:

                # we create an instance of Neighbours Classifier and fit the data.
                clf = KNeighborsClassifier(n_neighbors, weights=weights)
                
                clf.fit(X, y)

                fig = plt.figure(figsize=(14, 10))
                fig, ax = plt.subplots()
                DecisionBoundaryDisplay.from_estimator(
                    clf,
                    X,
                    cmap=cmap_light,
                    ax=ax,
                    response_method="predict",
                    plot_method="pcolormesh",
                    # xlabel=X.feature_names[0],
                    # ylabel=iris.feature_names[1],
                    shading="auto",
                )

                # Plot also the training points
                sns.scatterplot(
                    x=X[:, 0],
                    y=X[:, 1],
                    hue=y,
                    palette=cmap_bold,
                    alpha=1.0,
                    edgecolor="black",
                )
                plt.title(
                    "3-Class classification (k = %i, weights = '%s')" % (n_neighbors, weights)
                )
                plt.savefig(DIRS_DATA_SS_FIG + r'\KNC_neighbors_{}_weight_{}_{}.png'.format(n_neighbors, weights, key), dpi=200)

    # if plot_space_tsne:
        
    #     comparison_compliance = dict()

    #     ini_embedding = 'random'
    #     value_perplexity = 50
    #     num_step = 1000
            
    #     for key in compliance_keys:
            
    #         # X: X_svm -> X_scaled -> X_scaled_reduced. 
    #         X_svm = copy.deepcopy(X)
    #         scaler = RobustScaler()
    #         X_scaled = scaler.fit_transform(X_svm)
    #         X_scaled_reduced =  TSNE(
    #             n_components=2,
    #             init=ini_embedding,
    #             perplexity=value_perplexity,
    #             n_iter = num_step,
    #             random_state=random_seed).fit_transform(X_scaled)

    #         # check:
    #         # https://stackoverflow.com/questions/66186272/how-to-plot-the-decision-boundary-of-a-one-class-svm
    #         # https://towardsdatascience.com/t-sne-clearly-explained-d84c537f53a
    #         # https://distill.pub/2016/misread-tsne/
    #         # [important]: https://github.com/scikit-learn/scikit-learn/issues/5361

    #         # parametric t-SNE
    #         # https://github.com/kylemcdonald/Parametric-t-SNE
    #         # https://github.com/scikit-learn/scikit-learn/pull/4025

    #         # Y: y.
    #         y = Y[key]

    #         # split for train and test.
    #         X_train, X_test, y_train, y_test = train_test_split(X_scaled_reduced, y, test_size=0.3, random_state=random_seed)

    #         # The standard approach is to use t-SNE to reduce the dimensionality of the data for visualization purposes.
    #         # Once you have reduced the data to two dimensions you can easily replicate the visualization in the scikit-learn tutorial
    #         # define the meshgrid

    #         # build and fit the svm.
    #         # svm_classifer = svm.SVC(kernel='linear',random_state=random_seed)
    #         svm_classifer = svm.SVC(random_state=random_seed)
    #         svm_classifer.fit(X_train, y_train)

    #         # predict and check accuracy
    #         # y_pred = svm_classifer.predict(X_scaled_reduced)

    #         y_pred_train = svm_classifer.predict(X_train)
    #         y_pred_test = svm_classifer.predict(X_test)

    #         accuracy = accuracy_score(y_test, y_pred_test)
    #         print ("accuracy for ", key, ": ", accuracy)
    #         # comparison_compliance.update({key: accuracy})
            
    #         # plot the decision function and the reduced data
    #         # x_min, x_max = X_scaled_reduced[:, 0].min() - 5, X_scaled_reduced[:, 0].max() + 5
    #         # y_min, y_max = X_scaled_reduced[:, 1].min() - 5, X_scaled_reduced[:, 1].max() + 5

    #         x_min, x_max = X_train[:, 0].min() - 5, X_train[:, 0].max() + 5
    #         y_min, y_max = X_train[:, 1].min() - 5, X_train[:, 1].max() + 5
    #         x_ = np.linspace(x_min, x_max, 500)
    #         y_ = np.linspace(y_min, y_max, 500)
    #         xx, yy = np.meshgrid(x_, y_)

    #         # create the decision function on the meshgrid.
    #         z = svm_classifer.decision_function(np.c_[xx.ravel(), yy.ravel()])
    #         z = z.reshape(xx.shape)
            
    #         # data plot.
    #         fig = plt.figure(figsize=(12, 12))
    #         plt.contourf(xx, yy, z, cmap=plt.cm.PuBu)
    #         a = plt.contour(
    #             xx, yy, z, levels=[0], linewidths=2, colors='darkred')
    #         b = plt.scatter(
    #             X_train[y_train == 1, 0], X_train[y_train == 1, 1], c="white", edgecolors='k') # Valid-train
    #         c = plt.scatter(
    #             X_train[y_train == 0, 0], X_train[y_train == 0, 1], c="gold", edgecolors='k') # Invalid-train
            
    #         # d = plt.scatter(
    #         #     X_test[y_pred_test == 1, 0], X_test[y_pred_test == 1, 1], c=sns.color_palette("seismic")[0], edgecolors='k') # Valid-test
    #         # e = plt.scatter(
    #         #     X_test[y_pred_test == 0, 0], X_test[y_pred_test == 0, 1], c=sns.color_palette("seismic")[-1], edgecolors='k') # Invalid-test
            
    #         plt.legend(
    #             [a.collections[0], b, c], #, d, e],
    #             ['learned frontier', 'Valid-train', 'Invalid-train'], #, 'Valid-test', 'Invalid-test'],
    #             bbox_to_anchor=(0.75, 0.996),
    #             ncol=5)
            
    #         plt.axis('tight')
    #         plt.savefig(DIRS_DATA_SS_FIG + r'\DecisionboundarySVM_{}_compliance_{}_init_{}_perplexity_{}_numstep_{}.png'.format(
    #             sourceSpace, key, ini_embedding, value_perplexity, num_step), dpi=200)
            
            # # = = = = = = = = = = = = = = = = = = = = = = = = =
            # # for scaler testing
            # # not scaled
            # # svm
            # svm_clf = svm.SVC(random_state=random_seed)
            # svm_clf.fit(X_train, y_train)
            # # result
            # y_pred_test_svm = svm_clf.predict(X_test)
            # accuracy_test_svm = accuracy_score(y_test, y_pred_test_svm)
            # comparison_scalers = dict()
            # comparison_scalers.update({
            #     'NonScaler': accuracy_test_svm
            # })
            # # scaled
            # for scaler, scaler_name in zip(
            #     [StandardScaler(), MinMaxScaler(), RobustScaler(), Normalizer()], ['StandardScaler', 'MinMaxScaler', 'RobustScaler', 'Normalizer']):
            #     X_train_scaled = scaler.fit_transform(X_train)
            #     X_test_scaled = scaler.transform(X_test)
            #     svm_clf_scaled = svm.SVC(random_state=random_seed)
            #     svm_clf_scaled.fit(X_train_scaled, y_train)
            #     y_pred_test_svm_scaled = svm_clf_scaled.predict(X_test_scaled)
            #     accuracy_test_svm_scaled = accuracy_score(y_test, y_pred_test_svm_scaled)
            #     comparison_scalers.update({
            #         scaler_name: accuracy_test_svm_scaled
            #         })
            # comparison_compliance.update({
            #     key: comparison_scalers
            # })
            # # = = = = = = = = = = = = = = = = = = = = = = = = =

        # with open(DIRS_DATA_SS + r"\scalers.json", "w") as outfile:
        #     json.dump(comparison_compliance, outfile)
        
            # - - - - -  
            # scaler = StandardScaler()
            # This class standardizes the features by subtracting the mean and dividing by the standard deviation,
            # which makes the features have a mean of 0 and a standard deviation of 1.

            # scaler = MinMaxScaler()
            # it works by subtracting the minimum value and dividing by the range,
            # which makes the features have a minimum of 0 and a maximum of 1.

            # scaler = RobustScaler()
            # It scales the features based on the median and interquartile range (IQR), making it robust to outliers.

            # scaler = Normalizer()
            # Normalization is a type of feature scaling where the goal is 
            # to adjust the values of a feature vector to have a unit norm, i.e.,
            # the sum of the squares of the feature values equals 1.
            # It is often used when working with distance-based algorithms, such as k-Nearest Neighbors,
            # to ensure that all features contribute equally to the distance calculation.
            
            # or by the previous PCA SVM.
