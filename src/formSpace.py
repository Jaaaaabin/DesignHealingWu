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
from funct_svm import displaySVCinPC

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
    set_new_space=False,
    transfer_space=False,
    plot_space_pairwise=False,
    plot_space_svm=False,
    random_seed=521,
    ):

    # load the space
    sourceSpace = '_'.join([data.replace('\\','') for data in dataset])
    file_solutionspace = DIRS_DATA_SS + r'\Space_' + sourceSpace + r'.pickle'
    currentSpace = load_dict(file_solutionspace)
    
    # extract X values.
    constantVariable = currentSpace.data_X_df.columns[currentSpace.data_X_df.nunique() <= 1].tolist()
    spaceVariable = [v for v in currentSpace.data_X_df.columns.tolist() if v not in constantVariable]
    X = currentSpace.data_X_df[spaceVariable]
    
    if transfer_space:
        X['U1_OK_d_wl_sn26'] = X['U1_OK_d_wl_sn26'] - X['U1_OK_d_wl_sn10']
        X['U1_OK_d_wl_sn10'] = X['U1_OK_d_wl_sn10'] - X['U1_OK_d_wl_sn21']
        X['U1_OK_d_wl_ew35'] = X['U1_OK_d_wl_ew35'] - X['U1_OK_d_wl_ew6']

    # extract Y values.
    compliance_keys = list(currentSpace.data_Y_dict.keys())
    Y = currentSpace.data_Y_dict

    # plot
    if plot_space_pairwise:

        # INPUT Data distribution.
        # https://www.kaggle.com/code/sunaysawant/iris-pair-plot-palettes-all-170-visualisations

        for label_compliance in list(currentSpace.valid_idx.keys()):

            spaceVariable_label = spaceVariable.copy()

            # add the initial design. (x and y)
            ini_plot = [currentSpace.ini_parameters[variable] for variable in spaceVariable_label]  # adding a row
            ini_plot.append('Initial Design')

            # preparet the dataframe for plot.
            spaceVariable_label.append(label_compliance)
            df_data = currentSpace.data_X_Y[spaceVariable_label]

            # add the initial design and its label.
            df_data.loc[-1] = ini_plot 
            df_data.index = df_data.index + 1
            df_data = df_data.sort_index()

            # transfer_space
            if transfer_space:
                
                versionSpace = 'TSpace'
                df_data['U1_OK_d_wl_sn26'] = df_data['U1_OK_d_wl_sn26'] - df_data['U1_OK_d_wl_sn10']
                df_data['U1_OK_d_wl_sn10'] = df_data['U1_OK_d_wl_sn10'] - df_data['U1_OK_d_wl_sn21']
                df_data['U1_OK_d_wl_ew35'] = df_data['U1_OK_d_wl_ew35'] - df_data['U1_OK_d_wl_ew6']
            else:

                versionSpace = 'Space'

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
                plot_kws={"s": 5},
                palette=palette_3d)
            
            for ax in g.axes.ravel():
                
                param_x = ax.get_xlabel()
                param_y = ax.get_ylabel()

                if param_x and param_y:

                    param_x_v = df_plot[param_x].iloc[-1]
                    param_y_v = df_plot[param_y].iloc[-1]
                    ax.axvline(x=param_x_v, ls='-.', linewidth=0.5, alpha=0.75, c='black')
                    ax.axhline(y=param_y_v, ls='-.', linewidth=0.5, alpha=0.75, c='black')
            
            g.map_lower(sns.kdeplot, levels=2, color=".8", linestyles= 'dashed', alpha=0.75, linewidths=1)
            
            sns.move_legend(g, "upper center", bbox_to_anchor=(.45, 1), ncol=3, title=None)
            
            plt.savefig(DIRS_DATA_SS_FIG + r'\{}_{}_pairwise_relationship_{}.png'.format(versionSpace, sourceSpace, label_compliance), dpi=300)


    if plot_space_svm:
        
        comparison_compliance = dict()

        for key in compliance_keys:
            
            y = Y[key]

            # scale
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)
            X_scaled_reduced =  TSNE(n_components=2, random_state=random_seed).fit_transform(X_scaled)

            # split
            X_train, X_test, y_train, y_test = train_test_split(X_scaled_reduced, y, test_size=0.3, random_state=random_seed)

            # reduce
            # The standard approach is to use t-SNE to reduce the dimensionality of the data for visualization purposes.
            # Once you have reduced the data to two dimensions you can easily replicate the visualization in the scikit-learn tutorial
            # define the meshgrid

            # svm
            svm_clf_scaled = svm.SVC(random_state=random_seed)
            svm_clf_scaled.fit(X_train, y_train)

            # predict and check accuracy
            y_pred_test_svm_scaled = svm_clf_scaled.predict(X_test)
            accuracy_test_svm_scaled = accuracy_score(y_test, y_pred_test_svm_scaled)
            comparison_compliance.update({key: accuracy_test_svm_scaled})
            
            # start here.
            # check: https://stackoverflow.com/questions/66186272/how-to-plot-the-decision-boundary-of-a-one-class-svm
            x_min, x_max = X_test_reduced[:, 0].min() - 5, X_test_reduced[:, 0].max() + 5
            y_min, y_max = X_test_reduced[:, 1].min() - 5, X_test_reduced[:, 1].max() + 5
            x_ = np.linspace(x_min, x_max, 500)
            y_ = np.linspace(y_min, y_max, 500)
            xx, yy = np.meshgrid(x_, y_)

            # # evaluate the decision function on the meshgrid
            # z = svm_clf_scaled.decision_function(np.c_[xx.ravel(), yy.ravel()])
            # z = z.reshape(xx.shape)

            # # plot the decision function and the reduced data
            # plt.contourf(xx, yy, z, cmap=plt.cm.PuBu)
            # a = plt.contour(xx, yy, z, levels=[0], linewidths=2, colors='darkred')
            # b = plt.scatter(X_test_scaled_reduced[y_pred_test_svm_scaled == 1, 0], X_test_scaled_reduced[y_pred_test_svm_scaled == 1, 1], c='white', edgecolors='k')
            # c = plt.scatter(X_test_scaled_reduced[y_pred_test_svm_scaled == -1, 0], X_test_scaled_reduced[y_pred_test_svm_scaled == -1, 1], c='gold', edgecolors='k')
            # plt.legend([a.collections[0], b, c], ['learned frontier', 'regular observations', 'abnormal observations'], bbox_to_anchor=(1.05, 1))
            # plt.axis('tight')
            # plt.show()

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

        with open(DIRS_DATA_SS + r"\scalers.json", "w") as outfile:
            json.dump(comparison_compliance, outfile)
        
            # displaySVCinPC(
            #     X, y, path = DIRS_DATA_SS_FIG, rule_label = key)



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
