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

def plot_weighted_euclidean_distance(
    compliant_data, compliant_data_labels, x_distance, 
    subdistances, y_compliant_amount, y_compliant_amount_sum, 
    control_minor_valid, tol_subdistance, res_plot_idx, DIRS_DATA_SS_FIG):
        
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={'height_ratios': [1, 3]})
    
    c_j = ['#D9BC62','#D96273','#62D979','#6275D9','#67846D']
    
    for jj in range(len(compliant_data_labels[:5])):
        # plot the background line 
        ax2.plot(
            x_distance, 
            subdistances[jj,:],
            linewidth=0.3, 
            alpha=0.5, 
            c = c_j[jj],
            zorder=1,
        )
        
        # plot the background scatters
        ax2.scatter(
            x_distance[y_compliant_amount[-1]==1],
            subdistances[jj,y_compliant_amount[-1]==1],
            s=10, 
            alpha=0.5, 
            c = c_j[jj],
            edgecolors="k",
            linewidth=0.25,
            label='$\mathbf{H}_{j,i}$ (i=' + compliant_data_labels[jj].replace("U1_OK_d_wl_","") + ')',
            zorder=5,
        )
        
        # plot the minor scatters.
        ax2.scatter(
            x_distance[res_plot_idx[1:]],
            subdistances[jj,res_plot_idx[1:]],
            s=15, 
            alpha=0.8, 
            c = c_j[jj],
            edgecolors="k",
            linewidth=1,
            zorder=10,
        )

    for id in res_plot_idx[1:]:
        x_v = x_distance[id]
        ax2.axvline(
            x = x_v,
            ls = '--',
            linewidth=0.50,
            color='#808080'
        )
    
    original_ticks = x_distance[res_plot_idx[1:]]
    additional_ticks = np.arange(0, max(x_distance) + 0.25, 0.25)
    combined_ticks = np.unique(np.concatenate((original_ticks, additional_ticks)))
    combined_ticks = [round(x_v, 3) for x_v in combined_ticks]
    ax2.set_xticks(combined_ticks)
    for label in ax2.get_xticklabels():
        label.set_rotation(90)
    ax2.tick_params(axis='x', direction='in', pad=-42, labelsize=13)  
        
    ax2.legend(loc=3, prop={'size': 14}) # lower
    ax2.set_yscale("log")
    ax2.set_ylabel("The factorial Euclidean Distance $\mathbf{H}_{j,i}$", color="black", fontsize=18)
    ax2.set_xlabel("The weighted Eucliean Distance $\mathbf{H}_{j}$", color="black", fontsize=18)

    c_i = ['#938888','#938888','#938888','#000000']
    l_styles = ['dotted','dashed','dashdot','solid']
    for ii in range(len(compliant_data_labels[11:])):
        ax1.plot(
            x_distance,
            y_compliant_amount_sum[ii,:],
            label="$N_{valid}$ (" + compliant_data_labels[11+ii] + ")",
            linewidth=1,
            linestyle = l_styles[ii],
            c = c_i[ii],
        )

    ax1.legend(loc=2, prop={'size': 10}) # upper
    ax1.set_ylabel("$N_{valid}$", color="black", fontsize=18)
    
    fig.tight_layout()
    plt.savefig(DIRS_DATA_SS_FIG + r'\Distance_compliance_relationship.png', dpi=400)
    
def reasonSolutionSpaceWeighted(
    dataset=[],
    calc_valid_distance=False,
    ):
    # load the space
    sourceSpace = '_'.join([data.replace('\\','') for data in dataset])
    file_solutionspace = DIRS_DATA_SS + r'\Space_' + sourceSpace + r'.pickle'
    
    currentSpace = load_dict(file_solutionspace)
    currentSpace.__addini__()
    currentSpace.remove_constant()
    currentSpace.data_X_Y.to_csv(DIRS_DATA_SS+ r'\all_data.csv', header=True)
    
    if calc_valid_distance:
        
        # after manual selection, some of the valid designs are selected for visualization.
        # weighted * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        # # 1282 381-1 smallest distance. w-euc = 0.419
        # # 1439 451-1 smallest distance. w-euc = 0.445
        # # 1522: 482-1 two values very small. w-euc = 0.454
        res_plot_nr = [0, 1282, 1439, 1522] # choose and add
        res_plot_idx = [0, 381-1, 451-1, 482-1] # choose and add
        # weighted * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

        # currentSpace.calculate_distance()
        currentSpace.calculate_weighted_distance()
        compliant_data = currentSpace.distance_X_Y_sorted.values.tolist()
        
        # drop the initial design
        compliant_data.pop(0) 
        compliant_data = np.array(compliant_data).T

        compliant_data_labels = list(currentSpace.distance_X_Y_sorted.keys())
        subdistances = compliant_data[:5]
        x_distance = compliant_data[5]
        y_compliant_amount = compliant_data[11:]
        y_compliant_amount_sum = np.cumsum(y_compliant_amount, axis=1)

        # find the lower limit(index) of 97.5% values   
        subdistances_flatten = subdistances.flatten()
        subdistances_flatten_sorted = np.argsort(subdistances_flatten)[:int(subdistances_flatten.shape[0]*(1-0.975))]

        # calculate the subdistance value 
        tol_subdistance = subdistances_flatten[subdistances_flatten_sorted[-1]]

        # mark the subdistance values smaller than the limited value.
        control_minor = []
        control_valid = list(y_compliant_amount[-1]==1)
        for id in range(subdistances.shape[1]):
            valid_dist = (subdistances.T)[id]
            if (valid_dist < tol_subdistance).sum() >= 1:
                control_minor.append(True)
            else:
                control_minor.append(False)
        
        control_minor_valid = [all([v1,v2]) for v1,v2 in zip(control_minor,control_valid)]
        
        # output the potential alternatives.
        ouput_df = copy.deepcopy(currentSpace.distance_X_Y_sorted)
        ouput_df['designnumber'] = ouput_df.index
        ouput_df = ouput_df.reset_index(drop=True)

        # type1: via distance.
        ouput_df_valid_via_distance = copy.deepcopy(ouput_df)
        ouput_df_valid_via_distance = ouput_df_valid_via_distance[ouput_df_valid_via_distance['compliance'] == True]
        ouput_df_valid_via_distance.to_csv(DIRS_DATA_SS+ r'\valid_via_weighted_distance.csv', header=True)

        # type2: via minor change.
        ouput_df_valid_via_minor = copy.deepcopy(ouput_df)
        control_minor_valid_withorigin = copy.deepcopy(control_minor_valid)
        control_minor_valid_withorigin.insert(0,False)
        ouput_df_valid_via_minor = ouput_df_valid_via_minor.loc[ouput_df_valid_via_minor.index[control_minor_valid_withorigin]]
        ouput_df_valid_via_minor.to_csv(DIRS_DATA_SS+ r'\valid_via_weighted_minor.csv', header=True)

        plot_weighted_euclidean_distance(
            compliant_data, compliant_data_labels, x_distance, 
            subdistances, y_compliant_amount, y_compliant_amount_sum, 
            control_minor_valid, tol_subdistance, res_plot_idx, DIRS_DATA_SS_FIG
        )

def reasonSolutionSpace(
    dataset=[],
    transfer_space=False,
    inter_level=0,
    plot_space_pairwise=False,
    plot_space_svm=False,
    plot_knc=False,
    calc_valid_distance=False,
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
    
    currentSpace.data_X_Y.to_csv(DIRS_DATA_SS+ r'\all_data.csv', header=True)
    
    if calc_valid_distance:
        
        # after manual selection, some of the valid designs are selected for visualization.

        # # non-weighted * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        # res_plot_nr = [0, 1282, 1439, 1522, 1718, 669]
        # res_plot_idx = [0, 362-1, 378-1, 403-1, 837-1, 1720-1]
        # # non-weighted * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        # here. todo.
        # weighted * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
        # # 1282 381-1 smallest distance. w-euc = 0.419
        # # 1522: 482-1 two values very small. w-euc = 0.454
        # # 1718: 1079-1 two values very small. w-euc = 0.577
        res_plot_nr = [0, 1282, 1522, 1718] # choose and add
        res_plot_idx = [0, 381-1, 482-1, 1079-1] # choose and add
        # weighted * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


        # currentSpace.calculate_distance()
        currentSpace.calculate_weighted_distance()
        compliant_data = currentSpace.distance_X_Y_sorted.values.tolist()
        
        # drop the initial design
        compliant_data.pop(0) 
        compliant_data = np.array(compliant_data).T

        compliant_data_labels = list(currentSpace.distance_X_Y_sorted.keys())
        subdistances = compliant_data[:5]
        x_distance = compliant_data[5]
        y_compliant_amount = compliant_data[11:]
        y_compliant_amount_sum = np.cumsum(y_compliant_amount, axis=1)

        # find the lower limit(index) of 97.5% values   
        subdistances_flatten = subdistances.flatten()
        subdistances_flatten_sorted = np.argsort(subdistances_flatten)[:int(subdistances_flatten.shape[0]*(1-0.975))]

        # calculate the subdistance value 
        tol_subdistance = subdistances_flatten[subdistances_flatten_sorted[-1]]

        # mark the subdistance values smaller than the limited value.
        control_minor = []
        control_valid = list(y_compliant_amount[-1]==1)
        for id in range(subdistances.shape[1]):
            valid_dist = (subdistances.T)[id]
            if (valid_dist < tol_subdistance).sum() >= 1:
                control_minor.append(True)
            else:
                control_minor.append(False)
        
        control_minor_valid = [all([v1,v2]) for v1,v2 in zip(control_minor,control_valid)]
        
        # output the potential alternatives.
        ouput_df = copy.deepcopy(currentSpace.distance_X_Y_sorted)
        ouput_df['designnumber'] = ouput_df.index
        ouput_df = ouput_df.reset_index(drop=True)

        # # type1: via distance.
        # ouput_df_valid_via_distance = copy.deepcopy(ouput_df)
        # ouput_df_valid_via_distance = ouput_df_valid_via_distance[ouput_df_valid_via_distance['compliance'] == True]
        # ouput_df_valid_via_distance.to_csv(DIRS_DATA_SS+ r'\valid_via_distance.csv', header=True)

        # # type2: via minor change.
        # ouput_df_valid_via_minor = copy.deepcopy(ouput_df)
        # control_minor_valid_withorigin = copy.deepcopy(control_minor_valid)
        # control_minor_valid_withorigin.insert(0,False)
        # ouput_df_valid_via_minor = ouput_df_valid_via_minor.loc[ouput_df_valid_via_minor.index[control_minor_valid_withorigin]]
        # ouput_df_valid_via_minor.to_csv(DIRS_DATA_SS+ r'\valid_via_minor.csv', header=True)

        # type1: via distance.
        ouput_df_valid_via_distance = copy.deepcopy(ouput_df)
        ouput_df_valid_via_distance = ouput_df_valid_via_distance[ouput_df_valid_via_distance['compliance'] == True]
        ouput_df_valid_via_distance.to_csv(DIRS_DATA_SS+ r'\valid_via_weighted_distance.csv', header=True)

        # type2: via minor change.
        ouput_df_valid_via_minor = copy.deepcopy(ouput_df)
        control_minor_valid_withorigin = copy.deepcopy(control_minor_valid)
        control_minor_valid_withorigin.insert(0,False)
        ouput_df_valid_via_minor = ouput_df_valid_via_minor.loc[ouput_df_valid_via_minor.index[control_minor_valid_withorigin]]
        ouput_df_valid_via_minor.to_csv(DIRS_DATA_SS+ r'\valid_via_weighted_minor.csv', header=True)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={'height_ratios': [1, 3]})
    
        c_j = ['#86BE3C','#E4B645','#C33734','#748EC4','#808080']
        for jj in range(len(compliant_data_labels[:5])):

            # plot the background line 
            ax2.plot(
                x_distance, 
                subdistances[jj,:],
                linewidth=0.3, 
                alpha=0.5, 
                c = c_j[jj],
                zorder=1,
                )
            
            # plot the background scatters
            ax2.scatter(
                x_distance[y_compliant_amount[-1]==1],
                subdistances[jj,y_compliant_amount[-1]==1],
                s=10, 
                alpha=0.5, 
                c = c_j[jj],
                edgecolors="k",
                linewidth=0.25,
                label='$\mathbf{H}_{j,i}$ (i=' + compliant_data_labels[jj] + ')',
                # label='$\mathbf{H}$' + '$i=' + compliant_data_labels[jj],
                zorder=5,)
            
            # plot the minor scatters.
            ax2.scatter(
                x_distance[control_minor_valid],
                subdistances[jj,control_minor_valid],
                s=10, 
                alpha=0.5, 
                c = c_j[jj],
                edgecolors="k",
                linewidth=1,
                zorder=10,)
        
        # the tol.
        ax2.axhline(
            y = tol_subdistance,
            ls = '--',
            linewidth=0.75,
            color='#808080')
        ax2.text(1.05, tol_subdistance*1.05,'$\mathbf{H}_{j,i} = 0.01 * max(\mathbf{H}_{j})$')

        for id in res_plot_idx[1:]:
            x_v = x_distance[id]
            ax2.axvline(
                x = x_v,
                ls = '--',
                linewidth=0.50,
                color='#808080')
        ax2.set_xticks(x_distance[res_plot_idx[1:]])
        label_x_v = [round(x_v, 3) for x_v in x_distance[res_plot_idx[1:]]]
        ax2.set_xticklabels(label_x_v)
        for label in ax2.get_xmajorticklabels():
             label.set_rotation(90)
        ax2.tick_params(axis='x', direction='in', pad=-35)
            
        ax2.legend(loc=3, prop={'size': 14}) # lower
        ax2.set_yscale("log")
        ax2.set_ylabel("The factorial Euclidean Distance $\mathbf{H}_{j,i}$", color="black", fontsize=16)
        ax2.set_xlabel("The total Eucliean Distance $\mathbf{H}_{j}$", color="black", fontsize=16)

        c_i = ['#938888','#938888','#938888','#000000']
        l_styles = ['dotted','dashed','dashdot','solid']
        for ii in range(len(compliant_data_labels[11:])):
            ax1.plot(
                x_distance,
                y_compliant_amount_sum[ii,:],
                label="$N_{valid}$ (" + compliant_data_labels[11+ii] + ")",
                linewidth=1,
                linestyle = l_styles[ii],
                c = c_i[ii],
                )

        ax1.legend(loc=2,prop={'size': 11}) # upper
        ax1.set_ylabel("$N_{valid}$", color="black", fontsize=16)
        
        fig.tight_layout()
        plt.savefig(DIRS_DATA_SS_FIG + r'\Distance_compliance_relationship.png', dpi=400)
        

    if plot_space_svm:
        
        # X: X_svm -> X_scaled -> X_scaled_reduced. 
        X_svm = copy.deepcopy(X)

        # pca = PCA(n_components=2)
        # X_pca = pca.fit_transform(X_svm)
        # scaler = RobustScaler()
        # X_scaled = scaler.fit_transform(X_svm)

        svm_res = dict()

        for key in compliance_keys:
        
            # Y: y.
            y = Y[key]

            # split for train and test.
            # 
            # v_gamma = 1
            # svm_classifer = svm.SVC(kernel="rbf",gamma=v_gamma,random_state=random_seed)
            
            svm_res_perrule = dict()
            v_svm_C_items = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e+2, 1e+3, 1e+4, 1e+5, 1e+6]

            for v_svm_C in v_svm_C_items:
                
                # calculate the decision boundary.
                svm_classifer = svm.SVC(kernel="linear", C=v_svm_C, random_state=random_seed)
                svm_classifer.fit(X_svm, y)

                svm_cofs = np.concatenate((svm_classifer.coef_, [svm_classifer.intercept_]),axis=1).tolist()
                svm_cofs = flatten(svm_cofs)
                
                # [tempo] calculate the accuracy.
                X_train, X_test, y_train, y_test = train_test_split(X_svm, y, test_size=0.3, random_state=random_seed)
                svm_classifer = svm.SVC(kernel="linear", C=v_svm_C, random_state=random_seed)
                svm_classifer.fit(X_train, y_train)
                y_pred_test = svm_classifer.predict(X_test)
                svm_75_accuracy = accuracy_score(y_test, y_pred_test)
                # [tempo] end
                
                svm_cofs.append(svm_75_accuracy)

                svm_ftrs = list(svm_classifer.feature_names_in_)
                svm_ftrs.append('b')
                svm_ftrs.append('accuracy')

                dict_svm_res_perrule = {f: c for (f,c) in zip(svm_ftrs, svm_cofs)}
                svm_res_perrule.update({v_svm_C: dict_svm_res_perrule})


            svm_res.update({key: svm_res_perrule})

        with open(DIRS_DATA_SS + r"\svm_results.json", "w") as outfile:
            json.dump(svm_res, outfile)


            # displaySVCinPC(X_scaled, y, path = DIRS_DATA_SS_FIG, rule_label = key)

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

                        xmin, xmax = ax.get_xlim()
                        ymin, ymax = ax.get_ylim()
                        total_line_segs = 50
                        lseg = int(total_line_segs/2)

                        x = np.linspace(xmin, xmax, total_line_segs)
                        y = np.linspace(ymin, ymax, total_line_segs)

                        shift = 0.05
                        txt_color = 'navy'
                        txt_fontsize = 6

                        # ---------------------------------
                        # 1020_2

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
                                    x[lseg], y_u[lseg] + 2*shift, ax_label, fontsize=txt_fontsize, c = txt_color, ha='center', va='center',
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

                            ax_label = '$ ({sn}_{21} - 0.175) * ({ew}_{6} - 0.175) ≥ 6.5 $'

                            if param_y == 'U1_OK_d_wl_ew6' and param_x == 'U1_OK_d_wl_sn21':
                                y_l = 6.5 / (x-0.175) + 0.175
                                ax.plot(x, y_l, ls='-', linewidth=1.0, alpha=0.95, c = txt_color, label="Analytical Boundaries")
                                ax.text(
                                    x[lseg], y_l[lseg] - shift, ax_label, fontsize=txt_fontsize, c = txt_color, ha='center', va='center',
                                    rotation=-35,)
                                
                                # legend for "Analytical Boundaries"
                                g._update_legend_data(ax)

                            txt_color = 'darkgreen'
                            txt_fontsize = 5

                            for ew6 in [2.309, 3]:
                                
                                ax_label = '$ ({sn}_{26} - {sn}_{10} - 0.1) * ({ew}_{6} - 0.175) ≥ 6.5, {ew}_{6}= $' + str(ew6)

                                if param_y == 'U1_OK_d_wl_sn26' and param_x == 'U1_OK_d_wl_sn10':
                                    
                                    y_l = x + 0.1 + 6.5 /(ew6 - 0.175)
                                    ax.plot(x, y_l, ls='--', linewidth=0.5, alpha=0.75, c=txt_color, label="Analytical Boundaries (dynamic)")
                                    ax.text(
                                        x[lseg], y_l[lseg] + shift, ax_label, fontsize=txt_fontsize, c=txt_color, ha='center', va='center',
                                        rotation=0,)

                                ax_label = '$ ({sn}_{10} - {sn}_{21} - 0.1)  * ({ew}_{6} - 0.175) ≥ 6.5, {ew}_{6}= $' + str(ew6)

                                if param_y == 'U1_OK_d_wl_sn21' and param_x == 'U1_OK_d_wl_sn10':
                                    y_u = x - 0.1 - 6.5 /(ew6 - 0.175)
                                    ax.plot(x, y_u, ls='--', linewidth=0.5, alpha=0.75, c=txt_color, label="Analytical Boundaries (dynamic)")
                                    ax.text(
                                        x[lseg], y_u[lseg] + shift, ax_label, fontsize=txt_fontsize, c=txt_color, ha='center', va='center',
                                        rotation=0,)
                                    
                                    # legend for "Analytical boundaries (dynamic)"
                                    g._update_legend_data(ax)

                        # if '1020_2' in label_compliance or label_compliance == 'compliance':
                            
                        if '1020_2' in label_compliance:
                        
                            ax_label = '$ {ew}_{6} ≤ 0.992 {ew}_{35}-6.908 $'

                            if param_y == 'U1_OK_d_wl_ew6' and param_x == 'U1_OK_d_wl_ew35':
                                
                                y_l = 0.992 * x - 6.908
                                ax.plot(x, y_l, ls='-', linewidth=1.0, alpha=0.95, c = 'cyan', label="Boundaries by SVM")
                                ax.text(
                                    x[lseg], y_l[lseg] - shift, ax_label, fontsize=txt_fontsize, c = 'cyan', ha='center', va='center',
                                    rotation=35,)
                                
                        #         # legend for "Analytical Boundaries"
                        #         # g._update_legend_data(ax)
                            

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
