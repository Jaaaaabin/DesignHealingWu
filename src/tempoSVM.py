#
# tempoCheck.py
#

# import modules

# https://www.analyticsvidhya.com/blog/2021/03/beginners-guide-to-support-vector-machine-svm/
# https://towardsdatascience.com/support-vector-machines-svm-clearly-explained-a-python-tutorial-for-classification-problems-29c539f3ad8
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
# http://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote09.html

from base_external_packages import *
from funct_data import *
from Space import SolutionSpace

# import matplotlib.pyplot as plt
# from sklearn import svm
# from sklearn.datasets import make_blobs
# from sklearn.inspection import DecisionBoundaryDisplay

set = '\sa-18-0.3'
pathIni = r'C:\dev\phd\ModelHealer\data'+ set + r'\DesignIni.pickle'
pathNew = r'C:\dev\phd\ModelHealer\data'+ set + r'\DesignsNew.pickle'

tempoIni = load_dict(pathIni)
tempoNew =  load_dict(pathNew)

pathRes= r'C:\dev\phd\ModelHealer\data'+set + r'\res'
lst_txt = []
for file in os.listdir(pathRes):
    if file.endswith(".txt"):
        lst_txt.append(os.path.join(file))
lst_txt = [txt.replace('results_y_','') for txt in lst_txt]
lst_txt = [txt.replace('.txt','') for txt in lst_txt]
lst_txt = [txt.split("_", 1) for txt in lst_txt]
test_txt = lst_txt[0]

testSpace = SolutionSpace(ifcguid=test_txt[0], rule=test_txt[1])
testSpace.set_space_center(tempoIni)
testSpace.form_space(tempoNew)


# results_y_2SzsE5m8T4h9JlM6XpBSn3_IBC1020_2

# we create two clusters of random points
n_samples_1 = 1000
n_samples_2 = 100
centers = [[0.0, 0.0], [2.0, 2.0]]
clusters_std = [1.5, 0.5]
X, y = make_blobs(
    n_samples=[n_samples_1, n_samples_2],
    centers=centers,
    cluster_std=clusters_std,
    random_state=0,
    shuffle=False,
)

# fit the model and get the separating hyperplane
clf = svm.SVC(kernel="linear", C=1.0)
clf.fit(X, y)

# fit the model and get the separating hyperplane using weighted classes
wclf = svm.SVC(kernel="linear", class_weight={1: 10})
wclf.fit(X, y)

# plot the samples
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors="k")

# plot the decision functions for both classifiers
ax = plt.gca()
disp = DecisionBoundaryDisplay.from_estimator(
    clf,
    X,
    plot_method="contour",
    colors="k",
    levels=[0],
    alpha=0.5,
    linestyles=["-"],
    ax=ax,
)

# plot decision boundary and margins for weighted classes
wdisp = DecisionBoundaryDisplay.from_estimator(
    wclf,
    X,
    plot_method="contour",
    colors="r",
    levels=[0],
    alpha=0.5,
    linestyles=["-"],
    ax=ax,
)

plt.legend(
    [disp.surface_.collections[0], wdisp.surface_.collections[0]],
    ["non weighted", "weighted"],
    loc="upper right",
)
plt.show()


# # Read the initial data
# df_init = pd.read_csv(r'data/0_initial_parameters.csv',
#                       index_col=0, header=None).T

# # Read the sampling data
# df_raw = pd.read_csv(r'data/samples.csv')

# # drop list
# # drop empty columns that concerns zone4 and other unassociated columns
# drop_list = ['room_zone4_x', 'room_zone4_y', 'zone4_sep_x1']
# add_drop_list = ['room_zone1_x', 'room_zone1_y', 'zone1_sep_x1',
#                  'zone3_sep_x1', 'zone3_sep_x2', 'zone3_sep_x3']
# drop_list = drop_list + add_drop_list
# df_raw.drop(columns=drop_list, inplace=True)
# df = df_raw.iloc[:, 5:12].copy()

# rules_list = ['comp_IBC1020_2', 'comp_IBC1207_1', 'comp_IBC1207_2', 'comp_IBC1207_3']
# y_train_byrule = dict.fromkeys(rules_list)
# y_train_weight_byrule = dict.fromkeys(rules_list)

# g_all_rl = []
# for rl in rules_list:
#     g_rl = df_raw[rl].to_numpy()
#     g_all_rl.append(g_rl)
#     y_train_byrule[rl] = [g_rl[i] for i in range(np.shape(g_rl)[0])]
#     y_train_weight_byrule[rl] = [4.0 if item else 1.0 for item in y_train_byrule[rl]]
    
# y_train_allrules = [g_all_rl[0][i] and g_all_rl[1][i] and g_all_rl[2][i] and g_all_rl[3][i] for i in range(np.shape(g_all_rl[0])[0])]
# y_train_weight_allrules = [4.0 if item else 1.0 for item in y_train_allrules]

# test_rule = 'all'

# if test_rule == 'all':
#     y_train = y_train_allrules
#     y_train_weight = y_train_weight_allrules
# else:
#     for rl in list(y_train_byrule.keys()):
#         if test_rule in rl:
#             y_train = y_train_byrule[rl]
#             y_train_weight = y_train_weight_byrule[rl]

# svc, y_pred, dist, y_pred_error_index, y_pred_val_index = execute_linearsvc(test_rule, X_train, y_train, y_train_weight)
# y_pred_error_dist = [item for item in dist[y_pred_error_index]]
# y_pred_val_dist = [item for item in dist[y_pred_val_index]]

# # initial parameter values
# sample_ini = [12., 7.5, 5., 6., 3., 6., 10.]

# # calculate the dissimilarity
# y_pred_val_dissimilarity = []
# for i in y_pred_val_index:
#     sp = list(df.iloc[i])
#     for ii in range(len(sample_ini)):
#         sp[ii] = abs((sp[ii] - sample_ini[ii]) / sample_ini[ii])
#         sp[ii] = sp[ii] * sp[ii]
#     y_pred_val_dissimilarity.append(sum(sp))

# # normalization of the dissimilarity of the parameter value (percent) to [0,1]
# α_norm, β_norm = get_α_β(y_pred_val_dissimilarity, norm=True, norm_01=True)
# norm_y_pred_val_dissimilarity = out_stnd_nrml(y_pred_val_dissimilarity, α_norm, β_norm)

# # normalization of the distance from valid design options to the predicted hyperplane to [0,1]
# α_norm, β_norm = get_α_β(y_pred_val_dist, norm=True, norm_01=True)
# norm_y_pred_val_dist = out_stnd_nrml(y_pred_val_dist, α_norm, β_norm)


# fig, ax = plt.subplots(figsize=(30, 8))
# ax.scatter(y_pred_val_index, norm_y_pred_val_dissimilarity, s=100, marker="o", label = 'via parameter square difference')
# ax.plot(y_pred_val_index, norm_y_pred_val_dissimilarity, linewidth=4)

# ax.scatter(y_pred_val_index, norm_y_pred_val_dist, s=100, marker="^", label = 'via the predicted distance to hyperplane')
# ax.plot(y_pred_val_index, norm_y_pred_val_dist, linewidth=4)

# ax.scatter(y_pred_error_index,[0]*len(y_pred_error_index), s=150, marker="X", color = "maroon", label = 'wrong classification prediction')

# ax.set_xticks(y_pred_val_index)
# ax.tick_params(axis='x', which='major', direction='out', length=5, width=2, color='grey',
#                     pad=5, labelsize=8, labelcolor='black', labelrotation=90)
# ax.tick_params(axis='y', labelsize=15, labelcolor='black')
# ax.legend(loc="best", fontsize=15)
# ax.set_title("Dissimilarity / Distance", size=25)
# # ax.set_title("Dissimilarity from predicted valid design options to the initial design", size=18)

# """
# Plot second sensitivity indices
# """


# def convert2Matrix(second):
#     param_names = []
#     for m in range(second.shape[0]):
#         for n in range(second.shape[1]):
#             if second.index[m][n] not in param_names:
#                 param_names.append(second.index[m][n])
#             else:
#                 continue

#     matrix = np.zeros((len(param_names), len(param_names)), float)
#     for k in range(second['S2'].shape[0]):
#         j = param_names.index(second['S2'].index[k][0])
#         i = param_names.index(second['S2'].index[k][1])
#         matrix[j][i] = second['S2'].iloc[k]
#     return param_names, matrix


# def plotSecondIndices(dirs_fig, rl, second):

#     param_names, matrix = convert2Matrix(second)

#     fig = plt.figure(figsize=(10, 10))  # unit of inch
#     ax1 = plt.axes((0.1, 0.1, 0.8, 0.8))  # in range (0,1)

#     pos = ax1.imshow(matrix, interpolation='none', cmap='BuPu')

#     ax1.set_xticks(np.arange(len(param_names)), param_names)
#     ax1.set_yticks(np.arange(len(param_names)), param_names)
#     ax1.tick_params(axis='x', which='major', direction='out', length=5, width=2, color='maroon',
#                     pad=10, labelsize=10, labelcolor='navy', labelrotation=15)
#     ax1.tick_params(axis='y', which='major', direction='out', length=5, width=2, color='maroon',
#                     pad=10, labelsize=10, labelcolor='navy', labelrotation=15)
#     ax1.set_title(r'Second-order sensitivity indices for rule: '+rl, size=16)
#     fig.colorbar(pos, location='right', shrink=0.8)

#     for (i, j), z in np.ndenumerate(matrix):
#         if z != 0:
#             ax1.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')

#     plt.savefig(dirs_fig + '/SA_' + rl + '_Second' + '_indices.png', dpi=200)

# nu_list = [0.01, 0.02, 0.05, 0.1, 0.125]
# display_svc_pc(X_train, y_train, svckernel="linear")

# for nu_v in nu_list:
#     display_svc_pc(X_train, y_train, svckernel="nu", nu_nu=nu_v)