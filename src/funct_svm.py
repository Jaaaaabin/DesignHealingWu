#
# funct_svm.py
# 

# import packages
from base_external_packages import *

# import modules
from base_functions import *

def out_stnd_nrml(u, α, β):
    ''' Standardize/normalize '''
    output = (u - β) / α
    return output


def u_stnd_nrml(output, α, β):
    ''' Un-standardize/un-normalize '''
    u = output * α + β
    return u


def get_α_β(u, norm=True, norm_01=False, no=False):
    ''' Get the coefficients of normalization '''
    # norm=True: if norm==False, no normalization.. go for standardization
    # norm_01=False: if norm_01==True, rescale to [0,1]; if norm_01==False, rescale to [-1,1]
    # no=False: if no==True, u_α = 1. u_β = 0., no conversion
    if no == False:
        # do something
        if norm == True:
            # do normalization
            if norm_01 == False:
                # normalization to 01
                u_max = np.amax(u)
                u_min = np.amin(u)
                # print('u_max : ', u_max)
                # print('u_min : ', u_min)
                u_α = (u_max - u_min) / 2.
                u_β = (u_max + u_min) / 2.
            else:
                # normalization that keeps signs
                u_α = np.amax(u)
                u_β = 0.

        elif norm == False:
            # do standarlization
            u_α = np.std(u, axis=0)
            u_β = np.mean(u, axis=0)
    else:
        # nothing happens
        u_α = 1.
        u_β = 0.
    return np.float32(u_α), np.float32(u_β)

"""
convert to relative values
"""


def convert_rela_df(df, df_init):
    clmns = df.columns.tolist()
    df_rela = df.copy()

    for clmn in clmns:
        df_rela[clmn] = df_rela[clmn].apply(lambda x: df_init[clmn].iloc[0]-x)

    return df_rela


"""
normalize the values
"""


def stnd_nrml(df, set_norm=True, set_norm_01=False):
    # set_norm=True, set_norm_01=False -> normalization, rescale to [-1,1]
    # set_norm=True, set_norm_01=True -> normalization, rescale to [0,1]
    # set_norm=False, set_norm_01=False/True -> standarlization to have a mean of 0 and standard deviation of 1
    
    clmns = df.columns.tolist()
    df_norm = df.copy()
    α_β_c = dict.fromkeys(clmns)

    for clmn in clmns:

        # option 1, rescale to (0-1)
        # α_c, β_c = get_α_β(df[clmn], norm_01=True)
        α_c, β_c = get_α_β(df[clmn], norm=set_norm, norm_01=set_norm_01)
        α_β_c[clmn] = [α_c, β_c]
        df_norm[clmn] = out_stnd_nrml(df[clmn], α_c, β_c)

        # option 2, rescale to (-1,1)

    return df_norm, α_β_c

def execute_linearsvc(test_rule, X_train, y_train, y_train_weight):

    # execute the linear svc
    svc = svm.SVC(kernel='linear', probability=True)
    svc.fit(X_train, y_train, y_train_weight)
    y_pred = svc.predict(X_train)
    supp = svc.support_vectors_
    print("\n svc.score:", svc.score(X_train, y_train, y_train_weight), "\n")

    # display those wrong predictions and the corresponding prediction probabilities
    y_pred_error_index = []
    for i, ys in enumerate(zip(y_train, y_pred.tolist())):
        if ys[0] != ys[1]:
            y_pred_error_index.append(i)
            print(
                f"\n Wrong prediction - number:{str(i)}, y_train: {str(ys[0])}, y_pred: {str(ys[1])}")
            print("with prediction probabilities on {} and {}".format(svc.classes_[0], svc.classes_[
                  1]), "\n", ["{:0.3f}".format(x) for x in svc.predict_proba(X_train)[i, :]])

    # display those predicted True samples and the corresponding prediction probabilities
    # save the (predicted) valid samples according
    y_pred_val_index = []
    for i, ys in enumerate(zip(y_train, y_pred.tolist())):
        if ys[1]:
            y_pred_val_index.append(i)
#             print(
#                 f"\n Predicted True Sample - number:{str(i)}, y_train: {str(ys[0])}, y_pred: {str(ys[1])}")
#             print("with prediction probabilities on {} and {}".format(svc.classes_[0], svc.classes_[
#                   1]), "\n", ["{:0.3f}".format(x) for x in svc.predict_proba(X_train)[i, :]])

    # calculate the distance from samples to the hyperplane
    # here the y value converges with y_pred, but not alwadys with y_train
    y = svc.decision_function(X_train)
    w_norm = np.linalg.norm(svc.coef_)
    dist = y / w_norm

    return svc, y_pred, dist, y_pred_error_index, y_pred_val_index

"""
display the sv classification by principal components
"""


def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 0.1, x.max() + 0.1
    y_min, y_max = y.min() - 0.1, y.max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def display_svc_pc(X, y, svckernel="linear", nu_nu=0.05):

    pca = PCA(n_components=2)
    Xreduced = pca.fit_transform(X)
    print("explained variance ratio (first two components): %s" %
          str(pca.explained_variance_ratio_))

    if svckernel == "linear":
        model = svm.SVC(kernel='linear')
        clf = model.fit(Xreduced, y)
        plot_name = 'fig/' + test_rule + "_" + svckernel
    elif svckernel == "nu":
        clf = svm.NuSVC(nu=nu_nu, gamma="scale")
        clf.fit(Xreduced, y)
        plot_name = 'fig/' + test_rule + "_" + svckernel + "_" + str(nu_nu)

    fig, ax = plt.subplots(figsize=(10, 6))

    # set-up grid for plotting.
    X0, X1 = Xreduced[:, 0], Xreduced[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    # plot the contours between classifications
    plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)

    # plot scatters
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_ylabel('PC2')
    ax.set_xlabel('PC1')

    xticks = np.linspace(-1.0, 1.0, num=9)
    yticks = np.linspace(-1.0, 1.0, num=9)
    plt.xticks(xticks)
    plt.yticks(yticks)
    ax.set_title(
        f'Decision surface of SVC with the first two principal components on rule: {test_rule}')

    plt.savefig(plot_name + '.png', dpi=200)