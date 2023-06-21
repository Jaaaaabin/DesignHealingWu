#
# funct_svm.py
# 

# import packages
from base_external_packages import *

# import modules
from funct_data import *

def out_stnd_nrml(u, α, β):
    """
    Standardize/normalize
    """

    output = (u - β) / α
    return output


def u_stnd_nrml(output, α, β):
    """
    Un-standardize/un-normalize
    """

    u = output * α + β
    return u

def get_α_β(u, norm=True, norm_01=False, no=False):
    """
    Get the coefficients of normalization 
    norm=True: if norm==False, no normalization.. go for standardization
    norm_01=False: if norm_01==True, rescale to [0,1]; if norm_01==False, rescale to [-1,1]
    no=False: if no==True, u_α = 1. u_β = 0., no conversion
    """

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


def stnd_nrml(df, set_norm=True, set_norm_01=False):
    """
    normalize the values
    """
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

def executeLinearSVC(X_train, y_train, C=1.0, y_train_weight=[]):

    # execute the linear svc
    if y_train_weight:
        model = svm.SVC(kernel="linear", C=C, probability=True, class_weight={1: int(y_train_weight)})
    else:
        model = svm.SVC(kernel="linear", C=C, probability=True)
    
    clf = model.fit(X_train, y_train)
    
    return clf

def evaluateLinearSVC(clf, X_test, y_test):

    y_pred = clf.predict(X_test)

    print ("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print ("Precision:", metrics.precision_score(y_test, y_pred))
    print ("Recall:", metrics.recall_score(y_test, y_pred))

    supp = clf.support_vectors_
    y_pred_error_index, y_pred_val_index = [], []
    for i, ys in enumerate(zip(y_test, y_pred.tolist())):
        if ys[0] != ys[1]:

            # display those wrong predictions and the corresponding prediction probabilities.
            y_pred_error_index.append(i)
            print(f"\n Wrong prediction - number:{str(i)}, y_test: {str(ys[0])}, y_pred: {str(ys[1])}")
            print("with prediction probabilities on {} and {}".format(
                clf.classes_[0], clf.classes_[1]), "\n", ["{:0.3f}".format(x) for x in clf.predict_proba(X_test)[i, :]])
        elif ys[0] == ys[1]:

            # display those predicted True samples and the corresponding prediction probabilities
            y_pred_val_index.append(i)
            print(f"\n Predicted True Sample - number:{str(i)}, y_test: {str(ys[0])}, y_pred: {str(ys[1])}")
            print("with prediction probabilities on {} and {}".format(
                clf.classes_[0], clf.classes_[1]), "\n", ["{:0.3f}".format(x) for x in clf.predict_proba(X_test)[i, :]])

    return y_pred_error_index, y_pred_val_index


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

def displaySVC(clf, X, y):

    fig, ax = plt.subplots(figsize=(10, 6))

    # set-up grid for plotting.
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    # plot the contours between classifications
    plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)

    # plot scatters
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_ylabel('X0')
    ax.set_xlabel('X1')

    xticks = np.linspace(-1.0, 1.0, num=9)
    yticks = np.linspace(-1.0, 1.0, num=9)
    plt.xticks(xticks)
    plt.yticks(yticks)
    ax.set_title(
        f'Decision surface of SVC with the first two principal components')

    plt.savefig('test.png', dpi=200)


def displaySVCinPC(X, y, svckernel="linear", nu_nu=0.05):

    pca = PCA(n_components=2)
    X_pc = pca.fit_transform(X)
    print("explained variance ratio (first two components): %s" %
          str(pca.explained_variance_ratio_))

    if svckernel == "linear":
        model = svm.SVC(kernel='linear')
        clf = model.fit(X_pc, y)
        plot_name = svckernel
    elif svckernel == "nu":
        clf = svm.NuSVC(nu=nu_nu, gamma="scale")
        clf.fit(X_pc, y)
        plot_name = svckernel + "_" + str(nu_nu)

    fig, ax = plt.subplots(figsize=(10, 6))

    # set-up grid for plotting.
    X0, X1 = X_pc[:, 0], X_pc[:, 1]
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
        f'Decision surface of SVC with the first two principal components')

    plt.savefig(plot_name + '.png', dpi=200)


# old
def convert_rela_df(df, df_init):     
    """
    convert to relative values
    """

    clmns = df.columns.tolist()
    df_rela = df.copy()

    for clmn in clmns:
        df_rela[clmn] = df_rela[clmn].apply(lambda x: df_init[clmn].iloc[0]-x)

    return df_rela
