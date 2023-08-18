#
# funct_svm.py
# 

# import packages
from base_external_packages import *

# import modules
from funct_data import *

def evaluateLinearSVC_decision(clf, X):

    y_dist = clf.decision_function(X)
    w_norm = np.linalg.norm(clf.coef_)
    y_dist /= w_norm

    return y_dist

def evaluateLinearSVC_prediction(clf, X, y):

    y_pred = clf.predict(X)
    
    # subset accuracy: the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true.
    print ("Accuracy:", metrics.accuracy_score(y, y_pred)) # Accuracy classification score.
    
    # The precision is intuitively the ability of the lassifier not to label as positive a sample that is negative.
    # The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives. 
    print ("Precision:", metrics.precision_score(y, y_pred))

    # The recall is intuitively the ability of the classifier to find all the positive samples.
    # The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives.
    print ("Recall:", metrics.recall_score(y, y_pred))


    # supp = clf.support_vectors_
    # y_pred_error_index, y_pred_val_index = [], []
    # for i, ys in enumerate(zip(y_test, y_pred.tolist())):
    #     if ys[0] != ys[1]:

    #         # display those wrong predictions and the corresponding prediction probabilities.
    #         y_pred_error_index.append(i)
    #         print(f"\n Wrong prediction - number:{str(i)}, y_test: {str(ys[0])}, y_pred: {str(ys[1])}")
    #         print("with prediction probabilities on {} and {}".format(
    #             clf.classes_[0], clf.classes_[1]), "\n", ["{:0.3f}".format(x) for x in clf.predict_proba(X_test)[i, :]])
    #     elif ys[0] == ys[1]:

    #         # display those predicted True samples and the corresponding prediction probabilities
    #         y_pred_val_index.append(i)
    #         print(f"\n Predicted True Sample - number:{str(i)}, y_test: {str(ys[0])}, y_pred: {str(ys[1])}")
    #         print("with prediction probabilities on {} and {}".format(
    #             clf.classes_[0], clf.classes_[1]), "\n", ["{:0.3f}".format(x) for x in clf.predict_proba(X_test)[i, :]])

    # return y_pred_error_index, y_pred_val_index



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

def displaySVC(X, y, path, rule_label, svckernel="linear"):

    fig, ax = plt.subplots(figsize=(10, 6))

    # set-up grid for plotting.
    X0 = X[:, 0]
    X1 = X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    model = svm.SVC(kernel=svckernel)
    clf = model.fit(X, y)
    plot_name = svckernel
    
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
    
    plot_name = path + '\\' + plot_name + '_{}.png'.format(rule_label)

    plt.savefig(plot_name, dpi=200)


def displaySVCinPC(X, y, path, rule_label, svckernel="linear", nu_nu=0.05):

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
    
    plot_name = path + '\\' + plot_name + '_{}.png'.format(rule_label)

    plt.savefig(plot_name, dpi=200)