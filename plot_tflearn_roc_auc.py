"""
=============================================================
Receiver Operating Characteristic (ROC) without cross validation
=============================================================

Receiver Operating Characteristic (ROC) metric to evaluate
classifier output quality without cross validation

ROC curves typically feature true positive rate on the Y axis, and false
positive rate on the X axis. This means that the top left corner of the plot is
the "ideal" point - a false positive rate of zero, and a true positive rate of
one. This is not very realistic, but it does mean that a larger area under the
curve (AUC) is usually better.

The "steepness" of ROC curves is also important, since it is ideal to maximize
the true positive rate while minimizing the false positive rate.

This example shows the ROC response of one dataset X_train,y_train, X_test,y_test
Taking all of these curves. This roughly shows how the
classifier output is affected by changes in the training data, and how
different the splits of training data and test data are from one another.

.. note::

    See also :func:`sklearn.metrics.auc_score`,
             :func:`sklearn.model_selection.cross_val_score`,
             :ref:`sphx_glr_auto_examples_model_selection_plot_roc.py`,

"""
print(__doc__)

import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

#force matplotlib to not use any xWindows backend
#import matplotlib
#matplotlib.use("Agg")

# #############################################################################
#  plot the ROC curve graph if drawplot=True
# otherwise, compute and return ROC_AUC value from y_test( contains label) and y_probas_( contains probability of 2 classes)
###############################################################################
def plot_tflearn_ROC(y_test,y_probas_,title,fig,nrow, ncol, plot_number,annotate=False,drawplot=False):
    ''' plot ROC curve and compute AUC
    plot the ROC curve in a one plot with dimension of nrow*ncol at  position plot_number only if drawplot=True'''
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)



    #y_probas_ = classifier.fit(X_train, y_train).predict_proba(X_test)
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y_test, y_probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)

    if drawplot == False:
        return roc_auc

    axn = fig.add_subplot(nrow, ncol, plot_number)
    axn.plot(fpr, tpr, lw=2, alpha=0.3,
             label='ROC (AUC = %0.4f)' % roc_auc)
    # annotate the thresholds on this fold
    if annotate:
        for j in xrange(len(thresholds)):
            axn.annotate(str(thresholds[j]), xy=(fpr[j], tpr[j]), xycoords='data',
                         xytext=(-20, 10), textcoords='offset points',
                         arrowprops=dict(facecolor='black', arrowstyle='->'),
                         horizontalalignment='right', verticalalignment='bottom')

    axn.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Luck', alpha=.8)

    # mean_tpr = np.mean(tprs, axis=0)
    # mean_tpr[-1] = 1.0
    # mean_auc = auc(mean_fpr, mean_tpr)
    # std_auc = np.std(aucs)
    # plt.plot(mean_fpr, mean_tpr, color='b',
    #          label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
    #          lw=3, alpha=.8)
    #
    # std_tpr = np.std(tprs, axis=0)
    # tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    # tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    # plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
    #                  label=r'$\pm$ 1 std. dev.')

    axn.set_xlim([-0.05, 1.05])
    axn.set_ylim([-0.05, 1.05])
    axn.set_xlabel('False Positive Rate')
    axn.set_ylabel('True Positive Rate')
    axn.set_title('ROC of ' +  title )
    axn.legend(loc="lower right")
    return roc_auc

#end of plotROC

# #############################################################################
# function call example using iris data
#  Data IO and generation
def main():
    # Import some data to play with
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    # remove data with label y==2, only left data with label = 0 or 1,
    # since roc_curve function can only handle _binary_clf_curve
    X, y = X[y != 2], y[y != 2]
    n_samples, n_features = X.shape

    # Add noisy features
    random_state = np.random.RandomState(0)
    X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    classifier = svm.SVC(kernel='linear', probability=True,
                         random_state=random_state)


    fignum = plt.figure("total ROC window name",figsize=(10,8))
    fignum.subplots_adjust(top=0.92, left=0.10, right=0.97, hspace=0.37, wspace=0.3)
    print fignum
    # Run classifier  and plot ROC curves
    probas_ = classifier.fit(X_train, y_train).predict_proba(X_test)
    roc_aucSVC = plot_tflearn_ROC(y_test,probas_,"svm",fignum,2,1,1,drawplot=True)
    print 'ROC_AUC of %s is %0.4f' %(str(classifier),roc_aucSVC)

    dtClf= DecisionTreeClassifier()
    probas_ = dtClf.fit(X_train, y_train).predict_proba(X_test)
    roc_aucDT= plot_tflearn_ROC(y_test,probas_,"DecisionTree",fignum,2,1,2,drawplot=True)
    print 'ROC_AUC of %s is %0.4f' % (str(dtClf),roc_aucDT)
    # uncomment the following sentence if you want to see the plot
    plt.show()  #show all the graphs now.


if __name__=="__main__":
    main()
