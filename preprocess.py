from sklearn.decomposition import PCA           #try pca
from sklearn.preprocessing import MinMaxScaler  #try sklearn normalizer by libo
from sklearn.preprocessing import StandardScaler  #try sklearn normalizer by libo
import sklearn.utils.validation as val

import numpy as np

#below import are from self written .py
from utility import log

#pay special attention to below list. this version includes stockcode as a feature, if you remove stockcode,need readjust
midNormalizedColumnids = [0,1,42,43,44,45,46,47,68,69,70,71,72,73,74,75,76,77,78,79,80,81,\
                          82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,108,109,110,111,112,113,114,115,116,117,138,139,140,141,142,\
                          148,149,150,151,152,153,154,155,156,157,158,159,161,162,163,164,165]


class MidRangeScaler(object):
    """apply midrange transform to the specified columns of training data X,
            return the transformed X and the scaler to be applied with test/validation data
             midrange=(minX+maxX)/2
             fullrange=maxX-minX
             scaled = (x-midrange)/(fullrange/2)

             scaled the X data into [-1,1] range
    """
    def __init__(self,copy=True):
        self.copy = copy

    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary.

        __init__ parameters are not touched.
        """

        # Checking one attribute is enough, becase they are all set together
        # in fit
        if hasattr(self, 'data_min_'):
            del self.n_samples_seen_
            del self.data_min_
            del self.data_max_
            del self.midrange
            del self.fullrange2





    def fit(self, X): # compute the scaler for all columns for simplicity including required and not required columnid
        """Compute the minimum and maximum to be used for later scaling.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data used to compute the per-feature minimum and maximum
            used for later scaling along the features axis.
        """

        # Reset internal state before fitting
        self._reset()
        return self.partial_fit(X)

    def partial_fit(self, X):
        """Online computation of midrange and fullrange2 on X for later scaling.
        All of X is processed as a single batch. This is intended for cases
        when `fit` is not feasible due to very large number of `n_samples`
        or because X is read from a continuous stream.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.
        """
        X = val.check_array(X, copy=self.copy, warn_on_dtype=True,
                        estimator=self, dtype=val.FLOAT_DTYPES)



        cmin = np.min(X, axis=0)
        cmax = np.max(X, axis=0)

        # for col in columnids:
        #     cmax = np.max(X[:, col], axis=0)
        #     cmin = np.min(X[:, col], axis=0)

        # First pass
        if not hasattr(self, 'n_samples_seen_'):
            self.n_samples_seen_ = X.shape[0]
        # Next steps
        else:
            cmin = np.minimum(self.data_min_, cmin)
            cmax = np.maximum(self.data_max_, cmax)
            self.n_samples_seen_ += X.shape[0]


        self.data_min_ = cmin
        self.data_max_ = cmax

        midrange = (cmax + cmin) / 2
        fullrange2 = (cmax - cmin) / 2

        self.midrange= midrange
        self.fullrange2=fullrange2

        return self



    def transform(self,X,columnids):  # only apply midrange scaler to those specified columns
        val.check_is_fitted(self, 'midrange')

        X = val.check_array(X, copy=self.copy, dtype=val.FLOAT_DTYPES)

        for col in columnids:
            X[:, col] = (X[:, col] - self.midrange[col]) / self.fullrange2[col]
        return X

#move this clause out of   _init_ of DataPreprocess class  to avoid creating multiple instances of 3 scaler classes
dictMap = {'MinMax': MinMaxScaler(), 'Standard': StandardScaler(), 'MidRange': MidRangeScaler()}

class DataPreprocess(object):
    '''
    :param preScalerString is the abbreviation name of the the scaler in string format
    '''
    def __init__(self,preScalerString):

        assert preScalerString in ('MinMax', 'Standard', 'MidRange')

        self.preScaler = dictMap[preScalerString]  # point to one of the  instance of a preprocess scaler class

        self.preScalerClassName = self.preScaler.__class__.__name__  # get classname as part of the title in ROC plot for readability

        self.dataScaler = None


    def fit(self,data_set):
        '''
        :param data_set to be used
        :return: dataScaler
        '''
        # my dataScaler does not need additional parameters. I just calculate scaler for all columns but will only apply
        # those to required columns in transform


        self.dataScaler = self.preScaler.fit(data_set.data)
        #log('the data range of features are %s' % self.dataScaler.data_range_)


        # pca = PCA(n_components=150)
        # pca.fit(X)
        # X = pca.transform(X)

        # print ("pca.explained_variance_ratio_ and pca.explained_variance_ are:")
        # print(pca.explained_variance_ratio_)
        # print(pca.explained_variance_)


        return self.dataScaler

    def transform(self,data_set):
        if self.preScalerClassName == 'MidRangeScaler':  # special handle since it needs additional input params.
             # my dataScaler needs additional parameters.
            X = self.dataScaler.transform(data_set.data, midNormalizedColumnids)
            #log('the scaler of my MidRangeScaler are %s%s' % (self.dataScaler.midrange, self.dataScaler.fullrange2))

        elif self.preScalerClassName == 'MinMaxScaler' or self.preScalerClassName == 'StandardScaler':

            X = self.dataScaler.transform(data_set.data)
        else:
            raise ValueError("invalid proprecess keyword,must be one of MinMax,Standard,MidRange")

        return X




