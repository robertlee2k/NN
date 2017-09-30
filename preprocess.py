from sklearn.decomposition import PCA           #try pca
from sklearn.preprocessing import MinMaxScaler  #try sklearn normalizer by libo
from sklearn.preprocessing import StandardScaler  #try sklearn normalizer by libo

import numpy as np

#below import are from self written .py
from utility import log

midNormalizedColumnids = [0,41,42,43,44,45,46,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,\
                          82,83,84,85,86,87,88,89,90,91,92,93,94,95,137,138,139,140,141,\
                          147,148,149,150,151,152,153,154,155,156,157,158,160,161,162,163,164]

class MidRangeScaler(object):
    """apply midrange transform to the specified columns of training data X,
            return the transformed X and the scaler to be applied with test/validation data
             midrange=(minX+maxX)/2
             fullrange=maxX-minX
             scaled = (x-midrange)/(fullrange/2)

             scaled the X data into [-1,1] range
    """
    def __init__(self):

        pass


    def fit(self,X,columnids):   # need to compute the scaler for all desired columnid
        self.midrange = np.zeros(X.shape[1])
        self.fullrange2 = np.zeros(X.shape[1])


        for col in columnids:
            cmax = np.max(X[:, col], axis=0)
            cmin = np.min(X[:, col], axis=0)
            self.midrange[col] = (cmax + cmin) / 2
            self.fullrange2[col] = (cmax - cmin) / 2

        return self

    def transform(self,X,columnids):  # need to double check this implementation, do NOT change input param
        for col in columnids:
            X[:, col] = (X[:, col] - self.midrange[col]) / self.fullrange2[col]
        return X



class DataPreprocess(object):
    '''
    :param preScalerString is the abbreviation name of the the scaler in string format
    '''
    def __init__(self,preScalerString):
        dictMap = {'MinMax': MinMaxScaler(), 'Standard': StandardScaler(), 'MidRange': MidRangeScaler()}

        self.preScaler = dictMap[preScalerString]  # create an instance of a preprocess scaler class

        self.preScalerClassName = self.preScaler.__class__.__name__  # get classname as part of the title in ROC plot for readability
        self.dataScaler = None


    def fit(self,data_set):
        '''
        :param data_set to be used
        :return: dataScaler
        '''

        if self.preScalerClassName == 'MidRangeScaler':  # special handle since it needs additional input params.
            self.dataScaler = self.preScaler.fit(data_set.data,
                                       midNormalizedColumnids)  # my dataScaler needs additional parameters.
            log('the scaler of my MidRangeScaler are %s%s' % (self.dataScaler.midrange, self.dataScaler.fullrange2))

        elif self.preScalerClassName == 'MinMaxScaler' or self.preScalerClassName == 'StandardScaler':
            self.dataScaler = self.preScaler.fit(data_set.data)
            #log('the data range of features are %s' % self.dataScaler.data_range_)

        else:
            raise ValueError("invalid proprecess keyword,must be one of MinMax,Standard,MidRange")

        # pca = PCA(n_components=150)
        # pca.fit(X)
        # X = pca.transform(X)

        # print ("pca.explained_variance_ratio_ and pca.explained_variance_ are:")
        # print(pca.explained_variance_ratio_)
        # print(pca.explained_variance_)

        # plotFeatures(X,training_set.featurenames,"custmidrange13-15train500",savePlotToDisk=True,scatterAdjust=False)
        # plotFeatures(X, training_set.featurenames, "minmaxscale13-15train500",savePlotToDisk=True,scatterAdjust=False)

        return self.dataScaler

    def transform(self,data_set):
        if self.preScalerClassName == 'MidRangeScaler':  # special handle since it needs additional input params.
             # my dataScaler needs additional parameters.
            X = self.dataScaler.transform(data_set.data, midNormalizedColumnids)
            log('the scaler of my MidRangeScaler are %s%s' % (self.dataScaler.midrange, self.dataScaler.fullrange2))

        elif self.preScalerClassName == 'MinMaxScaler' or self.preScalerClassName == 'StandardScaler':

            X = self.dataScaler.transform(data_set.data)
        else:
            raise ValueError("invalid proprecess keyword,must be one of MinMax,Standard,MidRange")

        return X




