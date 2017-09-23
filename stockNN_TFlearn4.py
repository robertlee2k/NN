
'''
examples that use tflearn high layer API to implement DNN classifier with StockNN network structure pre-verified with irisdata

previous version is stockNN_loadIris_TFLearn.py

updated on 2017-9-22 process data format for data 201101-201612

'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tflearn
import time
import os
import numpy as np
import tensorflow as tf

from sklearn.decomposition import PCA           #try pca
from sklearn.preprocessing import MinMaxScaler  #try sklearn normalizer by libo
from sklearn.preprocessing import StandardScaler  #try sklearn normalizer by libo

from tensorflow.python.platform import gfile
import csv


# Data sets



TRAINDATASTART = 1      # the row# of the beginning of training data
TRAINDATASTOP = 3000000  # the row# of the end of the training data record  2013-2015csv file
TESTDATASTART=1     # the row# of the starting in test csv file 2016-2017
TESTDATASTOP=159273  # the last row of the whole file, this row# is excluded in test data

TrainDataStart = 1
TrainDataStop = TRAINDATASTOP  #for debugging purpose ,you can adjust this to get a small part for time saving now
TestDataStart = 1 #350603
TestDataStop = TESTDATASTOP # TESTDATASTOP     # for debugging purpose ,you can adjust this to get a small part for time saving now


trainfilename = "/home/topleaf/stock/tensorFlowData/tensorFlowData(201101-201612).csv"   #training data file
testfilename = "/home/topleaf/stock/tensorFlowData/tensorFlowData(201701-201709).csv"  # test data  file


#modelfilename = '201315custMidrangescaleTrainModel'  # trained model dump to disk

logfilename = "/tmp/stockNN2.log"

EXPORT_DIR = "/tmp/savedstockmodel/"  #export dir base

testResultfile="DNN_Training_results.csv"

# the following definition specified the column id# in original csv data file,starting from 0

DataDateColumn = 4

midNormalizedColumnids = [0,41,42,43,44,45,46,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,\
                          82,83,84,85,86,87,88,89,90,91,92,93,94,95,137,138,139,140,141,\
                          147,148,149,150,151,152,153,154,155,156,157,158,160,161,162,163,164]


import datetime
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from plot_tflearn_roc_auc import plot_tflearn_ROC #for plotting ROC curve

from sklearn.metrics import roc_curve,auc

class TestResult(object):
    def __init__(self,filename):
        self.columnsname=["RunId","PreProcessor","AUC(Test)","AUC(Train)","Loss","TestAccuracy",\
                          "Duration","StartTime","EndTime","Epoch","Minibatch","Optimizer""ROC Curve Location"]
        self.filename=filename
        if os.path.exists(filename)==False :
            with open(filename,'w') as csv_file:
                writer=csv.writer(csv_file)
                writer.writerow(self.columnsname)
    def append(self,rows):
        if os.path.exists(self.filename):
            with open(self.filename,'ab+') as csv_file:
                writer=csv.writer(csv_file)
                writer.writerow(rows)








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

    def transform(self,X,columnids):
        for col in columnids:
            X[:, col] = (X[:, col] - self.midrange[col]) / self.fullrange2[col]
        return X


#get roc_auc for predicted data with label =1
def get_roc_auc(y_true,y_pred_prob):
    fpr,tpr,thresholds= roc_curve(y_true,y_pred_prob[:,1])
    roc_auc = auc(fpr,tpr)
    return roc_auc


def printConfusionMatrix(cm):
    """

    :param cm: confusion matrix    the value of cm[i,j] is the number that a trueclass i is predicted to be class j
                          predict=0  predict=1 .... predict=n_class
       truelabel=0
       truelabel=1
       ...
       truelabel=n_class

    :return: a string of above format
    """
    output="\n               "
    row,col= cm.shape
    for j in range(col):
        output=output+"  predict=%d" %j
    output+="\n"   # print first title line
    for i in range(row):
        output += "truelabel=%d" %i
        for j in range(col):
            output +="  % 9d" %cm[i][j]
        output+="\n"
    return output

import collections
Dataset = collections.namedtuple('Dataset',['data','target','featurenames'])

#write info to log file
def log(info,logfilename=logfilename):
    with gfile.Open(logfilename,'a+') as logfileid:
        logfileid.write(info)
        print ("%s" %info)

# def appendRunResult(filename,onerow):
#     with gfile.Open(filename, "a+") as csvfile:
#         writer = csv.writer(csvfile)
#
#         # columns_name
#         writer.writerow(["index", "a_name", "b_name"])
#         # writerows
#         writer.writerows([[0, 1, 3], [1, 2, 3], [2, 3, 4]])
#     pass


def load_partcsv_without_header(filename,
                            target_dtype,
                            features_dtype,
                            start_rowid,
                            end_rowid,
                            fromDate,
                            toDate,
                            discard_colids,
                            target_column = -1,
                            filling_value=1.0):
  """Load dataset from CSV file without a header row to indicate how many records in this files. the format of the csv is:
  row 0:  column names
  row 1 ~ n: column  data

  fromDate format: '2017/3/13'
  """
  with gfile.Open(filename) as csv_file:
    data_file = csv.reader(csv_file)
    data, target = [], []
    dest_rowid = 0
    missingvaluecount=0
    hitCount=0      # the total row number that has been read into data
    featureNames = []  # a list to hold all required feature names

    stDate = datetime.datetime.strptime(fromDate, "%Y/%m/%d")
    endDate = datetime.datetime.strptime(toDate, "%Y/%m/%d")

    for i,row in enumerate(data_file):
        # check the content of the row and print out if there's  missing value
        for k in xrange(0, row.__len__()):
            if row[k] == '?':
                #log("\n ? value in record id:%s,(stockcode = %s) column#=%d" % (row[0], row[2], k+1 ))
                row[k]=filling_value  # setting default value to those missing values
                missingvaluecount+=1
        if i==0: # put the required feature names into a list
            row.pop(target_column)                          #discard the target_column name
            for j in sorted(discard_colids,reverse=True):  # delete the columns whose indexes are in list discard_colids
                del row[j]
            featureNames.extend(row)   #add multiple values at once from  row list to my featurename list

        if i < start_rowid:  # or i > end_rowid:
            continue    #skip the rows between start_rowid and end_rowid
        elif i > end_rowid:
            log("\n WARNING: skipping rows in %s after line# %d, NOT FULL data are loaded" %(filename,i))
            break  # skip reading the rest of the file
        else:
            dt = datetime.datetime.strptime(row[DataDateColumn],"%Y/%m/%d") # convert DataDate from string to datetime format
            if  dt >= stDate  and  dt <= endDate:  #read the desired data between 'From' and 'To' date
                target.append(row.pop(target_column))
                for j in sorted(discard_colids,reverse=True):   #delete the columns whose indexes are in list discard_colids
                    del row[j]
                data.append(row)
                hitCount+=1
            #else:
                #print ('attention:  this row[%d] dataDate %s is not between [%s,%s],so discard this row' %(i, row[DataDateColumn],fromDate,toDate))

    log ('\ntotal row# read from this data file %s is (%d out of %d)' %(filename,hitCount,i))
    if missingvaluecount != 0:
        log("\n!!! WARNING: the input data file %s has %d blank value(s),setting its value to %d as a workaround, please check and fill them later!!!" % (filename, missingvaluecount,filling_value))

    data=np.array(data,dtype=features_dtype)
    target=np.array(target,dtype=target_dtype)

  return Dataset(data=data, target=target,featurenames=featureNames)


# visualize  the batch features in both a scatter subplot to review the min-max range of the features in a whole picture
# and a subplot of histogram for each features' distribution

def plotFeatures(batch,datasetFeatureNames,desc=None,savePlotToDisk=True,scatterAdjust=False):
    """
batch: is 2-D matrix [n_sample,n_feature]
datasetFeatureNames: a list that contains all the feature names in the same order as batch
desc is the string that describes the data,such as "2013-15training" ,will be used as part of plot title.
savePlotTodisk : whether or not save the plots to disk
scatterAdjust : whether scale Y scale to fit for current min/max , set it to False if you want to compare all features'
absolute min/max value in one scatter plot
    """
    xscatter=[]   # hold the x axis coordinates, which is column id#
    yscatter=[]   # hold a tuple with (minvalue,maxvalue) for that x
    n_sample,n_feature= batch.shape

    # get and fill  min and max value for each features,prepare data for scatter plotting later
    for j in range(n_feature):
        y = batch[:, j]  # fetch all items in this column
        xscatter.append(j)
        yscatter.append((y.min(),y.max()))
        # xscatter.append(j)
        # yscatter.append(y.max())


    for j in range(n_feature): # for all features
        log("Plotting %s feature# %d in progress,time = %s " %(desc,j,(time.ctime())))
        featureFig = plt.figure("plotFeature"+str(j)+"_"+str(datasetFeatureNames[j]), figsize=(8, 6))
        featureFig.subplots_adjust(top=0.92, left=0.10, right=0.97,hspace=0.37, wspace=0.3)

        axf = featureFig.add_subplot(2, 1, 1)  # histogram each feature to check its distribution
        y= batch[:,j]  #fetch all items in this column

        axf.clear()
        #axn.clear()   #clear previous histogram plot
        # bins=[]    #prepare a list to seperate data into 1000 equal groups
        # stepsize=(y.max()-y.min())/1000
        # for k in range(0,1001):   #show distribution of 1000 equal parts 1000+1 to show the y.max()
        #     bins.append(y.min()+stepsize*k)
        #
        # n,bins,patchs=axn.hist(y,bins,histtype='bar',rwidth=0.8,label=str(datasetFeatureNames[j]))
        # axn.set_xlabel( 'feature value range: (%0.4f - %0.4f)' %(y.min(),y.max()))
        # axn.set_ylabel( 'number')
        # #axn.set_xlim(y.min(),y.max())
        # #axn.set_ylim()
        # axn.set_title('histograms of feature id# '+ str(j)+",name="+str(datasetFeatureNames[j]))
        # axn.legend()

        #histogram of the data in 1000 pieces ??? this histogram is weird , display id#1 seems to be wrong and misleading
        # change num_bins to 500 to solve the problem.
        num_bins=500
        n,bins,patches = axf.hist(y,num_bins,normed=0,label=str(datasetFeatureNames[j]))
        axf.legend()

        #add a  'best fit' line
        mu=np.mean(y)
        sigma=np.std(y)   #standard deviation
        ybestfit= mlab.normpdf(bins,mu,sigma)
        axf.plot(bins,ybestfit,'--')
        axf.set_xlabel("feature value range:[%0.4f-%0.4f],num_bins=%d" % (y.min(), y.max(),num_bins))
        axf.set_ylabel("count")
        axf.set_title('histograms of feature id# ' + str(j) + ",name=" + str(datasetFeatureNames[j])+ r',$\mu=%.2f$,$\sigma=%.4f$' %(mu, sigma))
        #
        #tweak spacing to prevent clipping of ylabel
        #axf.tight_layout()


        axs = featureFig.add_subplot(2, 1, 2)  # show the scatter subplot for min,max value of all the features in a whole.
        axs.clear()

        # at first, convert [ (min1,max1), (min2,max2), ....(minn,maxn)] to [(min1,min2,...minn),(max1,max,...maxn)]
        ycoord = zip(*yscatter)
        for i,colors,names,markers in zip([0,1],['red','blue'],['min','max'],['x','o']):
            axs.scatter(xscatter, list(ycoord[i]), label=names, color=colors, s=25, marker=markers)



        axs.set_xlabel('feature id#')
        axs.set_ylabel('actual value ')
        axs.set_title("scatter min & max for "+ str(n_feature) +" features in "+ desc + " with " +str(n_sample) +" samples")

        if scatterAdjust==True:
            # adjust Y scale to show this column's min and max scatter point in the graph,at the cost of possibly sacrifice other columns
            if y.max()>0 :
                ymax=y.max()*1.2
            else:
                ymax=y.max()*0.8
            if y.min()>0 :
                ymin=y.min()*0.8
            else:
                ymin=y.min()*1.2
            axs.set_ylim(ymin,ymax)

        axs.annotate('Here:id =' + str(j)+' min:%s' %yscatter[j][0], xy=(j, yscatter[j][0]), xycoords='data',
                     xytext=(20, 30), textcoords='offset points',
                     arrowprops=dict(facecolor='black', arrowstyle='->'),
                     horizontalalignment='right', verticalalignment='bottom')
        axs.annotate('Here:id =' + str(j) + ' max:%s'%yscatter[j][1], xy=(j, yscatter[j][1]), xycoords='data',
                     xytext=(0.5, 0.85), textcoords='axes fraction',
                     arrowprops=dict(color='blue', arrowstyle='->'),
                     horizontalalignment='right', verticalalignment='bottom')
        axs.legend()

        #plt.show()
        if savePlotToDisk:
            plt.savefig(desc+"feature"+str(j)+"_"+str(datasetFeatureNames[j])+".png", figsize=(8, 6))
        plt.close(featureFig)  #close figures explicitly to release memory

        # key_resp= raw_input("please press any key with a Enter to plot next feature, type 'exit' to quit plotting")
        # if key_resp=='exit':
        #     print('Exit plotting features...')
        #     break



    # for i in range(len(batch)):
    #     flat = np.reshape(batch[i], batch[i].size)  #get row by row

def myNormalizer(inputs):
     """Normalized the input data."""
     output = []
     output=MinMaxScaler().fit_transform(inputs)
     return output

#customized data preprocess function
# for each feature, normalizedvalue= (xi- expectvalue(x))/(sigma of x +0.001)
def myPreprocess(batch):
    #for i in range(len(batch)):
#       batch[i] -= np.mean(batch[i], axis=0) #samplewise_zero_center

    mean=np.mean(batch,axis=0)
    stdX=np.std(batch,axis=0)+0.001  # add a small number to prevent dividing 0
    batch = (batch-mean)/stdX
    # batch=(batch-mean)/((np.max(batch,axis=0)-np.min(batch,axis=0))+0.001)

    return batch



def main():
    # from sklearn.externals import joblib

  startTime=time.time();  # start time in ms.
  st=time.ctime()  #start time in date/time format
  log('\nloading training data from file %s in progress ... time:%s' %(trainfilename,st) ,logfilename)
  #Load datasets. discard column 0,1,3,4 in the original csvfile ,which represent  id ,tradedate,mc_date,datadate

  # Load datasets.

  training_set = load_partcsv_without_header(
      filename=trainfilename,
      target_dtype=np.int,
      features_dtype=np.float32,
      start_rowid=TrainDataStart,
      end_rowid=TrainDataStop,
      fromDate='2011/01/01',
      toDate='2016/12/29',
      discard_colids=[0,1,2,3,4,-1],  #add -1 in the last column to exclude the percentage infor 2# stockcode,
      target_column=5,
      filling_value=1.0
  )
# plot original data to review

  #plotFeatures(training_set.data,training_set.featurenames,"Orig13-15train500",savePlotToDisk=True,scatterAdjust=False)


  # added by libo : declare and store this scaler, and use the same one to scale the test data
  preScaler=StandardScaler()
  #preScaler=MinMaxScaler()
  #preScaler=MidRangeScaler()
  preScalerClassName = preScaler.__class__.__name__  # get classname as part of the title in ROC plot for readability
  dataScaler = preScaler.fit(training_set.data)
  # log('the data range of features in training set are %s' % dataScaler.data_range_)
  #dataScaler = preScaler.fit(training_set.data,midNormalizedColumnids)   # my dataScaler needs additional parameters.
  #X = dataScaler.transform(training_set.data,midNormalizedColumnids)    # my dataScaler needs additional parameters
  #log('the scaler of my MidRangeScaler are %s' %(dataScaler.midrange,dataScaler.fullrange2))


  X = dataScaler.transform(training_set.data)
  #log('the scaler factors got in training set are %s' % dataScaler.scale_)

  #customized midrangestandard  normolize those specified columns only
  #midr=np.zeros(training_set.data.shape[1])
  #fullr=np.zeros(training_set.data.shape[1])
  #X = midrangeStandardization(training_set.data, midNormalizedColumnids, isTraining=True, midrange=midr, fullrange2=fullr)



  # pca = PCA(n_components=150)
  # pca.fit(X)
  # X = pca.transform(X)

  # print ("pca.explained_variance_ratio_ and pca.explained_variance_ are:")
  # print(pca.explained_variance_ratio_)
  # print(pca.explained_variance_)

  #plotFeatures(X,training_set.featurenames,"custmidrange13-15train500",savePlotToDisk=True,scatterAdjust=False)
  #plotFeatures(X, training_set.featurenames, "minmaxscale13-15train500",savePlotToDisk=True,scatterAdjust=False)
  y = training_set.target
    #[training_set.target != 2]




  #log ('the first 20 label of the training set are %s' %y[0:21])

  #log('\nloading test data from file %s in progress ... time:%s' % (testfilename, time.ctime()),logfilename)


  test_set= load_partcsv_without_header(
      filename=testfilename,
      target_dtype=np.int,
      features_dtype=np.float32,
      start_rowid=TestDataStart,
      end_rowid=TestDataStop,
      fromDate='2016/12/30',
      toDate='2017/9/19',
      discard_colids=[0,1,2,3,4,-1],  #add -1 in the last column to exclude the percentage infor  2# stockcode
      target_column=5,
      filling_value=1.0
  )




    # plot original test data to review
  #plotFeatures(test_set.data,test_set.featurenames,"Orig16-17test500",savePlotToDisk=True,scatterAdjust=False)
  # the following 2 sentences only get the class 0,1 DATA,discard data with class=2.
 #added by libo
    # apply the SAME datascaler extracted from training data to test data, implicitly assuming they have same distribution(mu & sigma)
  X_test = dataScaler.transform(test_set.data)
  #X_test = dataScaler.transform(test_set.data,midNormalizedColumnids)  # my dataScaler needs additional parameters
    #[test_set.target != 2, :]

  #plotFeatures(X_test, test_set.featurenames, "minmaxscale16-17test ",False)

  # apply midrange normalization to test data using the same midr and fullr got from training dataset
  #X_test = midrangeStandardization(test_set.data, midNormalizedColumnids, isTraining=False, midrange=midr,fullrange2=fullr)

  #plotFeatures(X_test,test_set.featurenames,"cust_midrange16-17test500",savePlotToDisk=True,scatterAdjust=False)
  # try pca
  # X_test = pca.transform(X_test)

  y_test = test_set.target
    #[test_set.target != 2]
  log('the first 20 label of the test set are %s' % y_test[0:21])




  log('\ndata loading completed, building graph in progress ... time:%s' % (time.ctime()),logfilename)  # train a 3-hidden layer neural network

  tf.reset_default_graph()
  # train a 4-hidden layer neural network with 80 nodes (161*80*80*80*80*2= 1318912000 weights, ?? of the training data
  with tf.Graph().as_default():
      # dpp= tflearn.data_preprocessing.DataPreprocessing(name="dataPreprocess")
      # dpp.add_custom_preprocessing(myNormalizer) #Mean: 3163.05 (To avoid repetitive computation, add it to argument 'mean' of `add_featurewise_zero_center`)
       epoch = 20000
       learningrate = 0.001
       minibatch = 16384 #8192
       lrdecay=0.999
       decaystep=100
       rs= 2 #25799 #39987 # # 186
       rng =np.random.RandomState(rs)
       seedweight=rng.uniform(0.001,1)
       log('seedweight=%0.4f' %seedweight)
       seedbias=rng.uniform(0.01,0.09)
       log('seedbias=%0.4f' %seedbias)
       xavierInit=tflearn.initializations.xavier(uniform=True,seed=seedweight)
       normalInit=tflearn.initializations.normal(seed=seedbias)

       net = tflearn.input_data([None, 165], data_preprocessing=None, data_augmentation=None, name="inputlayer")
       #try pca
       # net = tflearn.input_data([None, 150], data_preprocessing=None, data_augmentation=None, name="inputlayer")


       net = tflearn.fully_connected(net, 150, activation='relu', weights_init=xavierInit, bias_init=normalInit,
                             regularizer=None, weight_decay=0.001, name='hidderlayer1')
       net = tflearn.fully_connected(net, 150, activation='relu', weights_init=xavierInit, bias_init=normalInit,
                             regularizer=None, weight_decay=0.001, name='hidderlayer2')
       net = tflearn.fully_connected(net, 150, activation='relu', weights_init=xavierInit, bias_init=normalInit,
                             regularizer=None, weight_decay=0.001, name='hidderlayer3')
       net = tflearn.fully_connected(net, 150, activation='relu', weights_init=xavierInit, bias_init=normalInit,
                            regularizer=None, weight_decay=0.001, name='hidderlayer4')
       # net = tflearn.fully_connected(net, 80,  activation='sigmoid',name='hidderlayer5')
       #
       # net = tflearn.fully_connected(net, 10, activation='relu', weights_init='xavier',
       #                                      regularizer='L2', weight_decay=0.001, name='hidderlayer6')
       #
       #
       #  net = tflearn.fully_connected(net, 10, activation='relu', weights_init='xavier',
       #                                regularizer='L2', weight_decay=0.001, name='hidderlayer7')
       #  net = tflearn.fully_connected(net, 10, activation='relu', weights_init='xavier',
       #                                regularizer='L2', weight_decay=0.001, name='hidderlayer8')
       #  net = tflearn.fully_connected(net, 10, activation='relu', weights_init='xavier',
       #                                regularizer='L2', weight_decay=0.001, name='hidderlayer9')
       #  net = tflearn.fully_connected(net, 10, activation='relu', weights_init='xavier',
       #                                regularizer='L2', weight_decay=0.001, name='hidderlayer10')

       net = tflearn.fully_connected(net, 2, activation='softmax',weights_init=xavierInit, bias_init=normalInit,
                                     name='outputlayer')

       #Y = to_categorical(training_set.target, 3)

       opt = tflearn.Adam(learning_rate=learningrate)
       #opt = tflearn.Momentum(learning_rate=learningrate,lr_decay=lrdecay,decay_step=decaystep)
       #opt=tflearn.RMSProp(learning_rate=learningrate,decay=lrdecay,momentum=0.1)
       #opt = tflearn.SGD(learning_rate=learningrate, lr_decay=lrdecay, decay_step=decaystep)
       acc = tflearn.metrics.Accuracy()
       #topk = tflearn.metrics.Top_k(1)

       #net = tflearn.regression(net, optimizer=admopt, loss = 'roc_auc_score', metric=acc, \
       #                         to_one_hot=True, n_classes=2)
       #net = tflearn.regression(net, optimizer=opt, loss = 'binary_crossentropy', to_one_hot=True, n_classes=2)

       #net = tflearn.regression(net, optimizer=momentum, loss='categorical_crossentropy', metric=acc, \
       #                          to_one_hot=True, n_classes=2)
       net = tflearn.regression(net, optimizer=opt, loss='categorical_crossentropy', metric=acc, \
                            to_one_hot=True, n_classes=2)

       model = tflearn.DNN(net, tensorboard_dir="/tmp/tflearn_10thlogs/", tensorboard_verbose=0)

       #if you need to load a previous model with all weights,uncomment the following lines to do it
       # key=raw_input("\nDo you want to load previous trained model from disk ? \n 1. Load & predict \n2. skip loading\nPlease input your choice(1/2):")
       # if key=='1':
       #   if os.path.exists(modelfilename+".meta"):
       #      log('\nLoading previous model from file %s, restoring all weights' %modelfilename)
       #      model.load(modelfilename)
       #      keyp = raw_input("\nPlease input a row# in test file to predict its label: (q to quit predicting, e to exit )")
       #      while (keyp != 'q' and keyp!='e'):
       #          rowid=int(keyp)
       #          tmp=X_test[rowid:rowid+1,:]
       #          #tmp=tmp.reshape(tmp.shape[0],1)
       #
       #          pred= model.predict(tmp)
       #          print("predicted probability:%s" %pred)
       #
       #          print ("predicted y label = %s" %(np.argmax(pred, axis=1)))
       #          print ("true y_label = %s" %(y_test[rowid:rowid+1]))
       #
       #          keyp = raw_input("\nPlease input a row# in test file to predict: (q to quit predicting, e to exit )")
       #      if keyp=='q':
       #          log("\nload savedModel %s, continue to train" %modelfilename)
       #      elif keyp=='e':
       #          return -1  # exit predicting & training.
       #
       #   else:
       #      log("\n the modelfile %s doesn't exist in current folder,so skip it" %modelfilename)
       # else:
       #     log("\n Skip loading previous trained modelfile %s" %modelfilename)

       # preview outputlayer and hiddenlayer4's weight
       #outlayer_var= tflearn.get_layer_variables_by_name("outputlayer")
       #log ("\noutput layer weight:")
       #log(outlayer_var[0])
       #log ("\noutputlayer bias:")
       #log (outlayer_var[1])

       #h4layer_var= tflearn.get_layer_variables_by_name("hidderlayer4")
       #log("\nhiddenlayer 4 weight:")
       #log(h4layer_var[0])

       #model.get_weights(net.W)
       log('\ntraining the DNN classifier using %s (rs=%d,alpha=%0.6f,decayrate=%0.4f,decaystep=%d) for %d epoches with mini_batch \
           size of %d in progress ... time:%s' \
           % (opt.name,rs,learningrate,lrdecay,decaystep,epoch,minibatch,time.ctime()))
       runId=tflearn.utils.id_generator()
       modelfilename = "2011-16%s%s_%s_alpha%0.4f_lrdecay_%0.2f_decaystep%d_epoch%d_batch%d_TrainedModel" \
                    % (preScalerClassName,runId, opt.name, learningrate, lrdecay, decaystep, epoch, minibatch)
       model.fit(X, y, validation_set= (X_test,y_test),shuffle=True,show_metric=True,run_id=runId,\
                 batch_size=minibatch,n_epoch=epoch,snapshot_epoch=False,snapshot_step=10000)

      # save the model to disk
       fullpath=''.join((EXPORT_DIR, runId))
       if os.path.isfile(fullpath):
           log("model file %s exists, do not overwrite it!!!!" %fullpath)
       elif os.path.isdir(fullpath)==False:
           os.mkdir(fullpath)
           fullpath=''.join((EXPORT_DIR, runId, '/', modelfilename))
           log('\ntraining completed, save the model to disk as %s' % fullpath)
           model.save(fullpath)
       else:
           fullpath = ''.join((EXPORT_DIR, runId, '/', modelfilename))
           log('\ntraining completed, folder exists, overwrite it with new model as %s' % fullpath)
           model.save(fullpath)






  def evalprint(X_predict, y_true,title,fig,nrow,ncol,plot_number,annotate=False,drawplot=True):
        log('\nevaluate the DNN classifier %s in progress... time:%s' % (title,(time.ctime())))
        # print('\n%s inputX[0:20] is' %title)
        # print(X_predict[0:20])
        #print('\n%s y_true[0:20] is' %title)
        #print(y_true[0:20])
        predicted = model.predict(X_predict)



        # get the index of the largest possibility value of predicted list as the label of prediction
        verdictVector = np.argmax(predicted, axis=1)
        #print('\n%s verdictVector[0:20] is' %title)
        #print(verdictVector[0:20])
        #print('\n%s raw probability of predicted[0:20,:] are ' %title)
        #print(predicted[0:20, :])
        # log('\n%s             y_true counts:' % title)        # log(y_true.value_counts())

        aucValue=get_roc_auc(y_true, predicted)
        testAccuracy=np.mean(y_true == verdictVector)
        log('\n %s, The AUC value =%f ' %(title, aucValue))
        log('\n               Null Accuracy= {}%' .format(100*max(y_true.mean(),(1-y_true.mean()))))
        log("               Test Accuracy: {}%".format(100 * testAccuracy))




        # use sklearn function to print
        from sklearn import metrics
        expected = y_true
        log(metrics.classification_report(expected, verdictVector, labels=[0, 1],
                                          target_names=["predict=0", 'predict=1']))
        #print out confusion matrix
        cmstr=printConfusionMatrix(metrics.confusion_matrix(expected, verdictVector))
        log(cmstr)
        log("\n   evaluate model with %s completed, time: %s" % (title,time.ctime()))

        # plot the ROC curve, review the implementation to make sure roc compute algorithm without using
        # the following example sklearn predict_proba function is correct or not ??? ...
        # probas_ = classifier.fit(X_train, y_train).predict_proba(X_test)
        plot_tflearn_ROC(y_true, predicted, title, fig, nrow, ncol, plot_number, cmstr, annotate, drawplot)
        return(aucValue,testAccuracy)



  figid = plt.figure("ROC 2011-16Train201709Test Runid(%s) %s_%s" %(runId,preScalerClassName,opt.name),figsize=(10,8))
  figid.subplots_adjust(top=0.95, left=0.12, right=0.90,hspace=0.43, wspace=0.2)

  #evaluate the model with Training data
  trainAuc,trainTa =evalprint(X, y,"with Training data %d epoch,loss=%0.4f " %(epoch,model.trainer.training_state.global_loss),\
                              figid,2,1,1,False,True)

  # evaluate the model with Test data
  testAuc,testTa= evalprint(X_test, y_test, "with Test data %d minibatch,testAccuracy=%0.4f "
                            %(minibatch,model.trainer.training_state.best_accuracy), figid, 2, 1, 2, False, True)

  endTime = time.time()  # end time in ms.
  elapseTime = (endTime - startTime)
  hour = int(elapseTime / 3600)
  minute = int((elapseTime % 3600) / 60)
  second = int((elapseTime % 3600) % 60)
  duration = "%dh%d'%d''" % (hour, minute, second)
  log("\nthe WHOLE ELAPSED time of loading data and training the model is %s"
                % (duration))  # update test result  to file

  plotName="%s_ROC2011-16%sTrain_201709Test_%s_alpha%0.4f_epoch%d_%d.png" %(runId,preScalerClassName,opt.name,learningrate,epoch,minibatch)
  fullpath = ''.join((EXPORT_DIR, runId))
  if os.path.isfile(fullpath):
        log("file %s exists, do not overwrite it!!!!" % fullpath)
  elif os.path.isdir(fullpath) == False:
        os.mkdir(fullpath)
  fullpath = ''.join((EXPORT_DIR, runId, '/', plotName))
  plt.savefig(fullpath, figsize=(10, 8))




  # update test result  to file
  result = [runId, preScalerClassName, "%0.4f"%(trainAuc), "%0.4f"%testAuc, \
            "%0.4f" %model.trainer.training_state.global_loss,\
            "%0.4f" %model.trainer.training_state.best_accuracy,
            duration, "%s"%(st), "%s"%(time.ctime()),str(epoch),str(minibatch),opt.name,fullpath]

  trackRecord = TestResult(testResultfile)

  trackRecord.append(result)


  plt.show()  # display the ROC plot onscreen, if plot ROC is not needed, you must comment this line out!!!
  plt.close(figid)  #close it to release memory



  # keyp = raw_input("\nPlease input a row# to predict: (-1 to quit)")
  # while (keyp != '-1'):
  #     rowid=int(keyp)
  #     tmp=X_test[rowid,:]
  #     #tmp=tmp.reshape(tmp.shape[0],1)
  #     pred= model.predict(tmp)
  #     print ("predicted y label = %d, true y label is %d" %(np.argmax(pred, axis=1),y_test[rowid,:] ))
  #
  #     keyp = raw_input("\nPlease input a row# to predict: (-1 to quit)")
  # log("End of program")


#save the trained model to a file


  # create_feature_spec_for_parsing  returns  a  dict  mapping  feature  keys  from feature_columns to
  # FixedLenFeature or VarLenFeature  values.
  # feature_spec = create_feature_spec_for_parsing(feature_columns)
  #
  # serving_input_fn = input_fn_utils.build_parsing_serving_input_fn(feature_spec)
  # servable_model_dir = EXPORT_DIR
  # servable_model_path = classifier.export_savedmodel(servable_model_dir, serving_input_fn,as_text=True)


#export_savedmodel will save the model in a savedmodel format and return the string path to the exported directory.

# servable_model_path will contain the following files:

# saved_model.pb variables
#   log('\nthe model has been saved to %s' %(servable_model_path))

# trying to load the saved model to a new estimator and use it to predict new samples
#   loadTrainedClassifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
#                                               hidden_units=[10,20,10],
#                                               n_classes=3,
#                                               model_dir=servable_model_path,\
#                                               config=None
#                                               )
#   loadedVariablenames = loadTrainedClassifier.get_variable_names()
#   print ('the loaded variable names are: ')
#   print (loadedVariablenames)



if __name__ == "__main__":
    main()
