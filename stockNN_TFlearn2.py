
'''
examples that use tflearn high layer API to implement DNN classifier with StockNN network structure pre-verified with irisdata

previous version is stockNN_loadIris_TFLearn.py
'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tflearn
import time
import os
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler  #try sklearn normalizer by libo
from sklearn.preprocessing import StandardScaler  #try sklearn normalizer by libo

from tensorflow.python.platform import gfile
import csv


# Data sets



TRAINDATASTART = 1      # the row# of the beginning of training data
TRAINDATASTOP = 559385  # the row# of the end of the training data record  2013-2015csv file
TESTDATASTART=1     # the row# of the starting in test csv file 2016-2017
TESTDATASTOP=352425  # the last row of the whole file, this row# is excluded in test data

TrainDataStart = 1
TrainDataStop = 559  #for debugging purpose ,you can adjust this to get a small part for time saving now
TestDataStart = 1 #350603
TestDataStop = 352 # TESTDATASTOP     # for debugging purpose ,you can adjust this to get a small part for time saving now


trainfilename = "/home/topleaf/stock/tensorFlowData/v_onceyield_group9all_2013-2015.csv"  # training data file
testfilename = "/home/topleaf/stock/tensorFlowData/v_onceyield_group9all_2016-2017.csv"  # test data  file
#"/home/topleaf/stock/tensorFlowData/short.csv"

modelfilename = 'stockNNmodel.pkl'  # trained model dump to disk
loaddt = False  # whether or not the trained modelfilename is loaded
logfilename = "/tmp/stockNN2.log"

EXPORT_DIR = "/tmp/serving_savedstockmodel"  #export dir base

# the following definition specified the column id# in original csv data file,starting from 0

DataDateColumn = 4

import datetime
import matplotlib.pyplot as plt

from plot_tflearn_roc_auc import plot_tflearn_ROC #for plotting ROC curve

from sklearn.metrics import roc_curve,auc
#get roc_auc for predicted data with label =1
def get_roc_auc(y_true,y_pred_prob):
    fpr,tpr,thresholds= roc_curve(y_true,y_pred_prob[:,1])
    roc_auc = auc(fpr,tpr)
    return roc_auc

import collections
Dataset = collections.namedtuple('Dataset',['data','target','featurenames'])

#write info to log file
def log(info,logfilename=logfilename):
    with gfile.Open(logfilename,'a+') as logfileid:
        logfileid.write(info)
        print ("%s" %info)




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
            if row[k] == '':
                #log("\nblank value in record id:%s,(stockcode = %s) column#=%d" % (row[0], row[2], k+1 ))
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
            else:
                print ('attention:  this row[%d] dataDate %s is not between [%s,%s],so discard this row' %(i, row[DataDateColumn],fromDate,toDate))

    log ('\ntotal row# read from this data file %s is (%d out of %d)' %(filename,hitCount,i))
    if missingvaluecount != 0:
        log("\n!!! WARNING: the input data file %s has %d blank value(s),setting its value to %d as a workaround, please check and fill them later!!!" % (filename, missingvaluecount,filling_value))

    data=np.array(data,dtype=features_dtype)
    target=np.array(target,dtype=target_dtype)

  return Dataset(data=data, target=target,featurenames=featureNames)


# visualize  the batch features in both a scatter subplot to review the min-max range of the features in a whole picture
# and a subplot of histogram for each features' distribution

def plotFeatures(batch,datasetFeatureNames,desc=None,savePlotToDisk=True):
    """
batch: is 2-D matrix [n_sample,n_feature]
datasetFeatureNames: a list that contains all the feature names in the same order as batch
desc is the string that describes the data,such as "2013-15training" ,will be used as part of plot title.
    """
    xscatter=[]
    yscatter=[]
    n_sample,n_feature= batch.shape

    # get and fill  min and max value for each features,prepare data for scatter plotting later
    for j in range(n_feature):
        y = batch[:, j]  # fetch all items in this column
        xscatter.append(j)
        yscatter.append(y.min())
        xscatter.append(j)
        yscatter.append(y.max())


    for j in range(n_feature): # for all features
        log("Plotting %s feature# %d in progress,time = %s " %(desc,j,(time.ctime())))
        featureFig = plt.figure("plotFeature"+str(j)+"_"+str(datasetFeatureNames[j]), figsize=(8, 6))
        featureFig.subplots_adjust(top=0.92, left=0.10, right=0.97,hspace=0.37, wspace=0.3)

        axn = featureFig.add_subplot(2, 1, 1)  # histogram each feature to check its distribution
        y= batch[:,j]  #fetch all items in this column

        axn.clear()   #clear previous histogram plot
        bins=[]    #prepare a list to seperate data into 1000 equal groups
        stepSize=(y.max()-y.min())/1000
        for k in range(0,1001):   #show distribution of 1000 equal parts 1000+1 to show the y.max()
            bins.append(y.min()+stepSize*k)

        axn.hist(y,bins,histtype='bar',rwidth=0.8,label=str(datasetFeatureNames[j]))
        axn.set_xlabel( 'feature value range: (%0.4f - %0.4f)' %(y.min(),y.max()))
        axn.set_ylabel( 'number')
        #axn.set_xlim(y.min(),y.max())
        #axn.set_ylim()
        axn.set_title('histograms of feature id# '+ str(j)+",name="+str(datasetFeatureNames[j]))
        axn.legend()

        axs = featureFig.add_subplot(2, 1, 2)  # show the scatter subplot for min,max value of all the feature in whole.
        axs.clear()
        axs.scatter(xscatter, yscatter, label="min/max value", color='k', s=25, marker='x')
        axs.set_xlabel('feature id#')
        axs.set_ylabel('actual value ')
        axs.set_title("scatter min & max for "+ str(n_feature) +" features in "+ desc + " with " +str(n_sample) +" samples")

        # adjust Y scale to show this column's min and max scatter point in the graph,at the cost of possibly sacrifice other columns
        # if y.max()>0 :
        #     ymax=y.max()*1.2
        # else:
        #     ymax=y.max()*0.8
        # if y.min()>0 :
        #     ymin=y.min()*0.8
        # else:
        #     ymin=y.min()*1.2
        # axs.set_ylim(ymin,ymax)

        axs.annotate('Here:id =' + str(j)+' min:%s' %yscatter[2*j], xy=(j, yscatter[2 * j]), xycoords='data',
                     xytext=(20, 30), textcoords='offset points',
                     arrowprops=dict(facecolor='black', arrowstyle='->'),
                     horizontalalignment='right', verticalalignment='bottom')
        axs.annotate('Here:id =' + str(j) + ' max:%s'%yscatter[2*j+1], xy=(j, yscatter[2*j+1]), xycoords='data',
                     xytext=(0.5, 0.85), textcoords='axes fraction',
                     arrowprops=dict(color='blue', arrowstyle='->'),
                     horizontalalignment='right', verticalalignment='bottom')
        axs.legend()

     #   plt.show()
        if savePlotToDisk:
            plt.savefig(desc+"feature"+str(j)+"_"+str(datasetFeatureNames[j])+".png", figsize=(8, 6))
        plt.close(featureFig)  #close figures explicitly to release memory
        #.clear()
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
  log('\nloading training data from file %s in progress ... time:%s' %(trainfilename,time.ctime()) ,logfilename)
  #Load datasets. discard column 0,1,3,4 in the original csvfile ,which represent  id ,tradedate,mc_date,datadate

  # Load datasets.

  training_set = load_partcsv_without_header(
      filename=trainfilename,
      target_dtype=np.int,
      features_dtype=np.float32,
      start_rowid=TrainDataStart,
      end_rowid=TrainDataStop,
      fromDate='2012/12/31',
      toDate='2015/12/31',
      discard_colids=[0,1,3,4,-1],  #add -1 in the last column to exclude the percentage infor 2# stockcode,52# market code
      target_column=5,
      filling_value=1.0
  )
# plot original data to review

  plotFeatures(training_set.data,training_set.featurenames,"Orig13-15train",True)

 # added by libo : declare and store this scaler, and use the same one to scale the test data

  dataScaler = MinMaxScaler().fit(training_set.data)
  #log('the data range of features in training set are %s' % dataScaler.data_range_)
  #dataScaler =  StandardScaler().fit(training_set.data)
  X = dataScaler.transform(training_set.data)
  log('the scaler factors got in training set are %s' % dataScaler.scale_)

  #plotFeatures(X,training_set.featurenames,"stdscale13-15train",False)
  #plotFeatures(X, training_set.featurenames, "minmaxscale13-15train",False)
  y = training_set.target
    #[training_set.target != 2]




  log ('the first 20 label of the training set are %s' %y[0:21])

  log('\nloading test data from file %s in progress ... time:%s' % (testfilename, time.ctime()),logfilename)


  test_set= load_partcsv_without_header(
      filename=testfilename,
      target_dtype=np.int,
      features_dtype=np.float32,
      start_rowid=TestDataStart,
      end_rowid=TestDataStop,
      fromDate='2016/1/1',
      toDate='2016/6/30',
      discard_colids=[0,1,3,4,-1],  #add -1 in the last column to exclude the percentage infor  2# stockcode,52# market code
      target_column=5,
      filling_value=1.0
  )
  # the following 2 sentences only get the class 0,1 DATA,discard data with class=2.
 #added by libo
  X_test = dataScaler.transform(test_set.data)
    #[test_set.target != 2, :]
  #plotFeatures(X_test,test_set.featurenames,"std16-17test ")
  #plotFeatures(X_test, test_set.featurenames, "minmaxscale16-17test ")


  y_test = test_set.target
    #[test_set.target != 2]
  log('the first 20 label of the test set are %s' % y_test[0:21])




  log('\ndata loading completed, building graph in progress ... time:%s' % (time.ctime()),logfilename)  # train a 3-hidden layer neural network


  # train a 4-hidden layer neural network with 80 nodes (161*80*80*80*80*2= 1318912000 weights, ?? of the training data
  with tf.Graph().as_default():
      # dpp= tflearn.data_preprocessing.DataPreprocessing(name="dataPreprocess")
      # dpp.add_custom_preprocessing(myNormalizer) #Mean: 3163.05 (To avoid repetitive computation, add it to argument 'mean' of `add_featurewise_zero_center`)

       net = tflearn.input_data([None, 161], data_preprocessing=None, data_augmentation=None, name="inputlayer")


       net = tflearn.fully_connected(net, 80, activation='tanh', weights_init='xavier', bias_init='normal',
                             regularizer=None, weight_decay=0.001, name='hidderlayer1')
       net = tflearn.fully_connected(net, 80, activation='tanh', weights_init='xavier', bias_init='normal',
                             regularizer='L2', weight_decay=0.001, name='hidderlayer2')
       net = tflearn.fully_connected(net, 80, activation='tanh', weights_init='xavier', bias_init='normal',
                             regularizer='L2', weight_decay=0.001, name='hidderlayer3')
       net = tflearn.fully_connected(net, 80, activation='tanh', weights_init='xavier', bias_init='normal',
                            regularizer='L2', weight_decay=0.001, name='hidderlayer4')
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

       net = tflearn.fully_connected(net, 2, activation='softmax',
                                     name='outputlayer')

       #Y = to_categorical(training_set.target, 3)

       admopt = tflearn.Adam(learning_rate=0.001)
       #momentum= tflearn.Momentum(learning_rate=0.01,lr_decay=0.96,decay_step=100)
       #rmsProp=tflearn.RMSProp(learning_rate=0.1,decay=0.9,momentum=0.1)
       # sgd = tflearn.SGD(learning_rate=0.01, lr_decay=0.96, decay_step=500)
       acc = tflearn.metrics.Accuracy()
       #topk = tflearn.metrics.Top_k(1)

       #net = tflearn.regression(net, optimizer=admopt, loss = 'roc_auc_score', metric=acc, \
       #                         to_one_hot=True, n_classes=2)
       #net = tflearn.regression(net, optimizer=admopt, loss = 'binary_crossentropy', to_one_hot=True, n_classes=2)

       net = tflearn.regression(net, optimizer=admopt, loss='categorical_crossentropy', metric=acc, \
                                to_one_hot=True, n_classes=2)

       model = tflearn.DNN(net, tensorboard_dir="/tmp/tflearn_6thlogs/", tensorboard_verbose=0)

       log('\ntraining the DNN classifier for %d epoches with mini_batch size of %d in progress ... time:%s' % (50,1024,time.ctime()))
       model.fit(X, y, validation_set= (X_test,y_test),show_metric=True, batch_size=1024,n_epoch=50,snapshot_epoch=False)




  def evalprint(X_predict, y_true,title,fig,nrow,ncol,plot_number,annotate=False,drawplot=True):
        log('\nevaluate the DNN classifier using %s in progress... time:%s' % (title,(time.ctime())))
        # print('\n%s inputX[0:20] is' %title)
        # print(X_predict[0:20])
        print('\n%s y_true[0:20] is' %title)
        print(y_true[0:20])
        predicted = model.predict(X_predict)



        # get the index of the largest possibility value of predicted list as the label of prediction
        verdictVector = np.argmax(predicted, axis=1)
        print('\n%s verdictVector[0:20] is' %title)
        print(verdictVector[0:20])
        print('\n%s raw probability of predicted[0:20,:] are ' %title)
        print(predicted[0:20, :])
        # log('\n%s             y_true counts:' % title)
        # log(y_true.value_counts())

        log('\n using %s, The AUC value =%f ' %(title, get_roc_auc(y_true, predicted)))
        log('\n               Null Accuracy= {}%' .format(100*max(y_true.mean(),(1-y_true.mean()))))
        log("               Test Accuracy: {}%".format(100 * np.mean(y_true == verdictVector)))

        #plot the ROC curve, review the implementation to make sure roc compute algorithm without using
        # the following example sklearn predict_proba function is correct or not ??? ...
        # probas_ = classifier.fit(X_train, y_train).predict_proba(X_test)

        plot_tflearn_ROC(y_true, predicted, title, fig, nrow, ncol, plot_number, annotate, drawplot)
        # use sklearn function to print
        from sklearn import metrics
        expected = y_true
        log(metrics.classification_report(expected, verdictVector, labels=[0, 1],
                                          target_names=['not buy', 'buy']))
        log(str(metrics.confusion_matrix(expected, verdictVector)))
        log("\n   evaluate model with %s completed, time: %s" % (title,time.ctime()))

  figid = plt.figure("total ROC 201315minmaxscaleTrain_201606minmaxTest",figsize=(10,8))
  figid.subplots_adjust(top=0.95, left=0.12, right=0.90,hspace=0.43, wspace=0.2)

  #evaluate the model with Training data
  evalprint(X, y,"Training data",figid,2,1,1,False,True)

    # evaluate the model with Test data
  evalprint(X_test, y_test, "Test data", figid, 2, 1, 2, False, True)

  endTime = time.time()  # end time in ms.
  elapseTime = (endTime - startTime)
  hour = int(elapseTime / 3600)
  minute = int((elapseTime % 3600) / 60)
  second = int((elapseTime % 3600) % 60)
  log("\nthe WHOLE ELAPSED time of loading data and training the model is %d hours:%d miniutes:%d seconds"
            % (hour, minute, second))
  plt.show()  # display the ROC plot onscreen, if plot ROC is not needed, you must comment this line out!!!
  plt.close(figid)  #close it to release memory


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

