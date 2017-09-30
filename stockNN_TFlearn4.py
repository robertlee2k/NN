
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



import csv


# Data sets



TRAINDATASTART = 1      # the row# of the beginning of training data
TRAINDATASTOP = 1132174  # the row# of the end of the training data record  2013-2015csv file
TESTDATASTART=1     # the row# of the starting in test csv file 2016-2017
TESTDATASTOP=1048576  # the last row of the whole file, this row# is excluded in test data

TrainDataStart = 1
TrainDataStop = 3000 #TRAINDATASTOP  #for debugging purpose ,you can adjust this to get a small part for time saving now
TestDataStart = 1 #350603
TestDataStop = 200# TESTDATASTOP # TESTDATASTOP     # for debugging purpose ,you can adjust this to get a small part for time saving now


trainfilename = "/home/topleaf/stock/tensorFlowData/tensorFlowData(201101-201612).csv"   #training data file
testfilename = "/home/topleaf/stock/tensorFlowData/tensorFlowData(201701-201709).csv"  # test data  file

hyperParamSetFile = "HyperParamSearchPlan.csv"

# the following definition specified the column id# in original csv data file,starting from 0

DataDateColumn = 4
import datetime
import collections
from tensorflow.python.platform import gfile
# from sklearn.externals import joblib

#the following packages  are part of the project
from utility import log
from dnnModel import DnnModel
from  preprocess import DataPreprocess
from hyperParam import HyperParam


Dataset = collections.namedtuple('Dataset',['data','target','featurenames'])



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



def main():


  runStartTime=time.time();  # start time in ms.
  st=time.ctime()  #start time in date/time format
  log('\nloading training data from file %s in progress ... time:%s' %(trainfilename,st))
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
  # plotFeatures(training_set.data,training_set.featurenames,"Orig11-16train",savePlotToDisk=True,scatterAdjust=False)

  log('\nloading test data from file %s in progress ... time:%s' % (testfilename, time.ctime()))
  test_set = load_partcsv_without_header(
      filename=testfilename,
      target_dtype=np.int,
      features_dtype=np.float32,
      start_rowid=TestDataStart,
      end_rowid=TestDataStop,
      fromDate='2016/12/30',
      toDate='2017/9/19',
      discard_colids=[0, 1, 2, 3, 4, -1],  # add -1 in the last column to exclude the percentage infor  2# stockcode
      target_column=5,
      filling_value=1.0
  )


  # plot original test data to review
  # plotFeatures(test_set.data,test_set.featurenames,"Orig1709test",savePlotToDisk=True,scatterAdjust=False)

  # instantiate a HyperParam class to read  hyperparameter search plan into a list
  hpIns = HyperParam(hyperParamSetFile)

  y = training_set.target

  y_test = test_set.target


  for  seqid in range(1,hpIns.rows.__len__()):
        hpDict,preProcessChanged = hpIns.readRow(rowId=seqid)

        if preProcessChanged:  # this row's preprocessor is different from last one,apply preprocessing
            dp = DataPreprocess(hpDict['Preprocessor'])
            dp.fit(training_set)
            X = dp.transform(training_set)
            X_test = dp.transform(test_set)   #use the same scaler to transform test_set.data

            # plotFeatures(X_test, test_set.featurenames, hpDict['Preprocessor'],savePlotToDisk=True,scatterAdjust=False)

            # try pca
            # X_test = pca.transform(X_test)
        log('\ndata preprocessing completed, building model in progress ... time:%s' % (time.ctime()))


        tf.reset_default_graph()
        runId = tflearn.utils.id_generator()
        mymodel = DnnModel(hpDict,runId)
        mymodel.train(X,y,X_test,y_test)
        mymodel.saveModel(hpDict)
        mymodel.evaluate(hpDict,X,y,X_test,y_test)


  endTime = time.time()  # end time in ms.
  elapseTime = (endTime - runStartTime)
  hour = int(elapseTime / 3600)
  minute = int((elapseTime % 3600) / 60)
  second = int((elapseTime % 3600) % 60)
  duration = "%dh%d'%d''" % (hour, minute, second)
  log("\nthe WHOLE ELAPSED time of loading data and training all the models is %s"% (duration))


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



if __name__ == "__main__":
    main()
