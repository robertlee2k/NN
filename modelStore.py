
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
from sklearn.feature_selection import SelectKBest

from testResult import TestResult
from utility import log, get_roc_auc, printConfusionMatrix,plotSample
from dnnModel import DnnModel
from preprocess import DataPreprocess
from plot_tflearn_roc_auc import plot_tflearn_ROC  # for plotting ROC curve

FEATURENUM = 165     # original dataset's total feature number
EXPORT_DIR = "/home/topleaf/stock/savedModel/"  # export dir base
# the file is used to store/load datapreprocessor that is used before training and predicting
dppfilename = 'dpp.bin'     # default datapreprocess filename to hold DataPreprocess class dump
hyperparamfilename = 'hyperParam.bin'  # default filename to hold hpDict
featureSelectorfilename = 'FeatureKBest.bin'
testResultfile = "DNN_Training_results.csv"  # file to record model training results
evalResultfile = "DNN_eval_results.csv"     # file to record model evaluation results against test set after loading training model

# batch size of for prediction , prevent runing out of memory when predict a large set of data
# total needed memory = predictBatchSize * nodeNumbers * 4
predictBatchSize = 102400


def evalprint(model, X_predict, y_true, title,
              fig, nrow, ncol, plot_number, annotate=False, drawplot=True):
    """
    :param model: model instance to be used to predict
    :param X_predict:  input dataset X,
    :param y_true:  ground truth of the dataset's y labels
    :param title: Any string text to be displayed as the title of the subplot
    :param fig: main figid
    :param nrow:
    :param ncol:
    :param plot_number: plot #
    :param annotate:  add annotation on plot, WARNING: it will take FOREVER TO SHOW the plot, be cautious!!
    :param drawplot: whether or not to plot
    :return: tuple of (Aucvalue, testAccuracy,NullAccuracy of the dataset)
    """
    log('\nevaluate the DNN classifier %s using BatchSize = %d in progress... time:%s'
        % (title,predictBatchSize,(time.ctime())))


    fulllength = len(X_predict)
    batchnum = fulllength / predictBatchSize
    lastbatch = fulllength % predictBatchSize
    if batchnum != 0:
        predicted = model.predict(X_predict[0:predictBatchSize])
        for batchid in range(1,batchnum):
            predicted = np.concatenate((predicted, model.predict(X_predict[batchid*predictBatchSize:
                        batchid*predictBatchSize + predictBatchSize])),axis=0)
        if lastbatch!=0:
            predicted = np.concatenate((predicted,model.predict(X_predict[lastbatch*(-1):])),axis=0)
    else:
        predicted=model.predict(X_predict)




    # get the index of the largest possibility value of predicted list as the label of prediction
    verdictVector = np.argmax(predicted, axis=1)
    #print('\n%s verdictVector[0:20] is' %title)
    #print(verdictVector[0:20])
    #print('\n%s raw probability of predicted[0:20,:] are ' %title)
    #print(predicted[0:20, :])
    # log('\n%s             y_true counts:' % title)        # log(y_true.value_counts())

    aucValue = get_roc_auc(y_true, predicted)
    testAccuracy = np.mean(y_true == verdictVector)
    nullAccuracy = max(y_true.mean(), (1-y_true.mean()))
    log('\n %s, The AUC value =%f ' % (title, aucValue))
    log('\n               Null Accuracy= {}%' .format("%0.2f" % (100*nullAccuracy)))
    log("               Test Accuracy={}%".format("%0.2f" % (100 * testAccuracy)))

    # use sklearn function to print
    from sklearn import metrics
    expected = y_true
    log(metrics.classification_report(expected, verdictVector, labels=[0, 1],
                                      target_names=["predict=0", 'predict=1']))
    # print out confusion matrix
    cmstr=printConfusionMatrix(metrics.confusion_matrix(expected, verdictVector))
    log(cmstr)
    log("\n   evaluate model with %s completed, time: %s" % (title,time.ctime()))

    # plot the ROC curve, review the implementation to make sure roc compute algorithm without using
    # the following example sklearn predict_proba function is correct or not ??? ...
    # probas_ = classifier.fit(X_train, y_train).predict_proba(X_test)
    plot_tflearn_ROC(y_true, predicted, title, fig, nrow, ncol, plot_number,\
                     cmstr + 'truelabel=1 Percentage= {}%'.format("%0.2f" % (100*y_true.mean())) +
                     '\n' + 'NullAccuracy= {}%'.format("%0.2f" % (100*nullAccuracy)) +
                     '\n'+'Accuracy={}%'.format("%0.2f" % (100*testAccuracy)),
                     annotate, drawplot)
    return aucValue, testAccuracy, nullAccuracy


def errorAnalysis(model, X_predict, y_true, title, plotnum=10,savePlot=False):
    """
    :param model: model instance to be used to predict
    :param X_predict:  input dataset X,
    :param y_true:  ground truth of the dataset's y labels
    :param title: Any string text to be displayed as the title of the subplot
    :param plotnum: how many samples to plot
    :param savePlot: True or False, to plot the error classified samples and save the plot to disk under current folder
    """
    log('\nploting errorAnalysis with the model using  %s in progress... time:%s' % (title,(time.ctime())))
    assert (isinstance(model, DnnModel))
    predicted = model.model.predict(X_predict)

    # get the index of the largest possibility value of predicted list as the label of prediction
    verdictVector = np.argmax(predicted, axis=1)
    count=0
    for m in range(X_predict.__len__()):
        if verdictVector[m] != y_true[m]:   # this is a wrong prediction
            count += 1
            if count <= plotnum:
                plotSample(X_predict[m],predicted[m],verdictVector[m],y_true[m],title + str(m),savePlotToDisk=savePlot)
                # keyp = raw_input("\nPlease press any key to continue to the next sample (x to quit)")
                # if (keyp != 'x'):
                #     continue
                # else:
                #     break
            else:
                log("complete %d samples' prediction error plotting" %(count-1))
                break


   # return aucValue, testAccuracy, nullAccuracy


class ModelStore(object):
    """

    save a trained model to disk
    load a trained model according to input parameters in load()


    """
    __instance = None  # define instance of the class

    # use the code to generate only one instance of the class
    #re think about this , do i need more instance of ModelStore ??  when training and eval run at the same time,
    #what consequence will happen ?
    def __new__(cls, *args, **kwargs):  # this method is called before __init()__
        if ModelStore.__instance is None:
            ModelStore.__instance = object.__new__(cls, *args, **kwargs)
        return ModelStore.__instance

    def __init__(self):
        self.dirname = None    # the folder name to be used to save the model to disk
        self.matchedModelLocations = None
        self.testRecordName = testResultfile
        self.evalRecordName = evalResultfile
        self.modelfilename = 'trainedModel'
        self.dpp = None
        self.hpDict = None
        self.loadmymodel = None
        self.featureSel = None
        self.inputlayerNodenum = FEATURENUM
        self.trainAuc = self.traindevAuc = None
        self.testAuc = self.valAuc = None
        self.testTa = self.valTa = None
        self.trainTa = self.traindevTa = None
        self.trainNa = self.traindevNa = None
        self.testNa = self.valNa = None

    # save a trainedModel with its datapreprocessor object to disk, remember its location
    def save(self, hpDict, modelinst, dp, featureSelector):
        """
        :param hpDict: including all the required hyper parameters of this model,
        :param modelinst: model instance to be saved
        :param dp: datapreprocess instance which includes datascaler.
        :param featureSelector: the instance of featureselector that have been used in this model
        :return: None
        """

        if not isinstance(modelinst, DnnModel):
            assert ("input parameter(%s) must be a instance of %s" %(modelinst, DnnModel.__class__.__name__))
        if not isinstance(dp, DataPreprocess):
            assert("input parameter(%s) must be a instance of %s" %( dp, DataPreprocess.__class__.__name__))

        if not isinstance(featureSelector, SelectKBest) and featureSelector is not None:
            assert ("input parameter(%s) must be a instance of %s or None" % (featureSelector,SelectKBest.__class__.__name__))
            # think about an unique name to identify this model, here I use runid as the folder name
        # since it's unique
        #

            # "2011-16%s%s_%s_alpha%0.4f_lrdecay_%0.3f_decaystep%d_epoch%d_batch%d_TrainedModel" \
            #             % (hpDict['Preprocessor'], self.runid, self.opt.name,
            #                self.learningrate, self.lrdecay,self.decaystep,
            #                self.epoch,self.minibatch)
        self.hpDict = hpDict
        self.dpp = dp
        self.featureSel = featureSelector
        # save the model to disk, using unique runid of this model as folder name
        self.dirname = ''.join((EXPORT_DIR, modelinst.runid))
        if os.path.isfile(self.dirname):
            log("model file %s exists, do not overwrite it!!!!" % self.dirname)
            raise IOError("model file %s exists, do not overwrite it" % self.dirname)
        elif not os.path.isdir(self.dirname):
            os.mkdir(self.dirname)
            log('\ntraining completed, save the model to disk under folder:%s' % self.dirname)
        else:
            log('\n save the model to existing folder')

        fullpath = ''.join((self.dirname, '/', self.modelfilename))

        try:
            modelinst.model.save(fullpath)
        except ValueError as ve:
            raise ValueError(ve)
        else:
            log('\nmodel saved as %s' % fullpath)

        # save the dataprocessor to disk
        fullpath = ''.join((self.dirname, '/', dppfilename))
        import pickle
        with open(fullpath, 'wb') as f:
            pickle.dump(dp, f)

        # save the hyperparameter to disk
        fullpath = ''.join((self.dirname, '/', hyperparamfilename))
        with open(fullpath, 'wb') as f:
            pickle.dump(hpDict, f)

        # save the featureSelector to disk
        if featureSelector is not None:
            fullpath = ''.join((self.dirname, '/', featureSelectorfilename))
            import pickle
            with open(fullpath, 'wb') as f:
                pickle.dump(featureSelector, f)

    def getModelFullpath(self, fromDate, toDate, minAuc='0.55',maxAuc='1'):
        """
        get the unique Runid,compose the fullpathname
        :param fromDate:
        :param toDate:
        :param minAuc: minimum Auc
        :param maxAuc: maximum Auc
        :return: a list that contains the fullpath of all matched models
        """
        # search a lookup table, DNN_Training_result, using 4 parameters: fromDate,toDate
        #  minAuc and maxAuc to locate its unique location in disk
        # open the csv file to search , if found, return the location, otherwise,return None
        lookupTable = TestResult(self.testRecordName)
        try:
            lookupTable.readRows()
        except ValueError as ve:        # file doesn't exist
            log(ve.message)
            raise ValueError(ve.message)  # throw out the exception for caller to handle
        self.matchedModelLocations = []
        start = datetime.datetime.strptime(fromDate, "%Y/%m/%d")
        end = datetime.datetime.strptime(toDate, "%Y/%m/%d")
        for row in lookupTable.rows:
            stDate = datetime.datetime.strptime(row["TFromDate"], "%Y/%m/%d")
            toDate = datetime.datetime.strptime(row["TToDate"], "%Y/%m/%d")

            if start == stDate and end == toDate:  # found a matched training from/to date
                if float(row['AUC(Val)']) >= float(minAuc) \
                   and float(row['AUC(Val)']) <= float(maxAuc):
                    self.matchedModelLocations.append(row['Model Location'])

        # remove duplicate
        self.matchedModelLocations = list(set(self.matchedModelLocations))

        return self.matchedModelLocations

    def load(self, dirname):
        """
        load a previous trained model according to given dirname

        load weight from a previous trained model from disk,
        load dataprocessor from a previous trained model at the same folder
        :param dirname: the dir name of the model to be loaded
        :return: a modelinst that loaded the trained model from dirname folder
        """

        self.dirname = dirname

        dppfullpath = ''.join((self.dirname, '/', dppfilename))
        hpfullpath = ''.join((self.dirname, '/', hyperparamfilename))
        modelfullname = ''.join((self.dirname, '/', self.modelfilename))
        fsfullpath = ''.join((self.dirname, '/', featureSelectorfilename))
        try:
            # load its preprocess  datascaler
            if os.path.exists(dppfullpath):
                log('load previous preprocess datascaler:%s' % dppfullpath)
                import pickle
                with open(dppfullpath, 'rb') as f:
                    self.dpp = pickle.load(f)
            else:
                raise IOError("datapreprocess file:%s doesn't exist" % dppfullpath)
            # load the model's hyper parameters back in order to rebuild the model
            if os.path.exists(hpfullpath):
                log('load previous hyper parameter:%s' % hpfullpath)
                import pickle
                with open(hpfullpath, 'rb') as f:
                    self.hpDict = pickle.load(f)
            else:
                raise IOError("hyper parameter file%s doens't exist" %hpfullpath)

            # load the model's featureSelector back in order to rebuild the model
            if os.path.exists(fsfullpath):
                log('load previous feature selector:%s' % fsfullpath)
                import pickle
                with open(fsfullpath, 'rb') as f:
                    self.featureSel = pickle.load(f)
                    self.inputlayerNodenum = self.featureSel.k
            else:
                log("featureSelection file %s doesn't exist, use default inputLayerNodeNum=%d" %(fsfullpath,FEATURENUM))
                self.featureSel=None
                self.inputlayerNodenum = FEATURENUM

            # reconstruct the DNN model and load the weights from previous trained model
            if os.path.exists(modelfullname+'.meta'):
                log('load previous trained model:%s' % modelfullname)
                tf.reset_default_graph()
                self.loadmymodel = DnnModel(self.hpDict, self.inputlayerNodenum)
                self.loadmymodel.model.load(modelfullname, weights_only=True)
                self.loadmymodel.runid = os.path.basename(self.dirname)
            else:
                raise IOError("model file %s doesn't exist, horrible, "
                      "check the naming rule of saving/loading model" % modelfullname)

        except IOError as ve:  # can't find the trained model from disk
                        log(ve.message)
                        log("\nWARNING:  Skip this loading process ... ")
                        raise IOError(ve.message)
        except Exception as e:    # any other exceptions, just skip this evaluation, not a big deal
                        log("\nWARNING:  the following Exception occurred, skip this loading... ")
                        log(e.message)
                        raise Exception(e.message)
        return self

    def evaluate(self, modelinst, dp, X, y, X_test, y_test, mode):
        """
        1.evaluate the model using training data and test data
        2.generate auc,accuracy etc,
        3. generate and plot ROC and save the plt file
        4. append the test result to csv file

        :param : mode is a description for this plot, will be used in plotfilename
        :return: None
        """

        log("Evaluate the trained model,save its plot")
        if not isinstance(modelinst, DnnModel):
            assert ("input parameter(%s) must be a instance of %s" % (modelinst, DnnModel.__class__.__name__))
        if not isinstance(dp, DataPreprocess):
            assert ("input parameter(%s) must be a instance of %s" % (dp, DataPreprocess.__class__.__name__))
        try:
            figTitle = "Runid_%s_%s_%s_epoch%d_minibatch%d" \
                       % (modelinst.runid, dp.preScalerClassName, modelinst.opt.name,
                          modelinst.epoch, modelinst.minibatch)
            figid = plt.figure(figTitle, figsize=(10, 8))
            figid.subplots_adjust(top=0.95, left=0.12, right=0.90, hspace=0.43, wspace=0.2)

            if mode == 'shuffleTrainDev':
                # evaluate the model with Training data
                self.trainAuc, self.trainTa, self.trainNa = \
                    evalprint(modelinst.model, X, y,"with Training data From %s to %s,(Runid=%s),training loss=%0.4f"
                              % (self.hpDict['TFromDate'], self.hpDict['TToDate'],
                                 modelinst.runid, modelinst.model.trainer.training_state.global_loss),
                              figid, 2, 1, 1, False, True)

                # evaluate the model with Training_dev data
                self.traindevAuc, self.traindevTa, self.traindevNa = \
                    evalprint(modelinst.model, X_test, y_test, "with 1% additional data as " + mode,
                              figid, 2, 1, 2, annotate=True, drawplot=True)
            elif mode == "TestSet":
                # evaluate the model with Validation data
                self.valAuc, self.valTa, self.valNa = \
                    evalprint(modelinst.model, X, y,"with %s day(s) Validation data(From %s to %s)"
                              % (self.hpDict['ValidationDays'],
                                 self.hpDict['ValidationFromD'], self.hpDict['ValidationToD']),
                              figid, 2, 1, 1, False, True)
                # evaluate the model with Test data
                self.testAuc, self.testTa, self.testNa = \
                    evalprint(modelinst.model, X_test, y_test, "with test data(From %s to %s) "
                              % (self.hpDict['TestFromD'], self.hpDict['TestToD']),
                              figid, 2, 1, 2, annotate=True, drawplot=True)

            # update test result  to file
            plotName = "%s_%s_%s_alpha%0.4f_epoch%d_%d.png"\
                       % (mode, dp.preScalerClassName, modelinst.opt.name, modelinst.learningrate,
                          modelinst.epoch, modelinst.minibatch)
            fullpath = ''.join((EXPORT_DIR, modelinst.runid))
            if os.path.isfile(fullpath):
                log("file %s exists, do not overwrite it!!!!" % fullpath)
                raise IOError("file %s exists, do not overwrite it!!!!" % fullpath)
            elif not os.path.isdir(fullpath):
                os.mkdir(fullpath)
            fullpath = ''.join((EXPORT_DIR, modelinst.runid, '/', plotName))
            plt.savefig(fullpath, figsize=(10, 8))      # if the file exists, overwrite it

            # model.trainer.training_state.val_loss, \
            # model.trainer.training_state.val_acc,\
            # plt.show()  # display the ROC plot onscreen, if plot ROC is not needed, comment this line out!!!
            plt.close(figid)  # close it to release memory
        except Exception as e1:
            print ('=' * 30 + "exception happened:" + '=' * 30)
            print(Exception)
            print(e1)
            print ('=' * 30 + "end of print exception" + '=' * 30)
            raise Exception

    def writeResult(self, hpDict, modelinst, st, endt, duration):
        """
        update test result  to testResult file according to above column sequence
        :param modelinst: model instance of DNNModel
        :param hpDict:  datapreprocess instance that has been used to preprocess datasets
        :param st: start time in date/time format to build this model
        :param endt: end time in date/time format
        :param duration: total elapsetime of building/training/evaluating the model
        :return:
        """
        assert (isinstance(modelinst, DnnModel))
        if self.dirname is None:
            log("the model has NOT been saved to disk/load from disk, call save method in class %s before "
                % self.__class__.__name__)
            raise ValueError("self.dirname is None")
        if (self.trainAuc is not None) and (self.trainTa is not None ) \
            and (self.trainNa is not None):
            result = [hpDict['Seqno'], modelinst.runid, hpDict['Preprocessor'], modelinst.opt.name,
                      modelinst.regularization,
                      "%02d" % modelinst.hiddenLayer,
                      "%d" % modelinst.hiddenUnit,
                      "%0.2f" % modelinst.inputKeepProb,
                      "%0.2f" % modelinst.keepProb,
                      "%0.4f" % modelinst.learningrate,
                      "%0.4f" % modelinst.lrdecay,
                      "%0.4f" % modelinst.decaystep,
                      "%d" % modelinst.rs,
                      str(modelinst.epoch), str(modelinst.minibatch),
                      "%0.4f" % self.trainAuc,
                      "%0.4f" % self.traindevAuc,
                      "%0.4f" % self.valAuc,
                      "%0.4f" % self.testAuc,
                      "%0.4f" % modelinst.model.trainer.training_state.global_loss,
                      "%0.2f" % (self.trainTa * 100) + '%',
                      "%0.2f" % (self.trainNa * 100) + '%',
                      "%0.2f" % (self.traindevTa * 100) + '%',
                      "%0.2f" % (self.traindevNa * 100) + '%',
                      "%0.2f" % (self.valTa * 100) + '%',
                      "%0.2f" % (self.valNa * 100) + '%',
                      "%0.2f" % (self.testTa * 100) + '%',
                      "%0.2f" % (self.testNa * 100) + '%',
                      duration,
                      "%s" % st,
                      "%s" % endt,  self.dirname,
                      hpDict["TFromDate"], hpDict["TToDate"],
                      hpDict['TrainRowNum'],
                      hpDict['ValidationFromD'], hpDict['ValidationToD'],
                      hpDict['TestFromD'], hpDict['TestToD']]
            trackRecord = TestResult(self.testRecordName)
            trackRecord.append(result)
        else:
            # update test result  to file according to above column sequence
            # training auc,loss, training accuracy,training null accuracy,
            # trainingdev auc, trainingdev accuracy,null accuracy
            # Val auc, val accuracy and val null accuracy are NA,since model is reload from disk,no training
            result = [hpDict['Seqno'], modelinst.runid, hpDict['Preprocessor'], modelinst.opt.name,
                      modelinst.regularization,
                      "%02d" % modelinst.hiddenLayer,
                      "%d" % modelinst.hiddenUnit,
                      "%0.2f" % modelinst.inputKeepProb,
                      "%0.2f" % modelinst.keepProb,
                      "%0.4f" % modelinst.learningrate,
                      "%0.4f" % modelinst.lrdecay,
                      "%0.4f" % modelinst.decaystep,
                      "%d" % modelinst.rs,
                      str(modelinst.epoch), str(modelinst.minibatch),
                      "NA",
                      "NA",
                      "NA",
                      "%0.4f" % self.testAuc,
                      "NA",
                      "NA",
                      "NA",
                      "NA",
                      "NA",
                      "NA",
                      "NA",
                      "%0.2f" % (self.testTa * 100) + '%',
                      "%0.2f" % (self.testNa * 100) + '%',
                      duration,
                      "%s" % st,
                      "%s" % endt,  self.dirname,
                      hpDict["TFromDate"], hpDict["TToDate"],
                      hpDict['TrainRowNum'],
                      hpDict['ValidationFromD'], hpDict['ValidationToD'],
                      hpDict['TestFromD'], hpDict['TestToD']]

            evalRecord = TestResult(self.evalRecordName)
            evalRecord.append(result)

    def evaluateTestSet(self, X_test,y_test,itemDict):
        '''
        1.evaluate the model using the test data only
        2.generate auc,accuracy etc
        3. generate and plot ROC and save the plt file
        4. append the test result to csv file
        :param itemDict:
        :param X_test:
        :param y_test:
        :return:
        '''
        log("Evaluate the trained model with test data only,save its plot")
        assert (self.loadmymodel is not None)
        assert(isinstance(self.loadmymodel, DnnModel))
        assert (self.dpp is not None)
        assert (isinstance(self.dpp, DataPreprocess))
        assert (self.hpDict is not None)
        try:
            figTitle = "%sRunid_%s_%s_%s_epoch%d_minibatch%d" \
                       % (itemDict['Seqno'],self.loadmymodel.runid, self.hpDict['Preprocessor'],
                          self.loadmymodel.opt.name,
                          self.loadmymodel.epoch, self.loadmymodel.minibatch)
            figid = plt.figure(figTitle, figsize=(10, 8))
            figid.subplots_adjust(top=0.95, left=0.12, right=0.90, hspace=0.43, wspace=0.2)

            # evaluate the model with Test data
            self.testAuc, self.testTa, self.testNa = evalprint(self.loadmymodel.model,
                                                               X_test, y_test,
                          "with Test data (seqno=%s) From %s to %s"
                          % (itemDict['Seqno'],itemDict["TestFromD"], itemDict["TestToD"]),
                          figid, 1, 1, 1, annotate=True, drawplot=True)

            # save the plot to a file under the same runid folder
            #  with seqid as part of its name to save multiple testset plots
            # that share the same training model
            plotName = "%s%s_%s_alpha%0.4f_epoch%d_%d.png" \
                       % (itemDict['Seqno'], self.dpp.preScalerClassName, self.loadmymodel.opt.name,
                          self.loadmymodel.learningrate,
                          self.loadmymodel.epoch, self.loadmymodel.minibatch)
            fullpath = ''.join((EXPORT_DIR, self.loadmymodel.runid))
            if os.path.isfile(fullpath):
                log("file %s exists, do not overwrite it!!!!" % fullpath)
            elif not os.path.isdir(fullpath):
                os.mkdir(fullpath)
            fullpath = ''.join((EXPORT_DIR, self.loadmymodel.runid, '/', plotName))
            plt.savefig(fullpath, figsize=(10, 8))  # if the file exists, overwrite it
            # plt.show()  # display the ROC plot onscreen, if plot ROC is not needed, you must comment this line out!!!
            plt.close(figid)  # close it to release memory
        except Exception as e1:
            print ('=' * 30 + "exception happened:" + '=' * 30)
            print(Exception)
            print(e1)
            print ('=' * 30 + "end of print exception" + '=' * 30)



