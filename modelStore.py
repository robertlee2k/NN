
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

from testResult import TestResult
from utility import log, get_roc_auc, printConfusionMatrix
from dnnModel import DnnModel
from preprocess import DataPreprocess
from plot_tflearn_roc_auc import plot_tflearn_ROC  # for plotting ROC curve

EXPORT_DIR = "/home/topleaf/stock/savedModel/"  # export dir base
# the file is used to store/load datapreprocessor that is used before training and predicting
dppfilename = 'dpp.bin'     # default datapreprocess filename to hold DataPreprocess class dump
hyperparamfilename = 'hyperParam.bin'  # default filename to hold hpDict
testResultfile = "DNN_Training_results.csv"


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
    log('\nevaluate the DNN classifier %s in progress... time:%s' % (title,(time.ctime())))

    predicted = model.predict(X_predict)

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



class ModelStore(object):
    """

    save a trained model to disk
    load a trained model according to input parameters in load()


    """
    __instance = None  # define instance of the class

    # use the code to generate only one instance of the class
    def __new__(cls, *args, **kwargs):  # this method is called before __init()__
        if ModelStore.__instance is None:
            ModelStore.__instance = object.__new__(cls, *args, **kwargs)
        return ModelStore.__instance

    def __init__(self):
        self.dirname = None        # the folder name to be used to save the model to disk
        self.testRecordName = testResultfile
        self.modelfilename = None
        self.dpp = None
        self.hpDict = None
        self.loadmymodel = None
        self.trainAuc = 0.0
        self.testAuc = 0.0
        self.testTa = 0.0
        self.trainTa = 0.0
        self.trainNa = None
        self.testNa = None

    # save a trainedModel with its datapreprocessor object to disk, remember its location
    def save(self, hpDict, modelinst, dp):
        """
        :param hpDict: including all the required hyper parameters of this model,
        :param modelinst: model instance to be saved
        :param dp: datapreprocess instance which includes datascaler.
        :return: None
        """

        if not isinstance(modelinst, DnnModel):
            assert ("input parameter(%s) must be a instance of %s" %(modelinst, DnnModel.__class__.__name__))
        if not isinstance(dp, DataPreprocess):
            assert("input parameter(%s) must be a instance of %s" %( dp, DataPreprocess.__class__.__name__))
        # think about an unique name to identify this model, here I use runid as the folder name
        # since it's unique
        #

        self.modelfilename = 'trainedModel'

            # "2011-16%s%s_%s_alpha%0.4f_lrdecay_%0.3f_decaystep%d_epoch%d_batch%d_TrainedModel" \
            #             % (hpDict['Preprocessor'], self.runid, self.opt.name,
            #                self.learningrate, self.lrdecay,self.decaystep,
            #                self.epoch,self.minibatch)

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

        fullpath = ''.join((self.dirname, '/', hyperparamfilename))
        with open(fullpath, 'wb') as f:
            pickle.dump(hpDict, f)

    def _getModelFullpath(self, fromDate, toDate):
        """
        get the unique Runid,compose the fullpathname
        :param fromDate:
        :param toDate:
        :return: a string that contains the fullpath of the model
        """
        # search a lookup table, DNN_Training_result, using 2 parameters: fromDate,toDate
        # to locate its unique location in disk

        # test code to be changed
        modelfilename = fromDate + '_' + toDate + '_TrainedModel'
        runid = '7U7TJ7'
        modelfullname = ''.join((EXPORT_DIR, runid, '/', modelfilename))
        return modelfullname

    def load(self, fromDate, toDate):
        """
        load a previous trained model according to given fromDate and toDate

        load weight from a previous trained model from disk,
        load dataprocessor from a previous trained model at the same folder
        :param fromDate: in format such as fromDate='2016/12/30',
        :param toDate:  in format as  toDate='2017/09/19',
        :return: modelinst
        """
        modelfullname = self._getModelFullpath(fromDate, toDate)

        dirname = os.path.dirname(modelfullname)
        dppfullpath = ''.join((EXPORT_DIR, dirname, '/', dppfilename))
        hpfullpath = ''.join((EXPORT_DIR, dirname, '/', hyperparamfilename))
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

            # reconstruct the DNN model and load the weights from previous trained model
            if os.path.exists(modelfullname+'.meta'):
                log('load previous trained model:%s' % modelfullname)
                tf.reset_default_graph()
                # use this fake runId distinguish load models from original model, don't retrain the model
                self.loadmymodel = DnnModel(self.hpDict, 'load')
                self.loadmymodel.load(modelfullname, weights_only=True)

            else:
                raise IOError("model file %s doesn't exist, horrible, "
                      "check the naming rule of saving/loading model" % modelfullname)

        except IOError as ve:  # can't find the trained model from disk
                        log(ve.message)
                        log("\nWARNING:  Skip this loading process ... ")
        except Exception:    # any other exceptions, just skip this evaluation, not a big deal
                        log("\nWARNING:  Exception occurred, skip this evaluation... ")
        return self.loadmymodel

    def evaluate(self, seqid, modelinst, dp, X, y, X_test, y_test):
        """
        1.evaluate the model using training data and test data
        2.generate auc,accuracy etc,
        3. generate and plot ROC and save the plt file
        4. append the test result to csv file

        :param hpDict:
        :return: None
        """

        log("Evaluate the trained model,save its plot")
        assert(isinstance(modelinst, DnnModel))
        assert (isinstance(dp, DataPreprocess))

        try:
            figTitle = "Runid_%s_%s_%s_epoch%d_minibatch%d" \
                       % (modelinst.runid, dp.preScalerClassName, modelinst.opt.name,
                          modelinst.epoch, modelinst.minibatch)
            figid = plt.figure(figTitle, figsize=(10, 8))
            figid.subplots_adjust(top=0.95, left=0.12, right=0.90, hspace=0.43, wspace=0.2)

            # evaluate the model with Training data
            self.trainAuc,self.trainTa, self.trainNa = evalprint(modelinst.model, X, y,
                                                   "with Training data(Runid=%s),training loss=%0.4f"
                                                   % (modelinst.runid, modelinst.model.trainer.training_state.global_loss),
                                                   figid, 2, 1, 1, False, True)

            # evaluate the model with Test data
            self.testAuc, self.testTa, self.testNa = evalprint(modelinst.model, X_test, y_test, "with Test data ",
                                                               figid, 2, 1, 2, annotate=True, drawplot=True)

            # update test result  to file
            plotName = "%s_%s_alpha%0.4f_epoch%d_%d.png" \
                       % (dp.preScalerClassName, modelinst.opt.name, modelinst.learningrate,
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

    def writeResult(self, seqid, hpDict, modelinst, st, endt, duration):
        """
        update test result  to testResult file according to above column sequence
        :param seqid:  the id in hyperParamSearch plan
        :param modelinst: model instance of DNNModel
        :param hpDict:  datapreprocess instance that has been used to preprocess datasets
        :param st: start time in date/time format to build this model
        :param endt: end time in date/time format
        :param duration: total elapsetime of building/training/evaluating the model
        :return:
        """
        assert (isinstance(modelinst, DnnModel))
        if self.dirname is None:
            log("the model has NOT been saved to disk, call save method in class %s before "
                % self.__class__.__name__)
            self.dirname = "None saved"

        result = [seqid, modelinst.runid, hpDict['Preprocessor'], modelinst.opt.name,
                  modelinst.regularization,
                  "%02d" % modelinst.hiddenLayer,
                  "%d" % modelinst.hiddenUnit,
                  "%0.2f" % modelinst.inputKeepProb,
                  "%0.2f" % modelinst.keepProb,
                  "%0.4f" % modelinst.learningrate,
                  "%0.4f" % modelinst.lrdecay,
                  "%0.4f" % modelinst.decaystep,
                  "%d" % modelinst.rs,
                  "%0.4f" % self.trainAuc,
                  "%0.4f" % modelinst.model.trainer.training_state.global_loss,
                  "%0.2f" % (self.trainTa * 100) + '%',
                  "%0.4f" % self.testAuc,
                  "%0.2f" % (self.testTa * 100) + '%',
                  "%0.2f" % (self.testNa * 100) + '%',
                  duration,
                  "%s" % st,
                  "%s" % endt, str(modelinst.epoch), str(modelinst.minibatch), self.dirname,
                  hpDict["TFromDate"], hpDict["TToDate"],
                  hpDict['TestFromD'], hpDict['TestToD']]

        trackRecord = TestResult(self.testRecordName)

        trackRecord.append(result)


    # def evaluateTestSet(self,hpDict,X_test,y_test):
    #     '''
    #     1.evaluate the model using the test data only
    #     2.generate auc,accuracy etc
    #     3. generate and plot ROC and save the plt file
    #     4. append the test result to csv file
    #     :param hpDict:
    #     :param X_test:
    #     :param y_test:
    #     :return:
    #     '''
    #     log("Evaluate the trained model with test data only,save its plot")
    #
    #     try:
    #         figTitle = "Runid_%s_%s_%s_epoch%d_minibatch%d" % (self.runid, hpDict['Preprocessor'], self.opt.name,
    #                                                            self.epoch, self.minibatch)
    #         figid = plt.figure(figTitle, figsize=(10, 8))
    #         figid.subplots_adjust(top=0.95, left=0.12, right=0.90, hspace=0.43, wspace=0.2)
    #
    #
    #         # evaluate the model with Test data
    #         testAuc, testTa, testNa = evalprint(self.model, X_test, y_test, "with Test data (Runid=%s)"%self.runid,
    #                                             figid, 1, 1, 1, annotate=True, drawplot=True)
    #
    #         # update test result  to file
    #
    #         plotName = "%s_%s_alpha%0.4f_epoch%d_%d.png" \
    #                    % (hpDict['Preprocessor'], self.opt.name, self.learningrate,
    #                       self.epoch, self.minibatch)
    #         fullpath = ''.join((EXPORT_DIR, self.runid))
    #         if os.path.isfile(fullpath):
    #             log("file %s exists, do not overwrite it!!!!" % fullpath)
    #         elif os.path.isdir(fullpath) == False:
    #             os.mkdir(fullpath)
    #         fullpath = ''.join((EXPORT_DIR, self.runid, '/', plotName))
    #         plt.savefig(fullpath, figsize=(10, 8))  # if the file exists, overwrite it
    #
    #         # model.trainer.training_state.val_loss, \
    #         # model.trainer.training_state.val_acc,\
    #
    #         # calculate the duration of building/training/evaluate the model
    #         endTime = time.time()  # end time in ms.
    #         elapseTime = (endTime - self.startTime)
    #         hour = int(elapseTime / 3600)
    #         minute = int((elapseTime % 3600) / 60)
    #         second = int((elapseTime % 3600) % 60)
    #         duration = "%dh%d'%d''" % (hour, minute, second)
    #
    #         log("\nthe time of building/training/evaluating the model is %s" % (duration))
    #
    #         # update test result  to file according to above column sequence training auc,loss, training accuracy are NA
    #         result = [hpDict['Seqno'], self.runid, hpDict['Preprocessor'], self.opt.name,
    #                   self.regularization,
    #                   "%02d" % self.hiddenLayer,
    #                   "%d" % self.hiddenUnit,
    #                   "%0.2f" % self.inputKeepProb,
    #                   "%0.2f" % self.keepProb,
    #                   "%0.4f" % self.learningrate,
    #                   "%0.4f" % self.lrdecay,
    #                   "%0.4f" % self.decaystep,
    #                   "%d" % self.rs,
    #                   "NA",
    #                   "NA",
    #                   "NA",
    #                   "%0.4f" % testAuc,
    #                   "%0.2f" % (testTa * 100) + '%',
    #                   "%0.2f" % (testNa * 100) + '%',
    #                   duration,
    #                   "%s" % (self.st),
    #                   "%s" % (time.ctime()), str(self.epoch), str(self.minibatch), fullpath,
    #                   "NA", 'NA','NA',
    #                   hpDict["Test"], hpDict['TestFromD'], hpDict['TestToD']]
    #
    #         trackRecord = TestResult(testResultfile)
    #
    #         trackRecord.append(result)
    #
    #         # plt.show()  # display the ROC plot onscreen, if plot ROC is not needed, you must comment this line out!!!
    #         plt.close(figid)  # close it to release memory
    #     except Exception as e1:
    #         print ('=' * 30 + "exception happened:" + '=' * 30)
    #         print(Exception)
    #         print(e1)
    #         print ('=' * 30 + "end of print exception" + '=' * 30)



