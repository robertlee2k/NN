import tflearn
import numpy as np

import time
import os
import matplotlib.pyplot as plt



#below are self written modules
from utility import log,get_roc_auc,printConfusionMatrix
from testResult import TestResult

from plot_tflearn_roc_auc import plot_tflearn_ROC #for plotting ROC curve

EXPORT_DIR = "/tmp/savedstockmodel/"  #export dir base

testResultfile="DNN_Training_results.csv"


def evalprint(model, X_predict, y_true, title,
              fig, nrow, ncol, plot_number, annotate=False, drawplot=True):
    '''

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
    '''
    log('\nevaluate the DNN classifier %s in progress... time:%s' % (title,(time.ctime())))

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
    nullAccuracy=max(y_true.mean(),(1-y_true.mean()))
    log('\n %s, The AUC value =%f ' %(title, aucValue))
    log('\n               Null Accuracy= {}%' .format("%0.2f"%(100*nullAccuracy)))
    log("               Test Accuracy={}%".format("%0.2f"%(100 * testAccuracy)))




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
    plot_tflearn_ROC(y_true, predicted, title, fig, nrow, ncol, plot_number,\
                     cmstr+'\n'+ 'NullAccuracy= {}%'.format("%0.2f"%(100*nullAccuracy))+\
                     '\n'+'Accuracy={}%'.format("%0.2f"%(100*testAccuracy)), \
                     annotate, drawplot)
    return(aucValue,testAccuracy,nullAccuracy)

class DnnModel(object):
    '''
    define a stock model for trial
    '''

    def __init__(self, hpDict, runid):
        self.startTime = time.time()  # start time in ms.
        self.st = time.ctime()  # start time in date/time format
        log('\n %s ----> After applying %s preprocessor %s, building the DNN Model'
            ' (rs=%s,alpha=%s,decayrate=%s,decaystep=%s) \
             for %s epoches with mini_batch size of %s in progress ... time:%s' \
            % (hpDict['Seqno'], hpDict['Preprocessor'], hpDict['Optimizer'],
               hpDict['RS'], hpDict['Alpha'], hpDict['lrdecay'],
               hpDict['decaystep'], hpDict['Epoch'], hpDict['Minibatch'],
               time.ctime()))

        optDictMap = {'Adam': tflearn.Adam(), 'Momentum': tflearn.Momentum(),
                      'RmsProp': tflearn.RMSProp(),'SGD':tflearn.SGD()}

        self.opt = optDictMap[hpDict['Optimizer']]  # create an instance of an optimizer class

        self.runid = runid  # unique id for this model's instance
        self.epoch = long(hpDict['Epoch'])
        self.learningrate = float(hpDict['Alpha'])
        self.regularization=hpDict['Regularization']

        self.minibatch = long(hpDict['Minibatch'])  # 16384  # 8192
        self.lrdecay = float(hpDict['lrdecay'])
        self.decaystep = float(hpDict['decaystep'])
        self.rs = int(hpDict['RS'])

        net = self.make_core_network()

        # opt = tflearn.Momentum(learning_rate=learningrate,lr_decay=lrdecay,decay_step=decaystep)
        # opt=tflearn.RMSProp(learning_rate=learningrate,decay=lrdecay,momentum=0.1)
        # opt = tflearn.SGD(learning_rate=learningrate, lr_decay=lrdecay, decay_step=decaystep)
        acc = tflearn.metrics.Accuracy()
        # topk = tflearn.metrics.Top_k(1)
        # net = tflearn.regression(net, optimizer=admopt, loss = 'roc_auc_score', metric=acc, \
        #                         to_one_hot=True, n_classes=2)
        # net = tflearn.regression(net, optimizer=opt, loss = 'binary_crossentropy', to_one_hot=True, n_classes=2)

        # net = tflearn.regression(net, optimizer=momentum, loss='categorical_crossentropy', metric=acc, \
        #                          to_one_hot=True, n_classes=2)
        net = tflearn.regression(net, optimizer=self.opt, loss='categorical_crossentropy', metric=acc, \
                                 to_one_hot=True, n_classes=2)

        model = tflearn.DNN(net, tensorboard_dir="/tmp/tflearn_12thlogs/", tensorboard_verbose=0)

        self.model = model

    def make_core_network(self):
        # train a 4-hidden layer neural network with 150 nodes in each layer
        # self.g=tf.Graph()
        # with self.g.as_default():
        # dpp= tflearn.data_preprocessing.DataPreprocessing(name="dataPreprocess")
        # dpp.add_custom_preprocessing(myNormalizer) #Mean: 3163.05 (To avoid repetitive computation, add it to argument 'mean' of `add_featurewise_zero_center`)
        rng = np.random.RandomState(self.rs)
        seedweight = rng.uniform(0.001, 1)
        log('seedweight=%0.4f' % seedweight)
        seedbias = rng.uniform(0.01, 0.09)
        log('seedbias=%0.4f' % seedbias)
        xavierInit = tflearn.initializations.xavier(uniform=True, seed=seedweight)
        normalInit = tflearn.initializations.normal(seed=seedbias)

        # try pca
        # net = tflearn.input_data([None, 150], data_preprocessing=None, data_augmentation=None, name="inputlayer")

        net = tflearn.input_data([None, 165], data_preprocessing=None, data_augmentation=None, name="inputlayer")
        net = tflearn.fully_connected(net, 150, activation='relu', weights_init=xavierInit, bias_init=normalInit,
                                      regularizer=self.regularization, weight_decay=0.001, name='hidderlayer1')
        net = tflearn.fully_connected(net, 150, activation='relu', weights_init=xavierInit, bias_init=normalInit,
                                      regularizer=self.regularization, weight_decay=0.001, name='hidderlayer2')
        net = tflearn.fully_connected(net, 150, activation='relu', weights_init=xavierInit, bias_init=normalInit,
                                      regularizer=self.regularization, weight_decay=0.001, name='hidderlayer3')
        net = tflearn.fully_connected(net, 150, activation='relu', weights_init=xavierInit, bias_init=normalInit,
                                      regularizer=self.regularization, weight_decay=0.001, name='hidderlayer4')
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

        net = tflearn.fully_connected(net, 2, activation='softmax', weights_init=xavierInit, bias_init=normalInit,
                                      name='outputlayer')
        return net

        # Y = to_categorical(training_set.target, 3)

    def train(self, X, y, X_test, y_test):

        self.model.fit(X, y, validation_set=(X_test, y_test), shuffle=True, show_metric=True, run_id=self.runid, \
                       batch_size=self.minibatch, n_epoch=self.epoch, snapshot_epoch=False, snapshot_step=10000)

    # if you need to load a previous model with all weights,uncomment the following lines to do it
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
    # outlayer_var= tflearn.get_layer_variables_by_name("outputlayer")
    # log ("\noutput layer weight:")
    # log(outlayer_var[0])
    # log ("\noutputlayer bias:")
    # log (outlayer_var[1])

    # h4layer_var= tflearn.get_layer_variables_by_name("hidderlayer4")
    # log("\nhiddenlayer 4 weight:")
    # log(h4layer_var[0])

    # model.get_weights(net.W)
    def saveModel(self, hpDict):
        '''
        generate a descriptive name for the model and save the trained model to disk
        :param hpDict: including all the required hyper parameters of this model
        :return: None
        '''
        self.modelfilename = "2011-16%s%s_%s_alpha%0.4f_lrdecay_%0.2f_decaystep%d_epoch%d_batch%d_TrainedModel" \
                        % (hpDict['Preprocessor'], self.runid, self.opt.name,
                           self.learningrate, self.lrdecay,self.decaystep,
                           self.epoch,self.minibatch)
        # save the model to disk
        fullpath = ''.join((EXPORT_DIR, self.runid))
        if os.path.isfile(fullpath):
            log("model file %s exists, do not overwrite it!!!!" % fullpath)
        elif os.path.isdir(fullpath) == False:
            os.mkdir(fullpath)
            fullpath = ''.join((EXPORT_DIR, self.runid, '/', self.modelfilename))
            log('\ntraining completed, save the model to disk as %s' % fullpath)
            self.model.save(fullpath)
        else:
            fullpath = ''.join((EXPORT_DIR, self.runid, '/', self.modelfilename))
            log('\ntraining completed, folder exists, overwrite it with new model as %s' % fullpath)
            self.model.save(fullpath)

    def evaluate(self,hpDict,X,y,X_test,y_test):
        '''
        1.evaluate the model using training data and test data
        2.generate auc,accuracy etc,
        3. generate and plot ROC and save the plt file
        4. append the test result to csv file

        :param hpDict:
        :return: None
        '''

        log("Evaluate the trained model,save its plot")
        # X Error of failed request:  BadAccess (attempt to access private resource denied)
        #  Major opcode of failed request:  88 (X_FreeColors)
        # Serial   number  of failed request:  1505
        #   Current   serial  number in output  stream:  1507
        try:
            figid = plt.figure("ROC 2011-16Train201709Test Runid(%s) %s_%s_epoch%d_minibatch%d"
                               % (self.runid, hpDict['Preprocessor'], self.opt.name,
                                  self.epoch, self.minibatch), figsize=(10, 8))
            figid.subplots_adjust(top=0.95, left=0.12, right=0.90, hspace=0.43, wspace=0.2)

            # evaluate the model with Training data
            trainAuc, trainTa, trainNa = evalprint(self.model, X, y, "with Training data,training loss=%0.4f" \
                                                   % (self.model.trainer.training_state.global_loss), \
                                                   figid, 2, 1, 1, False, True)

            # evaluate the model with Test data
            testAuc, testTa, testNa = evalprint(self.model, X_test, y_test, "with Test data ", \
                                                figid, 2, 1, 2, False, True)


            # update test result  to file

            plotName = "ROC2011-16Train_201709Test%s_%s_alpha%0.4f_epoch%d_%d.png" \
                       % (hpDict['Preprocessor'], self.opt.name, self.learningrate,
                          self.epoch, self.minibatch)
            fullpath = ''.join((EXPORT_DIR, self.runid))
            if os.path.isfile(fullpath):
                log("file %s exists, do not overwrite it!!!!" % fullpath)
            elif os.path.isdir(fullpath) == False:
                os.mkdir(fullpath)
            fullpath = ''.join((EXPORT_DIR, self.runid, '/', plotName))
            plt.savefig(fullpath, figsize=(10, 8))

            # model.trainer.training_state.val_loss, \
            # model.trainer.training_state.val_acc,\

            #calculate the duration of building/training/evaluate the model
            endTime = time.time()  # end time in ms.
            elapseTime = (endTime - self.startTime)
            hour = int(elapseTime / 3600)
            minute = int((elapseTime % 3600) / 60)
            second = int((elapseTime % 3600) % 60)
            duration = "%dh%d'%d''" % (hour, minute, second)

            log("\nthe time of building/training/evaluating the model is %s" % (duration))

            # update test result  to file according to above column sequence
            result = [hpDict['Seqno'],self.runid, hpDict['Preprocessor'], self.opt.name,
                      self.regularization,
                      "%0.4f" % self.learningrate,
                      "%0.4f" % self.lrdecay,
                      "%0.4f" % self.decaystep,
                      "%d" % self.rs,
                      "%0.4f" % trainAuc,
                      "%0.4f" % self.model.trainer.training_state.global_loss,
                      "%0.2f" % (trainTa*100) + '%',
                      "%0.4f" % testAuc,
                      "%0.2f" % (testTa*100) + '%',\
                      "%0.2f" % (testNa * 100) + '%',
                      duration, "%s" % (self.st), "%s" \
                      % (time.ctime()), str(self.epoch), str(self.minibatch), fullpath]

            trackRecord = TestResult(testResultfile)

            trackRecord.append(result)

            # plt.show()  # display the ROC plot onscreen, if plot ROC is not needed, you must comment this line out!!!
            plt.close(figid)  # close it to release memory
        except Exception as e1:
            print ('=' * 30 + "exception happened:" + '=' * 30)
            print(Exception)
            print(e1)
            print ('=' * 30 + "end of print exception" + '=' * 30)
        finally:
            log("DEBUGGING X ERROR CODE END")

        endTime = time.time()  # end time in ms.
        elapseTime = (endTime - self.startTime)
        hour = int(elapseTime / 3600)
        minute = int((elapseTime % 3600) / 60)
        second = int((elapseTime % 3600) % 60)
        duration = "%dh%d'%d''" % (hour, minute, second)
        log("\nmodel building/training/evaluation completed,duration is %s" % (duration))