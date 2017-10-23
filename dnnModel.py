import tflearn
import numpy as np

import time
import os




#below are self written modules
from utility import log
from hyperParam import supportedOptimizer, supportedRegularization

# move this clause out of   _init_ of DnnModel class
#  to avoid creating multiple instances of 4 Optimizer classes
optDictMap = {'Adam': tflearn.Adam(), 'Momentum': tflearn.Momentum(),
              'RMSProp': tflearn.RMSProp(), 'SGD': tflearn.SGD()}



class DnnModel(object):
    """
       define a stock model for trial
    """

    __instance = None       # this is the class' private attribute

    def __new__(cls, *args, **kwargs):   # this methold is called before __init() by default to instantiate an instance
        if DnnModel.__instance is None:
            DnnModel.__instance = object.__new__(cls, *args, **kwargs)
        return DnnModel.__instance

    def __init__(self, hpDict):
        log('\n %s : After applying %s preprocessor %s,regularization=%s,hiddenLayer=%s,hiddenUnit=%s,inputKeepProb=%s,\
            keep_prob=%s building the DNN Model(rs=%s,alpha=%s,decayrate=%s,decaystep=%s) for %s epoches with mini_batch size of %s in progress ... time:%s'
            % (hpDict['Seqno'], hpDict['Preprocessor'], hpDict['Optimizer'],
               hpDict['Regularization'], hpDict['HiddenLayer'],
               hpDict['HiddenUnit'], hpDict['InputKeepProb'],
               hpDict['KeepProb'],
               hpDict['RS'], hpDict['Alpha'], hpDict['lrdecay'],
               hpDict['decaystep'], hpDict['Epoch'], hpDict['Minibatch'],
               time.ctime()))

        assert hpDict['Optimizer'] in supportedOptimizer
        self.opt = optDictMap[hpDict['Optimizer']]  # create an instance of an optimizer class
        self.runid = None
        self.epoch = int(hpDict['Epoch'])
        self.learningrate = float(hpDict['Alpha'])
        self.keepProb= float(hpDict['KeepProb'])
        self.hiddenLayer=int(hpDict['HiddenLayer'])
        self.hiddenUnit=int(hpDict['HiddenUnit'])
        self.inputKeepProb=float(hpDict['InputKeepProb'])
        self.dpp = None

        assert hpDict['Regularization'] in supportedRegularization

        if hpDict['Regularization'] == 'None':
            self.regularization = None
        else:
            self.regularization = hpDict['Regularization']

        self.minibatch = int(hpDict['Minibatch'])  # 16384  # 8192
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
        net = tflearn.regression(net, optimizer=self.opt, loss='categorical_crossentropy', metric=acc,
                                 to_one_hot=True, n_classes=2)

        model = tflearn.DNN(net, tensorboard_dir="/tmp/tflearn_27thlogs/", tensorboard_verbose=0)

        self.model = model

    def make_core_network(self):
        # train a self.hiddenLayer layer neural network with self.hiddenUnit nodes in each layer

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
        net = tflearn.dropout(net, keep_prob=self.inputKeepProb, noise_shape=None, name='dropoutlayer0')

        for l in range(self.hiddenLayer):
            net = tflearn.fully_connected(net, self.hiddenUnit, activation='relu', weights_init=xavierInit,
                                          bias_init=normalInit, regularizer=self.regularization,
                                          weight_decay=0.001, name='hidderlayer%02d' % (l+1))
            net = tflearn.dropout(net, keep_prob=self.keepProb, noise_shape=None, name='dropoutlayer%02d' % (l+1))

        net = tflearn.fully_connected(net, 2, activation='softmax', weights_init=xavierInit, bias_init=normalInit,
                                      name='outputlayer')
        return net

        # Y = to_categorical(training_set.target, 3)

    def train(self, X, y, X_test, y_test):
        self.runid = tflearn.utils.id_generator()  # unique id for fit this model
        self.model.fit(X, y, validation_set=(X_test, y_test), shuffle=True, show_metric=True,
        #   self.model.fit(X, y, validation_set=0.01, shuffle=True, show_metric=True,
                       run_id=self.runid,batch_size=self.minibatch,
                       n_epoch=self.epoch, snapshot_epoch=False, snapshot_step=10000)

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

    def extractTrainTestFileDesc(self,hpDict):
        '''
        extract a descriptive name from  hpDict
        :return:
        string of Train
        '''
        pass

