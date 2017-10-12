
import csv
import os
import datetime

from utility import log
supportedOptimizer= ('Adam', 'Momentum', 'RMSProp','SGD')
supportedScaler =('MinMax', 'Standard', 'MidRange')
supportedRegularization = ('None','L2','L1')
supportedSkip=('N','n','Y','y')  #  'y'or 'Y' mark the row as skipping
selectedAvgline=('5','10','20','30','60')  # only load training and test data whose avglines are in this tuple

class HyperParam(object):
    '''
    encapsulate all hyper parameters in this class
    :return: all required hyper parameters in a dictionary

    '''
    def __init__(self,filename):
        self.filename=filename
        self.rows=[]

        self.lastPreProcessor='Initial' # to make sure it's different from any valid preprocessor in system
        self.preProcessorChanged = True

        log("\nLoading and parsing file:%s" %filename)
        if os.path.exists(self.filename):
            with open(self.filename, 'r') as csv_file:
                hpfile = csv.DictReader(csv_file)
                self.rows = [row for row in hpfile]
                # rows include all the row in a list as
                # [{'seqno':'1','Preprocessor':'MinMax',"Optimizer":'Adam', ... ...,'Minibatch':'1000'},
                # {'seqno':'2','Preprocessor':'Standard','Optimizer':'Adam', ... ...,'Minibatch':'2000'},
                #  {'seqno':'3','Preprocessor':'MidRange','Optimizer':"Momentum', ... ...,'Minibatch':'3000'},]
                # do some check to make sure its validaity here, and do name maping here
                # eg. MinMaxScaler, StandardScale,MidrangeScaler,Adam,Momentum,RMSProp etc
        else:
            raise ValueError("\n search plan file %s doesn't exist" %filename)

    def readRow(self,rowId):
        '''

        :param rowId:
        :return: row , whether or not preprocessor is different from previous readRow()
         update the member variables and return the param dictionary
        '''


        for row in self.rows:
            if int(row['Seqno']) == rowId:
                if row['Skip'] == supportedSkip[2] or row['Skip'] == supportedSkip[3]:  # this row is comment out, not run
                    log("\nSkip Seqno=%d  on purpose" % rowId)

                else:
                    if row['Preprocessor'] != self.lastPreProcessor:
                          self.lastPreProcessor = row['Preprocessor']
                          self.preProcessorChanged = True
                    else:
                          self.preProcessorChanged = False
                return  self.preProcessorChanged, row

        raise ValueError("rowid doesn't exist in search plan %s" %self.filename)

    def sanityCheck(self):
        '''
        make sure that all parameters settings read from the file are valid, otherwise, throw exceptions and send alarm NOW!!!
        an early alarm is much better for smooth running than a late one at run-time.
        :return:
        '''
        errorFound=False
        for row in self.rows:
            try:
                if not row['Preprocessor'] in supportedScaler or not row['Regularization'] in supportedRegularization \
                    or not row['Optimizer'] in supportedOptimizer:
                    log('\nFatal Error: wrong preprocess or scaler or regularization names in seq %s' %(row['Seqno']))
                    errorFound=True
                if not row['Skip'] in supportedSkip:
                    log('\n Skip column must be left either N/n or Y/y in seq %s' %row['Seqno'])
                    errorFound=True
                if not os.path.exists(row["Train"]) or not os.path.exists(row['Test']):
                    log('\n Train file or Test file in seq %s does NOT exist' % row['Seqno'])
                    errorFound = True
                # try conversion, if the raw data is in wrong format, the following will generate ValueError exception
                # time data 'xxxxxx' does not match format '%Y/%m/%d'
                # which will be captured by except clause
                stDate = datetime.datetime.strptime(row["TFromDate"], "%Y/%m/%d")
                toDate =  datetime.datetime.strptime(row["TToDate"], "%Y/%m/%d")
                stDate = datetime.datetime.strptime(row["TestFromD"], "%Y/%m/%d")
                toDate = datetime.datetime.strptime(row["TestToD"], "%Y/%m/%d")


                tmp=int(row['Seqno'])+int(row['decaystep'])+int(row['RS'])+int(row['Epoch'])+int(row['Minibatch'])\
                    +int(row["HiddenLayer"])+ int(row['HiddenUnit'])
                tmp=float(row['Alpha'])+float(row['lrdecay'])+float(row['KeepProb'])+float(row['InputKeepProb'])
            except ValueError as e:
                log(e.message)
                log("\nFatal Error: wrong data type in seq %s" %(row['Seqno']))
                errorFound=True
            except KeyError as e:
                errorFound=True
                raise KeyError(("\nFatal Error: KeyError happened, key %s is not found" %(e.message)))
        if errorFound:
            print ("\n valid keywords should be in the following tuples:")
            print (supportedScaler, supportedOptimizer, supportedRegularization)
            raise ValueError ("Invalid parameters found in file %s, please correct them before rerun" %self.filename)



