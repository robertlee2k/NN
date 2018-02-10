
import csv
import os
import datetime

from utility import log,isIntersection


DATAFILE_RANGE = {"mindate": datetime.datetime.strptime("2005/01/01","%Y/%m/%d"),
                  "maxdate": datetime.datetime.strptime("2017/12/28", "%Y/%m/%d")}
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
        self.rows = []
        self.next = 0   # the index of the rows list to be read

        # to make sure it's different from any valid preprocessor in system
        self.lastPreProcessor = 'Initial'
        self.lastTFromDate = 'Initial'
        self.lastTToDate = 'Initial'
        self.lastTestFromD = self.lastTestToD = 'Initial'
        self.validationdays="Initial"

        self.preProcessorChanged = True
        self.trainDataChanged=True
        self.testDataChanged=True
        self.validationdaysChanged=True

        log("\nLoading and parsing file:%s" %filename)
        if os.path.exists(self.filename):
            with open(self.filename, 'r') as csv_file:
                hpfile = csv.DictReader(csv_file)
                self.rows = [row for row in hpfile]
                # rows include all the row in a list as
                # [{'seqno':'1','Preprocessor':'MinMax',"Optimizer":'Adam', ... ...,'Minibatch':'1000'},
                # {'seqno':'2','Preprocessor':'Standard','Optimizer':'Adam', ... ...,'Minibatch':'2000'},
                #  {'seqno':'3','Preprocessor':'MidRange','Optimizer':"Momentum', ... ...,'Minibatch':'3000'},]
                # do some check to make sure its validaity here,
                #
        else:
            raise ValueError("\n search plan file %s doesn't exist" %filename)

    def readNextRow(self):
        """
        :param rowId:
        :return: row , whether or not Train,TFromDate,TToDate,Test,TestFromD,TestToD,Preprocessor is different
         from previous readRow()
         update the member variables and return the param dictionary
        """
        if self.next == len(self.rows):
            raise ValueError("end of the search plan %s reached" % self.filename)

        row = self.rows[self.next]
        self.next += 1

        if row['Skip'] == supportedSkip[2] or row['Skip'] == supportedSkip[3]:  # this row is comment out, not run
            log("\nSkip Seqno=%s  on purpose" % row['Seqno'])
        else:   # check if  this row has significant param change from last row
            if row['Preprocessor'] != self.lastPreProcessor:
                self.lastPreProcessor = row['Preprocessor']
                self.preProcessorChanged = True
            else:
                self.preProcessorChanged = False

            if row['TFromDate'] != self.lastTFromDate or row['TToDate'] != self.lastTToDate:
                self.lastTFromDate = row['TFromDate']
                self.lastTToDate = row['TToDate']
                self.trainDataChanged = True
            else:
                self.trainDataChanged = False

            if row['TestFromD'] != self.lastTestFromD or row['TestToD'] != self.lastTestToD:
                self.lastTestFromD = row['TestFromD']
                self.lastTestToD = row['TestToD']
                self.testDataChanged = True
            else:
                self.testDataChanged = False

            if row['ValidationDays'] != self.validationdays:
                self.validationdays = row['ValidationDays']
                self.validationdaysChanged = True
            else:
                self.validationdaysChanged = False

            #append ValidationSetDate and append to row
            stDate = datetime.datetime.strptime(row["TestFromD"], "%Y/%m/%d")
            stValidationDate = stDate - datetime.timedelta(int(row['ValidationDays']))
            toValidationDate = stDate - datetime.timedelta(1)
            row["ValidationFromD"] = datetime.datetime.strftime(stValidationDate,"%Y/%m/%d")
            row["ValidationToD"] = datetime.datetime.strftime(toValidationDate,"%Y/%m/%d")
        return self.validationdaysChanged,self.preProcessorChanged, \
               self.trainDataChanged, self.testDataChanged, row

    def sanityCheck(self):
        """
        make sure that all parameters settings read from the file are valid, otherwise, throw exceptions and send alarm NOW!!!
        an early alarm is much better for smooth running than a late one at run-time.
        :return:
        """
        errorFound = False
        for row in self.rows:
            try:
                if not row['Preprocessor'] in supportedScaler or \
                        not row['Regularization'] in supportedRegularization \
                        or not row['Optimizer'] in supportedOptimizer:
                    log('\nFatal Error: wrong preprocess or scaler or regularization names in seq %s' %(row['Seqno']))
                    errorFound = True
                if not row['Skip'] in supportedSkip:
                    log('\n Skip column must be left either N/n or Y/y in seq %s' %row['Seqno'])
                    errorFound = True
                # try conversion, if the raw data is in wrong format, the following will generate ValueError exception
                # time data 'xxxxxx' does not match format '%Y/%m/%d'
                # which will be captured by except clause
                stTrainDate = datetime.datetime.strptime(row["TFromDate"], "%Y/%m/%d")
                toTrainDate = datetime.datetime.strptime(row["TToDate"], "%Y/%m/%d")
                if stTrainDate > toTrainDate:
                    raise ValueError("TFromDate could not be later than TToDate in Seqno %s," % row['Seqno'])
                if stTrainDate < DATAFILE_RANGE['mindate'] or toTrainDate > DATAFILE_RANGE['maxdate']:
                    raise ValueError("TFromDate or TToDate is beyond the available datafile range"
                                     " [%s - %s] in seq %s" % (DATAFILE_RANGE['mindate'],
                                                           DATAFILE_RANGE['maxdate'],
                                                           row['Seqno']))

                stDate = datetime.datetime.strptime(row["TestFromD"], "%Y/%m/%d")
                toDate = datetime.datetime.strptime(row["TestToD"], "%Y/%m/%d")
                if stDate > toDate:
                    raise ValueError("TestFromD could not be later than TestToD in Seqno %s," % row['Seqno'])
                if stDate < DATAFILE_RANGE['mindate'] or toDate > DATAFILE_RANGE['maxdate']:
                    raise ValueError("TestFromD or TestToD is beyond the available datafile range"
                                     " [%s-%s] in seq %s" % (DATAFILE_RANGE['mindate'],
                                                           DATAFILE_RANGE['maxdate'],
                                                           row['Seqno']))
                tmp = int(row['Seqno'])+int(row['decaystep'])+int(row['RS'])+int(row['Epoch'])+int(row['Minibatch'])\
                    + int(row["HiddenLayer"]) + int(row['HiddenUnit'])
                tmp = float(row['Alpha'])+float(row['lrdecay'])
                tmp = float(row['KeepProb'])
                if tmp < 0.1  or tmp > 1.0:
                    raise ValueError("in Seq %s: KeepProb must be a float between 0.01 and 1.0" % row['Seqno'])
                tmp = float(row['InputKeepProb'])
                if tmp < 0.1  or tmp > 1.0:
                    raise ValueError("in Seq %s: InputKeepProb must be a float between 0.01 and 1.0" % row['Seqno'])

                valday = int(row['ValidationDays'])
                if valday <=0:
                    raise ValueError("in Seq %s: ValidationDays must be a integer that's greater than 0" %row['Seqno'])

                # make sure the validation period, Test period, and training period do NOT have intersections
                stValidationDate = stDate - datetime.timedelta(int(row['ValidationDays']))
                toValidationDate = stDate - datetime.timedelta(1)
                if isIntersection(stTrainDate,toTrainDate,stDate,toDate) or \
                   isIntersection(stTrainDate,toTrainDate,stValidationDate,toValidationDate) :
                    raise ValueError("Invalid setting: in Seqno %s:[TestFromD-TestToD],[TFromDate-TToDate],"
                                     "Validationdays overlaps." % row['Seqno'])
            except ValueError as e:
                log(e.message)
                log("Fatal Error: wrong data in seq %s" %(row['Seqno']))
                errorFound = True
            except KeyError as e:
                errorFound = True
                raise KeyError(("Fatal Error: KeyError happened, key %s is not found" % e.message))
        if errorFound:
            # print ("valid keywords should be in the following tuples:")
            # print (supportedScaler, supportedOptimizer, supportedRegularization)
            raise ValueError ("Invalid parameters found in file %s, please correct them before rerun" % self.filename)



