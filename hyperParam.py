
import csv
import os

class HyperParam(object):
    '''
    encapsulate all hyper parameters in this class
    :return: all required hyper parameters in a dictionary

    '''
    def __init__(self,filename):
        self.filename=filename
        self.rows=[]

        self.lastPreProcessor='None'
        self.preProcessorChanged = True

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

    def readRow(self,rowId):
        '''

        :param rowId:
        :return: row , whether or not preprocessor is different from previous readRow()
         update the member variables and return the param dictionary
        '''


        for row in self.rows:
            if long(row['Seqno']) == rowId:
                if row['Preprocessor'] != self.lastPreProcessor:
                    self.lastPreProcessor=row['Preprocessor']
                    self.preProcessorChanged=True
                else:
                    self.preProcessorChanged=False

                return row,self.preProcessorChanged
        raise ValueError("rowid doesn't exist in search plan %s" %self.filename)



