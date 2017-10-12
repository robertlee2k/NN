

# the following definition specified the column id# in original csv data file,starting from 0
DataDateColumn = 4
SelectedAvglineColumn=6

from tensorflow.python.platform import gfile
import datetime
import collections
import csv
import numpy as np

#the following packages  are part of the project
from hyperParam import selectedAvgline
from utility import log
Dataset = collections.namedtuple('Dataset',['data','target','featurenames'])




def load_csv_calc_profit(filename,
                         start_rowid,
                         end_rowid,
                         fromDate,
                         toDate,
                         target_column=-1,
                         selectedAvgline=selectedAvgline):
    '''

    :param filename: data file name
    :param start_rowid:
    :param end_rowid:
    :param fromDate:
    :param toData:
    :param target_column: shouyilv column' index
    :param selectedAvgline:  which avglines are included in the calculation
    :return:  the simple profit/loss rate for all the transactions
    '''

    log("Loading %s and Calculating the overall profit rate when selecting %s Avglines..." %(filename,selectedAvgline))
    with gfile.Open(filename)  as csv_file:
        data_file=csv.reader(csv_file)

        hitCount = 0  # the total row number that has been calculated
        stDate = datetime.datetime.strptime(fromDate, "%Y/%m/%d")
        endDate = datetime.datetime.strptime(toDate, "%Y/%m/%d")
        amount=[]     #initial investment, y axis
        x=[]        #x axis , valid transaction number
        capital=1.0

        for i, row in enumerate(data_file):
            if i < start_rowid:  # or i > end_rowid:
                continue  # skip the rows between start_rowid and end_rowid
            elif i > end_rowid:
                log("\n WARNING: skipping rows in %s after line# %d, NOT FULL data are loaded" % (filename, i))
                break  # skip reading the rest of the file
            else:
                dt = datetime.datetime.strptime(row[DataDateColumn],
                                                "%Y/%m/%d")  # convert DataDate from string to datetime format
                if dt >= stDate and dt <= endDate:  # read the desired data between 'From' and 'To' date
                    if row[SelectedAvglineColumn] in selectedAvgline:  # only load the required rows that match the avgline parameters
                        capital=capital*(1+float(row[target_column]))
                        amount.append(capital)
                        hitCount += 1
                        x.append(hitCount)
                        if capital <=0.01:  # if lost 99% of the initial investment
                            break  # skip the calculation and plot how the disaster happened
                    else:
                        print('Attention: this row[%d] SelectedAvgline is not in %s,so  discard this row' %(i,selectedAvgline))
                else:
                    print ('attention:  this row[%d] dataDate %s is not between [%s,%s],so discard this row' %(i, row[DataDateColumn],fromDate,toDate))
        log("\nThe overall profit/loss ratio in %s file with transactions %d out of %d is:" %(filename,hitCount,i))
        rate=(capital-1.0)/1.0
        log(" {}%".format("%0.4f" %(100*rate)))
        from utility import plotAmountOverTransaction
        plotAmountOverTransaction(x,amount,filename+" Selected Avgline="+selectedAvgline.__str__())



class FetchData(object):
    __instance = None       #define instance of the class

    #use the code to generate only one instance of the class
    def __new__(cls, *args, **kwargs):    # this method is called before __init__() only if the class is inherited from object
        if FetchData.__instance == None:
            FetchData.__instance = object.__new__(cls, *args, **kwargs)
        return FetchData.__instance

    def __init__(self,dataFilename,fromDate,toDate,startRow,endRow):
        self.filename=dataFilename
        self.fromDate=fromDate
        self.toDate=toDate
        self.startRow=startRow
        self.endRow=endRow


    def loadData(self):
        try:
            rawData = self.load_partcsv_without_header(
                filename=self.filename,
                target_dtype=np.int,
                features_dtype=np.float32,
                start_rowid=self.startRow,
                end_rowid=self.endRow,
                fromDate=self.fromDate,
                toDate=self.toDate,
                discard_colids=[0, 1, 3, 4, -1],
                # add -1 in the last column to exclude the percentage infor,include 2# stockcode,
                target_column=5,
                filling_value=1.0,
                selectedAvgline=selectedAvgline
            )
        except ValueError as e:
            raise ValueError(e)  #capture and throw the exception to the caller
        else:
            return rawData

    def load_partcsv_without_header(self,filename,
                                    target_dtype,
                                    features_dtype,
                                    start_rowid,
                                    end_rowid,
                                    fromDate,
                                    toDate,
                                    discard_colids,
                                    target_column=-1,
                                    filling_value=1.0,
                                    selectedAvgline=selectedAvgline):
        """Load dataset from CSV file without a header row to indicate how many records in this files. the format of the csv is:
        row 0:  column names
        row 1 ~ n: column  data

        fromDate format: '2017/3/13'
        selectedAvgline is a tuple that include all desired data whose selectedAvgline column match the content in the tuple,
        """
        with gfile.Open(filename) as csv_file:
            data_file = csv.reader(csv_file)
            data, target = [], []
            dest_rowid = 0
            missingvaluecount = 0
            hitCount = 0  # the total row number that has been read into data
            featureNames = []  # a list to hold all required feature names

            stDate = datetime.datetime.strptime(fromDate, "%Y/%m/%d")
            endDate = datetime.datetime.strptime(toDate, "%Y/%m/%d")

            for i, row in enumerate(data_file):
                # check the content of the row and print out if there's  missing value
                for k in xrange(0, row.__len__()):
                    if row[k] == '?':
                        # log("\n ? value in record id:%s,(stockcode = %s) column#=%d" % (row[0], row[2], k+1 ))
                        row[k] = filling_value  # setting default value to those missing values
                        missingvaluecount += 1
                if i == 0:  # put the required feature names into a list
                    row.pop(target_column)  # discard the target_column name
                    for j in sorted(discard_colids,
                                    reverse=True):  # delete the columns whose indexes are in list discard_colids
                        del row[j]
                    featureNames.extend(row)  # add multiple values at once from  row list to my featurename list

                if i < start_rowid:  # or i > end_rowid:
                    continue  # skip the rows between start_rowid and end_rowid
                elif i > end_rowid:
                    log("\n WARNING: skipping rows in %s after line# %d, NOT FULL data are loaded" % (filename, i))
                    break  # skip reading the rest of the file
                else:
                    dt = datetime.datetime.strptime(row[DataDateColumn],
                                                    "%Y/%m/%d")  # convert DataDate from string to datetime format
                    if dt >= stDate and dt <= endDate:  # read the desired data between 'From' and 'To' date
                        if row[
                            SelectedAvglineColumn] in selectedAvgline:  # only load the required rows that match the avgline parameters
                            target.append(row.pop(target_column))
                            for j in sorted(discard_colids,
                                            reverse=True):  # delete the columns whose indexes are in list discard_colids
                                del row[j]
                            data.append(row)
                            hitCount += 1
                            # else:
                            #     print('Attention: this row[%d] SelectedAvgline is not in %s,so  discard this row' %(i,selectedAvgline))
                            # else:
                            # print ('attention:  this row[%d] dataDate %s is not between [%s,%s],so discard this row' %(i, row[DataDateColumn],fromDate,toDate))

            log('\ntotal row# read from this data file %s is (%d out of %d)' % (filename, hitCount, i))
            if missingvaluecount != 0:
                log(
                    "\n!!! WARNING: the input data file %s has %d blank value(s),setting its value to %d as a workaround, please check and fill them later!!!" % (
                    filename, missingvaluecount, filling_value))
            if hitCount==0:     #no data is loaded
                raise ValueError("No data is loaded, check your %s and fromdate:%s,ToDate%s" %(filename,fromDate,toDate))

            data = np.array(data, dtype=features_dtype)
            target = np.array(target, dtype=target_dtype)

        return Dataset(data=data, target=target, featurenames=featureNames)
