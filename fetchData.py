
from tensorflow.python.platform import gfile
from sklearn.preprocessing import Imputer
import pandas as pd
import datetime
import collections
import csv
import numpy as np

#the following packages  are part of the project
from hyperParam import selectedAvgline, DATAFILE_RANGE
from utility import log,isIntersection

DATAFILEPATH = "/home/topleaf/stock/tensorFlowData/"

# make sure the files have meaningful names that reflect their data range!!!!  in alphabet order.
# DATAFILELIST = {"tensorFlowData(200501-201012).csv":
#                 (datetime.datetime.strptime("2005/01/01","%Y/%m/%d"),
#                 datetime.datetime.strptime("2010/12/31","%Y/%m/%d")),
#                 "tensorFlowData(201101-201612).csv":
#                 (datetime.datetime.strptime("2011/01/01","%Y/%m/%d"),
#                 datetime.datetime.strptime("2016/12/31","%Y/%m/%d")),
#                 "tensorFlowData(201701-201709).csv":
#                 (datetime.datetime.strptime("2017/01/01", "%Y/%m/%d"),
#                  datetime.datetime.strptime("2017/08/31", "%Y/%m/%d")),
#               }
DATAFILELIST = {"tensorFlowData200501-200712(group11).csv":
                (datetime.datetime.strptime("2005/01/04","%Y/%m/%d"),
                datetime.datetime.strptime("2007/12/27","%Y/%m/%d")),
                "tensorFlowData200801-201012(group11).csv":
                (datetime.datetime.strptime("2007/12/28","%Y/%m/%d"),
                datetime.datetime.strptime("2010/12/30","%Y/%m/%d")),
                "tensorFlowData201101-201312(group11).csv":
                (datetime.datetime.strptime("2010/12/31", "%Y/%m/%d"),
                 datetime.datetime.strptime("2013/12/30", "%Y/%m/%d")),
                "tensorFlowData201401-201612(group11).csv":
                (datetime.datetime.strptime("2013/12/31", "%Y/%m/%d"),
                datetime.datetime.strptime("2016/12/29", "%Y/%m/%d")),
                "tensorFlowData201701-201712(group11).csv":
                (datetime.datetime.strptime("2016/12/30", "%Y/%m/%d"),
                datetime.datetime.strptime("2017/12/28", "%Y/%m/%d")),
              }
DATASTART = 1     # the starting row# of the data file to be read
DATASTOP = 1500000  # the last row of the data file to be read, must set it to big enough
                    # use fromDate/toDate to specify the actual read row lines


# the following definition specified the column id# in original csv data file,starting from 0
DataDateColumn = 4
SelectedAvglineColumn = 6

Dataset = collections.namedtuple('Dataset', ['data', 'target', 'featurenames'])


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
    """
    this class handle data file fetch,
    its loadData method returns  rawdata in Dataset format
    input: fromDate, toDate
    internal

    """
    __instance = None       # define instance of the class

    # use the code to generate only one instance of the class
    def __new__(cls, *args, **kwargs):    # this method is called before __init__() only if the class is inherited from object
        if FetchData.__instance is None:
            FetchData.__instance = object.__new__(cls, *args, **kwargs)
        return FetchData.__instance

    def __init__(self):
        self.datafilelist = []
        # Create a list of features to dummy
        self.todummy_list = ['selected_avgline',   'zhishu_code']



    # get datafilelist according to fromDate and toDate
    def _getDatafileFullpath(self,fromDate,toDate):
        """
        looking for the data files' absolute full paths and return a list of those data file names
        :param fromDate:
        :param toDate:
        :return:  it update self.datafilelist member variable with
        a filename list that contains all files whose data are within fromDate,toDate range
        """
        stDate = datetime.datetime.strptime(fromDate, "%Y/%m/%d")
        endDate = datetime.datetime.strptime(toDate, "%Y/%m/%d")
        self.datafilelist = []

        for filename in DATAFILELIST.keys():
            if isIntersection(stDate,endDate,
                                    DATAFILELIST[filename][0], DATAFILELIST[filename][1]):
                self.datafilelist.append(''.join((DATAFILEPATH, filename)))

        # this might be a redundant check, the date validity should have been verified when calling
        # hyperParam.sanitycheck() upon starting the program.
        # anyway, just keep it to be more alertive to abnormal date input
        if stDate<DATAFILE_RANGE["mindate"] or endDate >DATAFILE_RANGE['maxdate']:
            raise ValueError("the fromDate %s is earlier than date of available datafiles"
                             " or toDate %s is later than date of available datafiles"
                             %(fromDate, toDate))

        datafilenum = self.datafilelist.__len__()
        if datafilenum == 0:
            raise ValueError(" failed to locate any data files between %s and %s!!!"
                             %(fromDate, toDate))

        self.datafilelist.sort()
        log(" %d data file(s) located:" % datafilenum)
        for i in range(datafilenum):
            log("%d: " % (i + 1) + str(self.datafilelist[i]))


    # Function to dummy all the categorical variables used for modeling
    def _dummy_df(self,df, todummy_list):
        for x in todummy_list:
            dummies = pd.get_dummies(df[x], prefix=x, dummy_na=False)
            df = df.drop(x, 1)
            df = pd.concat([df, dummies], axis=1)
        return df

    def loadData(self, fromDate, toDate):
        """
        this method handle loading data from files to memory in datasets format and return it
        the logic of which columns , rows, dates are loaded can be customized in this method.
        2018/1/31 add data cleaning functionality into this method, to
        :param fromDate:
        :param toDate:
        :return: Dataset format structure in memory
        """
        # Load datasets. discard column 0,1,3,4 in the original csvfile ,which represent  id ,tradedate,mc_date,datadate

        try:
            self._getDatafileFullpath(fromDate, toDate)
            # self.datafilelist is loaded with a sequence of datafile names that contains data between
            # fromData and toDate, the following code is to iterate those files, load data,
            # concatenate them into a fullData,fullTarget numpy array.
            datafilename = self.datafilelist.pop(0)
            fullDf, fullTarget = self.loadcsv(
                filename=datafilename,
                target_dtype=np.int,
                features_dtype=np.float32,
                fromDate=fromDate,
                toDate=toDate,
                discard_columns=['id','tradeDate','code','mc_date','shouyilv'],
                # add -1 in the last column to exclude the percentage infor,exclude 2# stockcode,
                target_column='positive',
                selectedAvgline=selectedAvgline
            )
            #
            # fullData, fullTarget, feature_names = self.load_partcsv_without_header(
            #     filename=datafilename,
            #     target_dtype=np.int,
            #     features_dtype=np.float32,
            #     start_rowid=DATASTART,
            #     end_rowid=DATASTOP,
            #     fromDate=fromDate,
            #     toDate=toDate,
            #     discard_colids=[0, 1, 2, 3, 4, -1],
            #     # add -1 in the last column to exclude the percentage infor,exclude 2# stockcode,
            #     target_column=5,
            #     filling_value=1.0,
            #     selectedAvgline=selectedAvgline
            # )
            for datafilename in self.datafilelist:
                # partData, partTarget, feature_names = self.load_partcsv_without_header(
                #     filename=datafilename,
                #     target_dtype=np.int,
                #     features_dtype=np.float32,
                #     start_rowid=DATASTART,
                #     end_rowid=DATASTOP,
                #     fromDate=fromDate,
                #     toDate=toDate,
                #     discard_colids=[0, 1, 2, 3, 4, -1],
                #     # add -1 in the last column to exclude the percentage infor,exclude 2# stockcode,
                #     target_column=5,
                #     filling_value=1.0,
                #     selectedAvgline=selectedAvgline
                # )
                partDf, partTarget = self.loadcsv(
                    filename=datafilename,
                    target_dtype=np.int,
                    features_dtype=np.float32,
                    fromDate=fromDate,
                    toDate=toDate,
                    discard_columns=['id', 'tradeDate', 'code', 'mc_date', 'shouyilv'],
                    # add -1 in the last column to exclude the percentage infor,exclude 2# stockcode,
                    target_column='positive',
                    selectedAvgline=selectedAvgline
                )
                # fullDf = np.concatenate((fullDf, partDf),axis=0)
                fullDf=pd.concat([fullDf,partDf])
                fullTarget = np.concatenate((fullTarget, partTarget), axis=0)

        except ValueError as e:
            raise ValueError(e)  #capture and throw the exception to the caller
        else:
            df=self._dummy_df(fullDf,self.todummy_list)
            log("apply dummy_df to change categorical features,after that the feature number is (%d,%d) " %(df.shape[0],df.shape[1]))
            return Dataset(data=df.values, target=fullTarget, featurenames=list(df.columns))

    def loadcsv(self,filename,target_dtype,features_dtype,fromDate,toDate,discard_columns,
                target_column='positive',
                selectedAvgline=selectedAvgline
                ):
        """
        load csv using pandas method,filling missing values with mean of this column
        :param filename:
        :param target_dtype:
        :param features_dtype:
        :param fromDate:
        :param toDate:
        :param discard_columns:
        :param target_column:
        :param selectedAvgline:
        :return:
        """
        log("Loading data to memory from %s using Pandas,this will take a while ... ..." % filename)
        dateparse = lambda dates: pd.datetime.strptime(dates, '%Y/%m/%d')

        # set 'dataDate' column as date, use it as index, '?' is the missing value, change it to 'NaN' after reading
        # set dtype of 6 specified columns to be int, others will be float64
        df = pd.read_csv(filename,
                                 na_values=['?'],
                                 parse_dates=['dataDate'],
                                 index_col='dataDate',
                                 date_parser=dateparse,
                                 dtype={'selected_avgline':np.int8,'positive':np.int8,'zhishu_code':np.int8,
                                        'is_st':np.int8,'ishs300':np.int8,'iszz500':np.int8,'zhangdieting':np.int8})
        originalrows=df.shape[0]
            #  only keep desired data between 'From' and 'To' date
        df = df[fromDate:toDate]
        if df.shape[0] == 0:  # no data is loaded
            raise ValueError("No data is loaded, check your %s and fromdate:%s,ToDate%s" % (filename, fromDate, toDate))
        log("load %d rows out of %d" %(df.shape[0],originalrows))
        # discard undesired columns
        for col in discard_columns:
            del df[col]

        y = df.pop(target_column)

        #  here , do sanity check, filling missing values with mean value of this column
        # log("filling missing values with mean value of the column")
        # imp = Imputer(missing_values='NaN',strategy="mean",axis=0)
        # imp.fit(df)
        #
        # # print ('before imputer: %s' %(df.isnull().sum().sort_values()))
        # df=pd.DataFrame(data=imp.transform(df),columns=df.columns)
        # # dtype is changed to float64 by the imputer, now convert dtype of specified columns to our required type
        # for column in ['is_st','ishs300', 'iszz500', 'zhangdieting']:
        #     df[column]=df[column].astype(bool)
        # for column in [ 'selected_avgline','zhishu_code']:
        #     df[column] = df[column].astype(np.int8)     # change to onehot encoder later
        # for column in ['circulation_marketVal_gears', 'leiji_ma10_top_days',
        #                        'leiji_ma20_top_days', 'leiji_ma30_top_days',
        #                        'leiji_ma5_top_days', 'leiji_ma60_top_days']:
        #     df[column]=df[column].astype(np.int64)

       # print ('after imputer: %s' % (df.isnull().sum().sort_values()))

        #return df.values,y.values,list(df.columns)
        return df,y.values

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
        log("Loading data to memory from %s ,this will take a while ... ..." % filename)
        with gfile.Open(filename) as csv_file:
            data_file = csv.reader(csv_file)
            data, target = [], []
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
                        # only load the required rows that match the avgline parameters
                        if row[SelectedAvglineColumn] in selectedAvgline:
                            target.append(row.pop(target_column))
                            for j in sorted(discard_colids,reverse=True):
                                del row[j]  # delete the columns whose indexes are in list discard_colids
                            data.append(row)
                            hitCount += 1
                            # else:
                            #     print('Attention: this row[%d] SelectedAvgline is not in %s,so  discard this row' %(i,selectedAvgline))
                            # else:
                            # print ('attention:  this row[%d] dataDate %s is not between [%s,%s],so discard this row' %(i, row[DataDateColumn],fromDate,toDate))

            log('\ntotal row# read from this data file %s is (%d out of %d)' % (filename, hitCount, i))
            if missingvaluecount != 0:
                log("\n!!! WARNING: the input data file %s has %d blank value(s),"
                    "setting its value to %d as a workaround, please check and fill them later!!!"
                    % (filename, missingvaluecount, filling_value))
            if hitCount == 0:     #no data is loaded
                raise ValueError("No data is loaded, check your %s and fromdate:%s,ToDate%s" %(filename,fromDate,toDate))

            data = np.array(data, dtype=features_dtype)
            target = np.array(target, dtype=target_dtype)

        return data, target, featureNames

