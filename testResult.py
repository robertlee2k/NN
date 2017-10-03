
import csv
import os

from utility import log

class TestResult(object):
    """use this class to automatically fill model training hyperparamaters and results into a csv file

    with block + open method has a good feature: when the with block is exit, open api will close this file
    automatically in its __exit__() method, so this file is correctly closed when not in use.
    """
    __instance = None  # define instance of the class

    # use the code to generate only one instance of the class
    def __new__(cls, *args, **kwargs):  # this method is called before __init__()
        if TestResult.__instance == None:
            TestResult.__instance = object.__new__(cls, *args, **kwargs)
        return TestResult.__instance

    def __init__(self,filename):
        self.columnsname=["Seqno","RunId","PreProcessor","Optimizer","Regularization",
                          "Alpha","lrdecay","decaystep","RS",
                          "AUC(Train)","Loss(Train)","Accuracy(Train)",
                          "AUC(Test)","Accuracy(Test)","NullAccuracy(Test)",\
                          "Duration","StartTime","EndTime","Epoch","Minibatch","ROC Curve Location"]
        self.filename=filename
        if os.path.exists(filename)==False :
            with open(filename,'w') as csv_file:
                writer=csv.writer(csv_file)
                writer.writerow(self.columnsname)
    def append(self,rows):
        if os.path.exists(self.filename):
            with open(self.filename,'ab+') as csv_file:
                writer=csv.writer(csv_file)
                writer.writerow(rows)
                log("\nUpdated TestResult record")
