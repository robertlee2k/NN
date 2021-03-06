import csv,os,datetime

from utility import log
from hyperParam import supportedSkip, DATAFILE_RANGE


class ModelEvalPlan(object):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if ModelEvalPlan._instance is None:
            ModelEvalPlan._instance = object.__new__(cls,*args,**kwargs)
        return ModelEvalPlan._instance

    def __init__(self,filename):
        self.evalplan = filename
        self.rows = []
        self.next = 0

        log("\nLoading and parsing file:%s" % filename)
        if os.path.exists(self.evalplan):
            with open(self.evalplan, 'r') as csv_file:
                mfile = csv.DictReader(csv_file)
                self.rows = [row for row in mfile]
                # rows include all the rows in a list as
                # [{'Seqno':'1','Skip':'N',"TFromDate":'2011/01/01', ... ...,'TestFromD':'2017/01/01'},
        else:
            raise ValueError("\n model evaluation plan file %s doesn't exist" % filename)

    def readNextRow(self):
        if self.next == len(self.rows):
            raise ValueError("end of the search plan %s reached" % self.evalplan)
        row = self.rows[self.next]
        self.next += 1

        if row['Skip'] == supportedSkip[2] or row['Skip'] == supportedSkip[3]:  # this row is comment out, not run
            log("\nSkip Seqno=%s  on purpose" % row['Seqno'])

        return row

    def sanityCheck(self):
        """
         make sure that all parameters settings read from the file are valid, otherwise, throw exceptions and send alarm NOW!!!
         an early alarm is much better for smooth running than a late one at run-time.
         :return:
         """
        errorFound = False
        for row in self.rows:
            try:
                if not row['Skip'] in supportedSkip:
                    log('\n Skip column must be left either N/n or Y/y in seq %s' % row['Seqno'])
                    errorFound = True

                # try conversion, if the raw data is in wrong format, the following will generate ValueError exception
                # time data 'xxxxxx' does not match format '%Y/%m/%d'
                # which will be captured by except clause
                stDate = datetime.datetime.strptime(row["TFromDate"], "%Y/%m/%d")
                toDate = datetime.datetime.strptime(row["TToDate"], "%Y/%m/%d")
                if stDate >= toDate:
                    raise ValueError("TFromDate could not be later than TToDate in Seqno %s," % row['Seqno'])
                if stDate < DATAFILE_RANGE['mindate'] or toDate > DATAFILE_RANGE['maxdate']:
                    raise ValueError("TFromDate or TToDate is beyond the available datafile range"
                                     " [%s - %s] in seq %s" % (DATAFILE_RANGE['mindate'],
                                                           DATAFILE_RANGE['maxdate'],
                                                           row['Seqno']))

                stDate = datetime.datetime.strptime(row["TestFromD"], "%Y/%m/%d")
                toDate = datetime.datetime.strptime(row["TestToD"], "%Y/%m/%d")
                if stDate >= toDate:
                    raise ValueError("TestFromD could not be later than TestToD in Seqno %s," % row['Seqno'])
                if stDate < DATAFILE_RANGE['mindate'] or toDate > DATAFILE_RANGE['maxdate']:
                    raise ValueError("TestFromD or TestToD is beyond the available datafile range"
                                     " [%s-%s] in seq %s" % (DATAFILE_RANGE['mindate'],
                                                           DATAFILE_RANGE['maxdate'],
                                                           row['Seqno']))

                tmp = int(row['Seqno'])
                tmpmin = float(row['MinAUC(Val)'])
                if tmpmin < 0.0 or tmpmin > 1.0:
                    raise ValueError("the range of MinAUC(Val) value in seq %s must be [0.0,1.0]" % row['Seqno'])
                tmpmax = float(row['MaxAUC(Val)'])
                if tmpmax < 0.0 or tmpmax > 1.0:
                    raise ValueError("the range of MaxAUC(Val) value in seq %s must be [0.0,1.0]" % row['Seqno'])
                if tmpmin>tmpmax:
                    raise ValueError("MinAux(Val) value is larger than MaxAux(Val) in seq %s" % row['Seqno'])
            except ValueError as e:
                log(e.message)
                errorFound = True
            except KeyError as e:
                errorFound = True
                raise KeyError(("\nFatal Error: KeyError happened, key %s is not found" % (e.message)))
        if errorFound:
            raise ValueError("Invalid parameters found in file %s, please correct them before rerun" % self.evalplan)

