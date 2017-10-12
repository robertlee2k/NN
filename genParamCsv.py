
import sys, getopt,os,csv


#testResultfile="DNN_Training_results.csv"
paramfile = "HyperParamSearchPlan.csv"
trainfilename = "/home/topleaf/stock/tensorFlowData/tensorFlowData(201101-201612).csv"   #training data file
testfilename = "/home/topleaf/stock/tensorFlowData/tensorFlowData(201701-201709).csv"  # test data  file

tFromDate="2011/01/01"
tToDate='2016/12/30'
testFromD='2016/12/30'
testToD='2017/09/19'

def usage():
    print ('genParam -o [output.csv]')
    print ("\nif output.csv is omitted, the default file name is %s" %paramfile)


class GenFile(object):
    def __init__(self,filename):
        self.columnsname = ["Seqno","Preprocessor", "Optimizer", "Regularization",'HiddenLayer','HiddenUnit',"InputKeepProb","KeepProb",
                            "Alpha", "lrdecay", "decaystep", "RS",
                            "Epoch", "Minibatch","Skip",
                            "Train","TFromDate","TToDate",
                            "Test","TestFromD","TestToD"]
        self.filename = filename
        if os.path.exists(filename) == False:
            with open(filename, 'w') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(self.columnsname)

    def append(self, rows):
        if os.path.exists(self.filename):
            with open(self.filename, 'ab+') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(rows)
    def generate(self):  #self define logic here
        seqno=0
        hiddenLayer=4
        #hiddenUnit=1000
        inputKeepProb=0.9
        minibatch=16384
        for preprocessor in ['Standard']:
            for opt in ['Adam']:
                for regularization in ['None','L2']:
                    for alpha in [0.001]:
                        for lrdecay in [0.99]:
                            for decaystep in [100]:
                                for epoch in [1000]:
                                    for hiddenUnit in [1000,500]:
                                        for rs in [39987]:
                                            for keepProb in [0.2,0.4,0.6,0.8,0.9]:

                                                row = ['%d'%seqno,preprocessor,opt,regularization,hiddenLayer,hiddenUnit,inputKeepProb,keepProb,alpha,lrdecay,
                                                   decaystep,rs,epoch,minibatch,'N',
                                                   trainfilename,tFromDate,tToDate,
                                                   testfilename,testFromD,testToD]
                                                self.append(row)
                                                seqno += 1
        print ('\n%s is generated' %self.filename)



def main():
    '''
    genParam -o output.csv
    :return:
    '''

    opts, args = getopt.getopt(sys.argv[1:], "ho:")
    for op, value in opts:
        if op == "-o":
            global paramfile
            paramfile = value
        elif op == "-h":
            usage()
            sys.exit()
        else:
            usage()
            sys.exit()

    if os.path.exists(paramfile):
        response= raw_input('\nthe output file %s exists, do you want to overwrite it ?(y/n)' %paramfile)
        if response == 'y' or response=='Y':
            try:
                os.remove(paramfile)
                GenFile(paramfile).generate()
            except Exception as ev:
                print 'horrible thing!'
                print Exception
                print ev
                sys.exit(-1)
        else:
            print ('\n Do not overwrite existing file, exit ...')
    else:
        GenFile(paramfile).generate()


if __name__=="__main__":
    main()