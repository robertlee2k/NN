
import sys, getopt,os,csv


#testResultfile="DNN_Training_results.csv"
paramfile = "HyperParamSearchPlan.csv"

def usage():
    print ('genParam -o [output.csv]')
    print ("\nif output.csv is omitted, the default file name is %s" %paramfile)


class GenFile(object):
    def __init__(self,filename):
        self.columnsname = ["Seqno","Preprocessor", "Optimizer", "Regularization",
                            "Alpha", "lrdecay", "decaystep", "RS",
                            "Epoch", "Minibatch"]
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
        for preprocessor in ['MinMax','Standard','MidRange']:
            for opt in ['Adam','Momentum','RMSProp','SGD']:
                for regularization in ['None', 'L2','L1']:
                    for alpha in [0.01,0.001,0.0001]:
                        for lrdecay in [0.99]:
                            for decaystep in [100]:
                                for epoch in [2000]:
                                    for minibatch in [16384]:
                                        for rs in [39987]:
                                            row = ['%d'%seqno,preprocessor,opt,regularization,alpha,lrdecay,
                                                   decaystep,rs,epoch,minibatch]
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