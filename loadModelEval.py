
"""
==================================================================================================
this program is a DNN stock models' consumer
its input: 1. ModelLoadEvalPlan.csv
           2. DNN_training_results.csv

output: 1. DNN_eval_results.csv.csv
    appending new test result to above record file


it read a model load/evaluate plan by the name of ModelLoadEvalPlan.csv, then load appropriate models
according to DNN_training_results.csv which is produced by model training producer
it also load test data according to the ModelLoadEvalPlan.csv
evaluate the trained model over the loaded test data,record their metrics, save the result back to
DNN_eval_results.csv
======================================================================================================
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

print(__doc__)
import time

# the following packages  are part of the project
from utility import log, plotFeatures,duration
from fetchData import FetchData
from hyperParam import supportedSkip
from modelStore import ModelStore,errorAnalysis
from modelEvalPlan import ModelEvalPlan

loadEvalPlan = "ModelLoadEvalPlan.csv"
# file is generated and reviewed manually, in the following format
# TFromDate :  specify the fromDate of  a model that is trained using data from this date
# TToDate   :  specify the toDate of a model that is trained using data to this date
# TestFromD :  specify the fromDate of a TestData set
# TestToD   :  specify the toDate of a TestData set
# MinAUC(Test) :   minimum AUC(Test) in DNN_Training_results.csv that a model has the TFromDate/TToDate
# MaxAUC(Test) : maximum AUC(Test)
# Seqno	Skip	TFromDate	TToDate	    TestFromD	TestToD	    MinAUC(Test) MaxAUC(Test)
# 0	    N	    2011/01/01	2016/12/31	2017/01/01	2017/09/19	0.6             0.8

def main():
    """
    this is a model consumer, it reads hyper parameters from ModelLoadEvalPlan.csv ,
    pick up matched model from DNN_Training_results.csv
    load the matched model and evaluate on specified testsets

    :return:
      its output are
      1. a csv file "DNN_Training_results.csv" that record the evaluation results of  all the picked up models ,
       including the model's hyper parameters, from/to date of training and testset

    """
    runStartTime = time.time()  # start time in ms.
    # instantiate a ModelEvalPlan class to read all designated trained model+test set combintion
    # plan into a list
    # do sanity check to make sure all parameters in the plan are valid before going further
    try:
        lp = ModelEvalPlan(loadEvalPlan)
        lp.sanityCheck()
    except ValueError as e:
        log("\nValueError Exception:")
        log(e.message)
        return
    except KeyError as e:
        log("KeyError Exception happened")
        log(e.message)
        return
    else:
        log("\n sanity check on Model Eval plan file(%s) PASSED " % loadEvalPlan)

    for seqid in range(0, lp.rows.__len__()):
        loopstartTime = time.time()  # start time in ms.
        st = time.ctime()  # time in date/time format
        try:
            itemDict = lp.readNextRow()
        except ValueError as e:
            log("WARNING: exception captured")
            log(e.message)
            log("\nSkip seqid=%d  and continue to next iteration, please double check your settings in %s"
                % (seqid, lp.filename))
            continue

        if itemDict['Skip'] == supportedSkip[2] or itemDict['Skip'] == supportedSkip[3]:  # this row is comment out, not run
            continue

        log('\nloading test data from:%s,to:%s in progress ... time:%s'
                % (itemDict["TestFromD"], itemDict["TestToD"], time.ctime()))
        try:
            test_set = FetchData().loadData(itemDict["TestFromD"], itemDict["TestToD"])
        except ValueError as e:
            log('=' * 30 + "ValueError Exception happened:" + '=' * 30)
            log(e.message)
            log('=' * 30 + "end of print ValueError exception" + '=' * 30)
            log("ValueError occurred in loading test data, skip this iteration")
            continue
        # plot original test data to review
        # plotFeatures(test_set.data,test_set.featurenames,[1],
        #               itemDict["TestFromD"]+'/'+itemDict["TestToD"],savePlotToDisk=True,scatterAdjust=False)

        # looking for required model and its accompanied dpp,hyperparam instance
        log("\n seqid=%d =====> looking for a trained Model fromDate:%s toDate:%s whose AUC(Val) is between %s and %s"
            % (seqid, itemDict['TFromDate'],itemDict['TToDate'],itemDict['MinAUC(Val)'],itemDict['MaxAUC(Val)']))
        try:
            modelst = ModelStore()
            modelst.getModelFullpath(itemDict['TFromDate'],
                                     itemDict['TToDate'],
                                     itemDict['MinAUC(Val)'],
                                     itemDict['MaxAUC(Val)'])
            if modelst.matchedModelLocations == []:
                log("Bypass this iteration since no satisfactory trainined model can be found from"
                    " %s" % modelst.testRecordName)
                continue
            log("\nFound %d trained model(s) !!!" % modelst.matchedModelLocations.__len__())
            log(str(modelst.matchedModelLocations))
            modelid = 1
            for dirname in modelst.matchedModelLocations:
                log('='*20 + "   seqid=%d,modelid=%d ---> using trained model at:%s   "
                    % (seqid,modelid, dirname) + '='*20)
                try:
                    # load the trained model and its accompanied hyperparam,data preprocess ,featureselection files
                    # from the retrieved location : dirname
                    modelst = modelst.load(dirname)
                    #  update hpDict record to replace original testFromD and TestToD dates when training
                    #  with newly loaded testset
                    modelst.hpDict['TestFromD'] = itemDict['TestFromD']
                    modelst.hpDict['TestToD'] = itemDict['TestToD']
                except ValueError as e:
                    log(e.message)
                    log("ValueError occured in loading a trained model, skip this iteration")
                    continue
                except IOError as e:
                    continue
                except Exception as e:  # any other exceptions, just skip this evaluation, not a big deal
                    continue

                # apply the same preprocess as used for training set to test_set
                log("\n apply datapreprocess %s to test_set" % modelst.dpp.__class__.__name__)
                X_test = modelst.dpp.transform(test_set)
                # plot transformed test data to review
                # plotFeatures(X_test,test_set.featurenames,[1],
                #               itemDict["TestFromD"]+'/'+itemDict["TestToD"],savePlotToDisk=True,scatterAdjust=False)
                y_test = test_set.target

                # apply the same feature selection as used for training set to test_set
                # if modelst.featureSel is not None:
                #     log("\n apply feature selector %s to test_set" % modelst.featureSel.__class__.__name__)
                #     X_test = modelst.featureSel.transform(X_test)
                    # plot transformed test data to review
                # plotFeatures(X_test,test_set.featurenames,[1],
                #               itemDict["TestFromD"]+'/'+itemDict["TestToD"],savePlotToDisk=True,scatterAdjust=False)

                try:
                    modelst.evaluateTestSet( X_test, y_test,itemDict)

                    # calculate the duration of this loop, update record
                    loopElapsedTime = duration(loopstartTime)
                    log("\nthe time of loading the model/evaluating the model is %s" % loopElapsedTime)
                    modelst.writeResult(modelst.hpDict, modelst.loadmymodel, st, time.ctime(), loopElapsedTime)

                    # visualize first 10 samples that have been wrongly classified for human review
                    errorAnalysis(modelst.loadmymodel, X_test, y_test, "Test data error analysis",10)
                except ValueError as ve:
                    log("Value Exception happened,bypass this iteration")
                    log(ve.message)
                    continue
                except IOError as ie:
                    log("IOError exception occured. bypass this iteration")
                    log(ie.message)
                    continue
                modelid += 1
            # reduce a reference to instance of DNNModel since its task has come to an end.
            # let system to gabage collect it
            del modelst
        except ValueError as ve:
            log("Value Error exception happened, can't find the Training model result file,abort execution.")
            log(ve.message)
            break

    wholetime = duration(runStartTime)
    log("\nthe WHOLE ELAPSED time of loading test data and evaluating all the models is %s"
        % wholetime)


if __name__ == "__main__":
    main()
