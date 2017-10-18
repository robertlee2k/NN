
from tensorflow.python.platform import gfile
from sklearn.metrics import roc_curve,auc
import time
import matplotlib.mlab as mlab
import numpy as np
import matplotlib.pyplot as plt

logfilename = "/tmp/stockNN2.log"

def duration(startTime):
    """
    calculation passed time from startTime
    :param startTime: a time of time.time() in ms
    :return: str format of elapsed time
    """
    endTime = time.time()  # end time in ms.
    elapseTime = (endTime - startTime)
    hour = int(elapseTime / 3600)
    minute = int((elapseTime % 3600) / 60)
    second = int((elapseTime % 3600) % 60)
    passedTime = "%dh%d'%d''" % (hour, minute, second)
    return passedTime

#write info to log file
def log(info,logfilename=logfilename):
    with gfile.Open(logfilename,'a+') as logfileid:
        logfileid.write(info)
        print ("%s" %info)

#get roc_auc for predicted data with label =1
def get_roc_auc(y_true,y_pred_prob):
    fpr,tpr,thresholds= roc_curve(y_true,y_pred_prob[:,1])
    roc_auc = auc(fpr,tpr)
    return roc_auc

def printConfusionMatrix(cm):
    """
    format the cm string before plot it to ROC curve
    calculate and show FPR and TPR value for label=1 class

    :param cm: confusion matrix    the value of cm[i,j] is the number that a trueclass i is predicted to be class j
                          predict=0  predict=1 .... predict=n_class
       truelabel=0
       truelabel=1
       ...
       truelabel=n_class
       FPR(1) =    , TPR(1) =

    :return: a string of above format
    """
    N1=0.0     # the total element number of truelabel is not 1
    output="\n               "
    row,col= cm.shape
    for j in range(col):
        output=output+"  predict=%d" %j
    output+="\n"   # print first title line
    for i in range(row):
        output += "truelabel=%d" %i
        for j in range(col):
            output +="  % 9d" %cm[i][j]
        if i!=1:
            N1 = N1+cm[i, :].sum()  # the sum of  truelabel!=1
        else:   #  i == 1, for those truelabel=1 data, calculate its tpr and fpr
            FP1=cm[:,i].sum()-cm[i,i]    # for all predict=1 data, minus true positive cm[1,1]
            TP1=cm[i,i]
            P1=cm[i,:].sum()   # the sum of the truelabel=1
        output+="\n"
    FPR1=float(FP1)/N1
    TPR1=float(TP1)/P1
    output+="FPR(1)=%0.4f, TPR(1)=%0.4f\n"%(FPR1,TPR1)
    return output

#visualize  the batch features in both a scatter subplot to review the min-max range of the features in a whole picture
# and a subplot of histogram for each features' distribution

def plotFeatures(batch,datasetFeatureNames,featureidlist,desc=None,savePlotToDisk=True,scatterAdjust=False):
    """
batch: is 2-D matrix [n_sample,n_feature]
datasetFeatureNames: a list that contains all the feature names in the same order as batch
desc is the string that describes the data,such as "2013-15training" ,will be used as part of plot title.
savePlotTodisk : whether or not save the plots to disk
scatterAdjust : whether scale Y scale to fit for current min/max , set it to False if you want to compare all features'
absolute min/max value in one scatter plot
    """
    xscatter=[]   # hold the x axis coordinates, which is column id#
    yscatter=[]   # hold a tuple with (minvalue,maxvalue) for that x
    n_sample,n_feature= batch.shape

    # get and fill  min and max value for each features,prepare data for scatter plotting later
    for j in range(n_feature):
        y = batch[:, j]  # fetch all items in this column
        xscatter.append(j)
        yscatter.append((y.min(),y.max()))
        # xscatter.append(j)
        # yscatter.append(y.max())

    assert type(featureidlist)==list

    for j in featureidlist: # for all required features

        log("Plotting %s feature# %d in progress,time = %s " %(desc,j,(time.ctime())))
        featureFig = plt.figure("plotFeature"+str(j)+"_"+str(datasetFeatureNames[j]), figsize=(8, 6))
        featureFig.subplots_adjust(top=0.92, left=0.10, right=0.97,hspace=0.37, wspace=0.3)

        axf = featureFig.add_subplot(2, 1, 1)  # histogram each feature to check its distribution
        y= batch[:,j]  #fetch all items in this column

        axf.clear()
        #axn.clear()   #clear previous histogram plot
        # bins=[]    #prepare a list to seperate data into 1000 equal groups
        # stepsize=(y.max()-y.min())/1000
        # for k in range(0,1001):   #show distribution of 1000 equal parts 1000+1 to show the y.max()
        #     bins.append(y.min()+stepsize*k)
        #
        # n,bins,patchs=axn.hist(y,bins,histtype='bar',rwidth=0.8,label=str(datasetFeatureNames[j]))
        # axn.set_xlabel( 'feature value range: (%0.4f - %0.4f)' %(y.min(),y.max()))
        # axn.set_ylabel( 'number')
        # #axn.set_xlim(y.min(),y.max())
        # #axn.set_ylim()
        # axn.set_title('histograms of feature id# '+ str(j)+",name="+str(datasetFeatureNames[j]))
        # axn.legend()

        #histogram of the data in 1000 pieces ??? this histogram is weird , display id#1 seems to be wrong and misleading
        # change num_bins to 500 to solve the problem.
        num_bins=500
        n,bins,patches = axf.hist(y,num_bins,normed=0,label=str(datasetFeatureNames[j]))
        axf.legend()

        #add a  'best fit' line
        mu=np.mean(y)
        sigma=np.std(y)   #standard deviation
        ybestfit= mlab.normpdf(bins,mu,sigma)
        axf.plot(bins,ybestfit,'--')
        axf.set_xlabel("feature value range:[%0.4f-%0.4f],num_bins=%d" % (y.min(), y.max(),num_bins))
        axf.set_ylabel("count")
        axf.set_title('histograms of feature id# ' + str(j) + ",name=" + str(datasetFeatureNames[j])+ r',$\mu=%.2f$,$\sigma=%.4f$' %(mu, sigma))
        #
        #tweak spacing to prevent clipping of ylabel
        #axf.tight_layout()


        axs = featureFig.add_subplot(2, 1, 2)  # show the scatter subplot for min,max value of all the features in a whole.
        axs.clear()

        # at first, convert [ (min1,max1), (min2,max2), ....(minn,maxn)] to [(min1,min2,...minn),(max1,max,...maxn)]
        ycoord = zip(*yscatter)
        for i,colors,names,markers in zip([0,1],['red','blue'],['min','max'],['x','o']):
            axs.scatter(xscatter, list(ycoord[i]), label=names, color=colors, s=25, marker=markers)



        axs.set_xlabel('feature id#')
        axs.set_ylabel('actual value ')
        axs.set_title("scatter min & max for "+ str(n_feature) +" features in "+ desc + " with " +str(n_sample) +" samples")

        if scatterAdjust==True:
            # adjust Y scale to show this column's min and max scatter point in the graph,at the cost of possibly sacrifice other columns
            if y.max()>0 :
                ymax=y.max()*1.2
            else:
                ymax=y.max()*0.8
            if y.min()>0 :
                ymin=y.min()*0.8
            else:
                ymin=y.min()*1.2
            axs.set_ylim(ymin,ymax)

        axs.annotate('Here:id =' + str(j)+' min:%s' %yscatter[j][0], xy=(j, yscatter[j][0]), xycoords='data',
                     xytext=(20, 30), textcoords='offset points',
                     arrowprops=dict(facecolor='black', arrowstyle='->'),
                     horizontalalignment='right', verticalalignment='bottom')
        axs.annotate('Here:id =' + str(j) + ' max:%s'%yscatter[j][1], xy=(j, yscatter[j][1]), xycoords='data',
                     xytext=(0.5, 0.85), textcoords='axes fraction',
                     arrowprops=dict(color='blue', arrowstyle='->'),
                     horizontalalignment='right', verticalalignment='bottom')
        axs.legend()

        #plt.show()
        if savePlotToDisk:
            plt.savefig(desc+"feature"+str(j)+"_"+str(datasetFeatureNames[j])+".png", figsize=(8, 6))
        plt.close(featureFig)  #close figures explicitly to release memory

        # key_resp= raw_input("please press any key with a Enter to plot next feature, type 'exit' to quit plotting")
        # if key_resp=='exit':
        #     print('Exit plotting features...')
        #     break

def plotAmountOverTransaction(x,amount,title):
    '''

    :param amount: overall capital at this point
    :param x: transaction numbers
    :return:
    '''


    amountFig = plt.figure(title , figsize=(10,8))
    #amountFig.subplots_adjust(top=0.92, left=0.10, right=0.97, hspace=0.37, wspace=0.3)

    # amf = amountFig.add_subplot(1, 1, 1)  # plot

    plt.plot(x,amount)
    plt.xlabel("transaction times")
    plt.ylabel("Investment value compared with initial=1")
    plt.show()


#customized data preprocess function
# for each feature, normalizedvalue= (xi- expectvalue(x))/(sigma of x +0.001)
def myPreprocess(batch):
    #for i in range(len(batch)):
#       batch[i] -= np.mean(batch[i], axis=0) #samplewise_zero_center

    mean=np.mean(batch,axis=0)
    stdX=np.std(batch,axis=0)+0.001  # add a small number to prevent dividing 0
    batch = (batch-mean)/stdX
    # batch=(batch-mean)/((np.max(batch,axis=0)-np.min(batch,axis=0))+0.001)

    return batch
