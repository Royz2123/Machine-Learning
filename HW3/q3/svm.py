import math
import random
import sys
import operator
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt

CLASSES = 10
FOLDS = 5

def read_samples(filename):
    samples = []
    count = 0
    with open(filename, 'r') as f:
        buf = ''
        while True:
            buf = f.readline()
            if not buf:
                break
            buf = buf.replace('\"', '')
            samples.append([float(v) for v in buf[0:-1].split(',')])
    return samples

def extract_label(set):
    labels = []
    trainSet = []
    for i in range(len(set)):
        labels.append(set[i][0])
        trainSet.append(set[i][1:])
    return trainSet,labels

#divide into k fold crossVal
def k_fold(set,k,c=1,g=-1):
    random.shuffle(set)
    errors = []
    test = []
    size = len(set)/k
    for i in range(k):
        train = set[i*size:(i+1)*size]
        if(i>0):
            test = set[:i*size]
        if(i<k):
            test = test + set[(i+1)*size:]

        errors.append(SVM_Error(train, test, c, g))

        train = []
        test = []
    return errors


def SVM_Error(trainSet, testSet, c, g):
    return oneVsAll(trainSet,testSet,c,g)

#train in One Against All
def oneVsAll(trainSet,testSet,c,g):
    fits = []
    trainSet, trainLabels = extract_label(trainSet)
    testSet, testLabels = extract_label(testSet)
    testLabels = np.asarray(testLabels).ravel()
    resultSet = []
    for i in range(CLASSES):
        labels = oneVsAll_labeling(trainLabels,i+1)
        labels = np.asarray(labels).ravel()

        if g != -1 :
            clf = svm.SVC(C=c, gamma=g, kernel='rbf')
        else:
            clf = svm.SVC(C=c, kernel='linear')
        if len(set(labels)) != 1:
            #do something ..
            clf.fit(np.asarray(trainSet),labels)
            fits.append(clf)

    for i in range(len(testSet)):
        yi = [testSet[i]]
        max = None
        maxj = -1
        for j in range(len(fits)):
            if(fits[j].decision_function(yi)>max or max==None):
                max = fits[j].decision_function(yi)
                maxj = j
        resultSet.append(maxj+1)
    return ErrorCompare(resultSet,testLabels)

#label one class as 1 and all the others as 0
def oneVsAll_labeling(set, rightClass):
    labels = []
    trainSet = []
    for i in range(len(set)):
        if set[i] == rightClass :
            labels.append([1])
        else :
            labels.append([0])
    return labels

#check error rate of one fold
def ErrorCompare(rs, tl):
    sum = 0.
    for i in range(len(rs)):
        if rs[i] == tl[i]:
            sum+=1
    return sum/len(rs)


#train linear SVM
def trainLinearSVM (X,K,allC):
    Cerrors = []
    for c in allC:
        errors = k_fold(X,K,c)
        #calc mean error of kCrossFold
        Cerrors.append(sum(errors) / float(len(errors)))
    printErrors (Cerrors,allC)

#train RBFS SVM
def trainRBFSVM (X,K,allC, allG):
    CGerrors = []
    for c in allC:
        for g in allG:
            errors = k_fold(X,K,c,g)
            CGerrors.append(sum(errors) / float(len(errors)))
    printErrors (CGerrors,allC,allG)



def printErrors(errList,clist,glist=[]):
    x=0
    for i in clist:
        if len(glist)>0:
            for j in glist:
                print ("c=",i,"g=", j, " error =", errList[x])
                x+=1
        else:
            print ("c=",i, " error =", errList[x])
            x+=1

def main():
    path = sys.argv[1] # here "csv/train.csv"
    train = read_samples(path)
    trainLinearSVM(train,FOLDS,[1, 5])
    trainRBFSVM(train,FOLDS,[1, 5],[0.005, 0.01, 0.05])


if __name__ == "__main__":
	main()
