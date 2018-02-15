import math
import random
import sys
import operator
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt

CLASSES=10

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

def k_fold(set,k,c=1,g=-1):
    random.shuffle(set)
    errors = []
    #print set
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

def oneVsAll(trainSet,testSet,c,g):
    fits = []
    trainSet, trainLabels = extract_label(trainSet)
    testSet, testLabels = extract_label(testSet)
    testLabels = np.asarray(testLabels).ravel()
    resultSet = []
    for i in range(CLASSES):
        #print CLASSES-i
        #print trainLabels
        labels = oneVsAll_labeling(trainLabels,i+1)
        labels = np.asarray(labels).ravel()
        #print labels

        if g != -1 :
            clf = svm.SVC(C=c, gamma=g, kernel='rbf')
            print g,c
        else:
            clf = svm.SVC(C=c, kernel='linear')
            print -1,c
        if len(set(labels)) != 1:
            #do something ..
            #return 0.5
            clf.fit(np.asarray(trainSet),labels)
            fits.append(clf)

    for i in range(len(testSet)):
        yi = [testSet[i]]
        max = -109
        maxj = -1
        for j in range(len(fits)):
            #print fits[j].decision_function(yi)
            if(fits[j].decision_function(yi)>max or max==-109):
                max = fits[j].decision_function(yi)
                maxj = j
        resultSet.append(maxj+1)
        #print "hey", maxj
    return ErrorCompare(resultSet,testLabels)


def oneVsAll_labeling(set, rightClass):
    labels = []
    trainSet = []
    for i in range(len(set)):
        if set[i] == rightClass :
            labels.append([1])
        else :
            labels.append([0])
    return labels


def ErrorCompare(rs, tl):
    print rs
    print tl
    sum = 0.
    for i in range(len(rs)):
        if rs[i] == tl[i]:
            sum+=1
    return sum/len(rs)


#train linear SVM

def trainLinearSVM (X,K,allC):
    Cerrors = []
    for c in allC:
        #for g in allG:
        errors = k_fold(X,K,c)
        Cerrors.append(sum(errors) / float(len(errors)))
    print Cerrors


def trainRBFSVM (X,K,allC, allG):
    CGerrors = []
    for c in allC:
        for g in allG:
            #for g in allG:
            errors = k_fold(X,K,c,g)
            CGerrors.append(sum(errors) / float(len(errors)))
    print CGerrors

def main():
    train = read_samples("csv/train.csv")
    trainLinearSVM(train,5,[1, 5])
    print ""
    trainRBFSVM(train,5,[1, 5],[0.005, 0.01, 0.05])


if __name__ == "__main__":
	main()






##      graveyard




def divide_samples(set, rightClass):
    pos = []
    neg = []
    for i in range(len(set)):
        if set[i][0] == rightClass :
            pos.append(set[i][1:])
        else :
            neg.append(set[i][1:])
    return pos,neg


def oneVsAll_labeling(set, rightClass):
    labels = []
    trainSet = []
    for i in range(len(set)):
        if set[i][0] == rightClass :
            labels.append([0])
        else :
            labels.append([1])
        trainSet.append(set[i][1:])
    return trainSet,labels



def trainAndTest(trainSet, testSet, num, c, g):
    trainSet,labels = oneVsAll_labeling(trainSet)
    resultSet = []
    if g != -1 :
        clf = svm.SVC(C=c, gemma=g, kernel='rbf')
    else:
        clf = svm.SVC(C=c, kernel='linear')

    labels = np.asarray(labels).ravel()

    if len(set(labels)) == 1:
        clf.fit(np.asarray(trainSet), np.asarray(labels))
        testSet,testLabels = oneVsAll_labeling(testSet, 2)
        resultSet = clf.predict(testSet)
        testLabels = np.asarray(testLabels)
        testLabels = testLabels.ravel()
        print(resultSet)
        print(testLabels)
        return resultSet,testLabels
    else:
        print("all the same type ..")
    return [],[]
