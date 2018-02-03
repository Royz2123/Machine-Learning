import math
import random
import sys
import operator


def read_samples(filename, amount):
    amount /= 4
    samples = []
    count = 0;
    with open(filename, 'r') as f:
        buf = ''
        while True:
            buf = f.readline()
            if not buf:
                break
            buf= buf.replace('\"','')
            samples.append([int(v) for v in buf[:-1].split(',')])
            samples[count] = [chr(int(count/amount)+65)]+samples[count]
            #print(samples[count])
            count+=1;
    return samples


def euclideanDistance(pt1, pt2):
	distance = 0
	for i in range(1,len(pt1)):
		distance += pow((pt1[i] - pt2[i]), 2)
	return math.sqrt(distance)


def getKNearesNeighbors(trainingSet, testPt, k):
	distances = []
	length = len(testPt)-1
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testPt, trainingSet[x])
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors


def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][0]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]


def chooseK (trainSet, validSet, maxInt):
    maxK = -1
    max = 0
    for k in range(1,maxInt+1):
        result = runKnn(trainSet,validSet,k)
        if getAccuracy(validSet,result) >= max:
            max = getAccuracy(validSet,result)
            maxK = k
        print("iteration num",k,(maxK == k))
    return maxK


def runKnn (trainSet, testSet ,k):
    TestResult = []
    for x in testSet:
        neighbors = getKNearesNeighbors(trainSet, x, k)
        prediction = getResponse(neighbors)
        TestResult.append(prediction)
    return TestResult


def getAccuracy (realSet, resultSet):
    right = 0.0
    for x in range(len(realSet)):
        if resultSet[x] == realSet[x][0]:
            right+=1.0
    return right/len(realSet)




def main():
    train = read_samples("csv/train_data.csv", 2000)
    valid = read_samples("csv/valid_data.csv", 400)
    k = 5 #chooseK(train,valid,50)      - we found that k = 5 is optimal
    test = read_samples("csv/test_data.csv", 400)
    testResult = runKnn(train,test,k)
    print("Classification rate:\t%f" % getAccuracy(test,testResult))




if __name__ == "__main__":
	main()
