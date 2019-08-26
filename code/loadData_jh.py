import csv
import numpy as np
import random
import itertools

allNum = 0
batchNum = 0
xData = []
yData = []
training_set_X = []
training_set_Y = []
test_set_X = []
test_set_Y = []

def load_data_jh(dirName,Kind):
	global xData;
	global yData;
	tmp_X = []
	tmp_Y = []
	for i in range(len(Kind)):
		InputFileName = dirName + Kind[i]+".csv"
		f=open(InputFileName,'r')
		csvReader = csv.reader(f)
		for train_setX in csvReader:
			tmp_X.append(train_setX)
			tmp_Y.append(i)

		c = list(zip(tmp_X, tmp_Y))
		random.shuffle(c)
		tmp_X, tmp_Y = zip(*c)
		tmp_X = list(tmp_X)
		tmp_Y = list(tmp_Y)

		xData = xData + tmp_X[:]
		yData = yData + tmp_Y[:]
		
		tmp_X = []
		tmp_Y = []
		f.close()

	return xData, yData

def load_data(dirName,Kind,training_rate):
	global training_set_X
	global training_set_Y
	global test_set_X
	global test_set_Y
	tmp_X = []
	tmp_Y = []
	for i in range(len(Kind)):
		InputFileName = dirName + Kind[i]+".csv"
		f=open(InputFileName,'r')
		csvReader = csv.reader(f)
		for train_setX in csvReader:
			tmp_X.append(train_setX)
			tmp_Y.append(i)

		c = list(zip(tmp_X, tmp_Y))
		random.shuffle(c)
		tmp_X, tmp_Y = zip(*c)
		tmp_X = list(tmp_X)
		tmp_Y = list(tmp_Y)

		splitSize = (int)(len(tmp_X) * training_rate)

		training_set_X = training_set_X + tmp_X[0:splitSize]
		test_set_X = test_set_X + tmp_X[splitSize:len(tmp_X)]
		training_set_Y = training_set_Y + tmp_Y[0:splitSize]
		test_set_Y = test_set_Y + tmp_Y[splitSize:len(tmp_Y)]
		tmp_X = []
		tmp_Y = []
		f.close()

	return training_set_X, training_set_Y, test_set_X, test_set_Y

def OneAndHot(batch_Y,Kind):
	targets = np.array(batch_Y).reshape(-1)
	one_hot_targets = np.eye(len(Kind))[targets]
	return one_hot_targets

def Kfold_jh(xData, yData, Kind, KFOLD):
	global allNum

	xTrain = []
	yTrain = []
	xTest = []
	yTest = []

	numOfTrain = len(xData)/len(Kind)
	numOfTrain = int(numOfTrain)

	if numOfTrain == allNum:
		allNum = 0
	for i in range(len(Kind)):
		number = allNum + i*numOfTrain
		number = int(number)
		xTest += xData[number:number+int(numOfTrain/KFOLD)]
		yTest += yData[number:number+int(numOfTrain/KFOLD)]
		if allNum == 0:
			xTrain += xData[number+int(numOfTrain/KFOLD):number+numOfTrain]
			yTrain += yData[number+int(numOfTrain/KFOLD):number+numOfTrain]
		else:
			xTrain += xData[i*numOfTrain:number]
			xTrain += xData[number+int(numOfTrain/KFOLD):numOfTrain*(i+1)]
			yTrain += yData[i*numOfTrain:number]
			yTrain += yData[number+int(numOfTrain/KFOLD):numOfTrain*(i+1)]
	
	allNum += int(numOfTrain/KFOLD)

	return xTrain, yTrain, xTest, yTest

def batchTrain_jh(xTrain, yTrain, Kind):
	global batchNum
	nTrain = len(xTrain)/len(Kind)
	batch_X = []
	batch_Y = []

	#print('batch size : ', nTrain)

	if nTrain == batchNum:
		batchNum = 0
	for i in range(len(Kind)):
		n = batchNum + i*nTrain
		n = int(n)
		batch_X += xTrain[n:n+10]
		batch_Y += yTrain[n:n+10]
	
	batchNum += 10
	
	return batch_X, batch_Y
