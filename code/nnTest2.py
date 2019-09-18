import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))) #상위 폴더 import
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))) # 상위 상위 폴더 import
import basicFunction as bf
import numpy as np
import keras as kr
import tensorflow as tf
import time
from numpy import array
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.model_selection import KFold

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # imac setting.

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_1x5(x):
	return tf.nn.max_pool(x, ksize=[1, 1, 5, 1], strides=[1, 1, 5, 1], padding='SAME')

def max_pool_2x1(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME')

def max_pool_1x2(x):
	return tf.nn.max_pool(x, ksize=[1, 1, 2, 1], strides=[1, 1, 2, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def aver_pool(x):
	return tf.nn.avg_pool(x, ksize=[1, 1, 3, 1], strides=[1, 1, 3, 1], padding='SAME')

if __name__ == "__main__":
	cTime = time.localtime()
	KFOLD = 5
	logDir = "../log/paperModel_CNN_%04d%02d%02d_" % (cTime.tm_year, cTime.tm_mon, cTime.tm_mday) + str(KFOLD) + "fold/"
	logFile = "%02d:%02d:%02d" % (cTime.tm_hour, cTime.tm_min, cTime.tm_sec) + ".log"
	if not os.path.isdir(logDir):
		os.mkdir(logDir)

	logPath = logDir + logFile
	BATCHSIZE = 50
	filePath = '../data/20190816/Dual/win15000/'
	patternName = ['1','2','3','4','5','6','7','8','9','a']

	numFreq = 6
	numAudioData = 569
	numTotalAud = numFreq * numAudioData
	numTotalAcc = 300
	windowSize = 15000
	numLabel = len(patternName)
	bf.allNumber = 0

	data = []
	#audioData = []
	for fileIndex in range(1,51):
		for patternIndex in range(10):
			fileName = patternName[patternIndex]+"/"+patternName[patternIndex]+"_"+str(fileIndex)+".csv"
			if (patternName[patternIndex] == 'a'):
				label = '0'
			else:
				label = patternName[patternIndex]
			tmp = bf.onlyFileRead(filePath, fileName, label)
			data.append(tmp)
	np.random.shuffle(data)

	result_accuracy = np.zeros(shape=(KFOLD))
	result_precision = np.zeros(shape=(KFOLD))
	result_recall = np.zeros(shape=(KFOLD))
	result_f1Score = np.zeros(shape=(KFOLD))

	result_accuracy2 = np.zeros(shape=(KFOLD))
	result_precision2 = np.zeros(shape=(KFOLD))
	result_recall2 = np.zeros(shape=(KFOLD))
	result_f1Score2 = np.zeros(shape=(KFOLD))

	count = 0

	#print('data : ', np.shape(data))
	#print('data : ', data[100])

	kf = KFold(n_splits = KFOLD)
	kf.get_n_splits(data)
	data = np.array(data)
	#kf.get_n_splits(audioData)
	#audioData = np.array(audioData)

	for train_index, test_index in kf.split(data):
		dataTrain, dataTest = data[train_index], data[test_index]

		dTrain = bf.onlySampleSize(dataTrain, 1)
		dTest = bf.onlySampleSize(dataTest, 1)

		acc_xTrain = []
		acc_yTrain = []
		aud_xTrain = []
		aud_yTrain = []
		xTrain = []
		yTrain = []
		for d in dTrain:
			acc_xTrain.append(d[0 : numTotalAcc])
			aud_xTrain.append(d[numTotalAcc : numTotalAcc + numTotalAud])
			xTrain.append(d[0 : numTotalAcc + numTotalAud])
			acc_yTrain.append(bf.oneHotLabel(int(d[-1]), numLabel))
			aud_yTrain.append(bf.oneHotLabel(int(d[-1]), numLabel))
			yTrain.append(bf.oneHotLabel(int(d[-1]), numLabel))
		print('acc_xTrain : ', len(acc_xTrain))
		print('aud_xTrain : ', len(aud_xTrain))

		acc_xTest = []
		acc_yTest = []
		aud_xTest = []
		aud_yTest = []
		for d in dTest:
			acc_xTest.append(d[0 : numTotalAcc])
			aud_xTest.append(d[numTotalAcc : numTotalAcc + numTotalAud])
			acc_yTest.append(bf.oneHotLabel(int(d[-1]), numLabel))
			aud_yTest.append(bf.oneHotLabel(int(d[-1]), numLabel))

		acc_xTest = array(acc_xTest).reshape(len(acc_xTest), numTotalAcc)
		aud_xTest = array(aud_xTest).reshape(len(aud_xTest), numTotalAud)

	#       on Imac the GPU is not working. so
		with tf.device('/gpu:3'):
			acc_inputX = tf.placeholder(tf.float32, [None, numTotalAcc])
			acc_outputY = tf.placeholder(tf.float32, [None, numLabel])

			#1 * 100 * 3
			W_conv11 = weight_variable([3, 1, 1, 9]) #width, height, channel input, channel output
			b_conv11 = bias_variable([9])
			x_image = tf.reshape(acc_inputX, [-1, 100, 3, 1]) # -1, width, height, channel)
			h_conv11 = tf.nn.relu(conv2d(x_image, W_conv11) + b_conv11)
			h_pool11 = max_pool_2x1(h_conv11)
			#1 * 150 * 9
			W_conv12 = weight_variable([3, 1, 9, 18])
			b_conv12 = bias_variable([18])
			h_conv12 = tf.nn.relu(conv2d(h_pool11, W_conv12) + b_conv12)
			h_pool12 = max_pool_2x1(h_conv12)
			# 1* 75 * 18

			# h2 = tf.nn.dropout(h1_pool2,0.5)
			W_conv13 = weight_variable([3, 1, 18, 36])
			b_conv13 = bias_variable([36])
			h_conv13 = tf.nn.relu(conv2d(h_pool12, W_conv13) + b_conv13)
			h_pool13 = max_pool_2x1(h_conv13)

			# 1* 13 * 36
			W_conv14 = weight_variable([3, 1, 36, 72])
			b_conv14 = bias_variable([72])
			h_conv14 = tf.nn.relu(conv2d(h_pool13, W_conv14) + b_conv14)
			h_pool14 = max_pool_2x1(h_conv14)

			# 1* 7 * 72
			W_conv15 = weight_variable([3, 1, 72, 144])
			b_conv15 = bias_variable([144])
			h_conv15 = tf.nn.relu(conv2d(h_pool14, W_conv15) + b_conv15)
			h_pool15 = max_pool_2x1(h_conv15)

			# 1* 4 * 144
			# h5 = tf.nn.dropout(h1_pool5,0.5)
			
			W_conv16 = weight_variable([3, 1, 144, 288])
			b_conv16 = bias_variable([288])
			h_conv16 = tf.nn.relu(conv2d(h_pool15, W_conv16) + b_conv16)
			h_pool16 = max_pool_2x1(h_conv16)

			# 1* 2 * 288
			W_fc11 = weight_variable([2*3*288, 576])
			b_fc11 = bias_variable([576])
			h_pool12_flat = tf.reshape(h_pool16, [-1, 2 * 3 * 288])
			h_fc11 = tf.nn.relu(tf.matmul(h_pool12_flat, W_fc11) + b_fc11)

			keep_prob = tf.placeholder(tf.float32)
			h_fc11_drop1 = tf.nn.dropout(h_fc11, keep_prob)

			W_fc12 = weight_variable([576, 144])
			b_fc12 = bias_variable([144])
			h_pool13_flat = tf.reshape(h_fc11_drop1, [-1, 576])
			h_fc12 = tf.nn.relu(tf.matmul(h_pool13_flat, W_fc12) + b_fc12)

			# drop out 연산의 결과를 담을 변수
			h_fc11_drop2 = tf.nn.dropout(h_fc12, keep_prob)

			W_fc13 = weight_variable([144, numLabel])
			b_fc13 = bias_variable([numLabel])
			y_conv1 = tf.nn.softmax(tf.matmul(h_fc11_drop2, W_fc13) + b_fc13)

			print('y_conv1 : ', y_conv1)
			print('y_conv1 : ', y_conv1.shape)
			print('y_conv1 : ', y_conv1[1])
			# Define loss and optimizer

			cross_entropy1 = -tf.reduce_sum(acc_outputY * tf.log(tf.clip_by_value(y_conv1, 1e-10, 1.0)))
			train_step1 = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy1)
			correct_prediction1 = tf.equal(tf.argmax(y_conv1, 1), tf.argmax(acc_outputY, 1))
			accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, tf.float32))


		#Audio model
			aud_inputX = tf.placeholder(tf.float32, [None, numTotalAud])
			aud_outputY = tf.placeholder(tf.float32, [None, numLabel])

			W_conv21 = weight_variable([1, 3, 1, 4])
			#bias = 출력 갯수= kernel 수= filter 수
			b_conv21 = bias_variable([4])
			x_image = tf.reshape(aud_inputX, [-1, numFreq, numAudioData, 1])
			h_conv21 = tf.nn.relu(conv2d(x_image, W_conv21) + b_conv21)
			h_pool21 = max_pool_1x5(h_conv21)
		
			W_conv22 = weight_variable([1, 3, 4, 8])
			b_conv22 = bias_variable([8])
			h_conv22 = tf.nn.relu(conv2d(h_pool21, W_conv22) + b_conv22)
			h_pool22 = max_pool_1x2(h_conv22)

			W_conv23 = weight_variable([1, 3, 8, 16])
			b_conv23 = bias_variable([16])
			h_conv23 = tf.nn.relu(conv2d(h_pool22, W_conv23) + b_conv23)
			h_pool23 = max_pool_1x2(h_conv23)

			W_conv24 = weight_variable([1, 3, 16, 32])
			b_conv24 = bias_variable([32])
			h_conv24 = tf.nn.relu(conv2d(h_pool23, W_conv24) + b_conv24)
			h_pool24 = max_pool_1x2(h_conv24)

			W_conv25 = weight_variable([1, 3, 32, 64])
			b_conv25 = bias_variable([64])
			h_conv25 = tf.nn.relu(conv2d(h_pool24, W_conv25) + b_conv25)
			h_pool25 = max_pool_1x2(h_conv25)
			
			W_conv26 = weight_variable([1, 3, 64, 128])
			b_conv26 = bias_variable([128])
			h_conv26 = tf.nn.relu(conv2d(h_pool25, W_conv26) + b_conv26)
			h_pool26 = max_pool_2x2(h_conv26)
		
			W_conv27 = weight_variable([1, 3, 128, 256])
			b_conv27 = bias_variable([256])
			h_conv27 = tf.nn.relu(conv2d(h_pool26, W_conv27) + b_conv27)
			h_pool27 = max_pool_2x2(h_conv27)

			W_conv28 = weight_variable([1, 3, 256, 512])
			b_conv28 = bias_variable([512])
			h_conv28 = tf.nn.relu(conv2d(h_pool27, W_conv28) + b_conv28)
			h_pool28 = max_pool_2x2(h_conv28)

			h_pool_aver = aver_pool(h_pool28)
			h_pool22_flat = tf.reshape(h_pool_aver, [-1, 1* 1 * 512])
			h_drop2 = tf.nn.dropout(h_pool22_flat, keep_prob)

			W_fc23 = weight_variable([512, numLabel])
			b_fc23 = bias_variable([numLabel])
			y_conv2 = tf.nn.softmax(tf.matmul(h_drop2, W_fc23) + b_fc23)	

			#y_conv3 = kr.layers.Add()([y_conv1, y_conv2])
			#y_conv3 = tf.maximum(y_conv1, y_conv2)
			y_conv3 = kr.layers.Concatenate()([h_pool12_flat, h_pool22_flat])
			W_conv4 = weight_variable([2240, 10])	
			b_conv4 = bias_variable([10])
			#y_conv4 = tf.nn.relu(conv2d(y_conv3, W_conv4)) + b_conv4)
			#y_conv4 = tf.nn.softmax(tf.matmul(y_conv3, W_conv4) + b_conv4)

			W_fc31 = weight_variable([2240, 560])
			b_fc31 = bias_variable([560])
			h_fc31 = tf.nn.relu(tf.matmul(y_conv3, W_fc31) + b_fc31)
			
			W_fc32 = weight_variable([560, 140])
			b_fc32 = bias_variable([140])
			h_fc32 = tf.nn.relu(tf.matmul(h_fc31, W_fc32) + b_fc32)

			W_fc33 = weight_variable([140, 10])
			b_fc33 = bias_variable([10])
			y_conv4 = tf.nn.softmax(tf.matmul(h_fc32, W_fc33) + b_fc33)

#			y_conv3 = tf.matmul(h_pool13_flat, h_pool22_flat)
#			y_conv3 = tf.reshape(y_conv3, [-1,301056])
#			W_conv4 = weight_variable([301056, 10])
#			b_conv4 = bias_variable([10])
#			y_conv4 = tf.nn.softmax(tf.matmul(y_conv3, W_conv4) + b_conv4)

			print('y_conv2 : ', y_conv2[1])
			print('y_conv2 : ', y_conv2.shape)
			print('y_conv3 : ', y_conv3)
			print('y_conv3 : ', y_conv3.shape)

			cross_entropy2 = -tf.reduce_sum(aud_outputY * tf.log(tf.clip_by_value(y_conv2, 1e-10, 1.0)))
			train_step2 = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy2)
			correct_prediction2 = tf.equal(tf.argmax(y_conv2, 1), tf.argmax(aud_outputY, 1))
			accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2, tf.float32))

			cross_entropy3 = -tf.reduce_sum(aud_outputY * tf.log(tf.clip_by_value(y_conv4, 1e-10, 1.0)))
			train_step3 = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy3)
			correct_prediction3 = tf.equal(tf.argmax(y_conv4, 1), tf.argmax(aud_outputY, 1))
			accuracy3 = tf.reduce_mean(tf.cast(correct_prediction3, tf.float32))

		#saver = tf.train.Saver()
		sess = tf.InteractiveSession()
		sess.run(tf.global_variables_initializer())
		acc_xTrain = array(acc_xTrain).reshape(len(acc_xTrain), numTotalAcc)
		acc_yTrain = array(acc_yTrain).reshape(len(acc_yTrain), numLabel)

		aud_xTrain = array(aud_xTrain).reshape(len(aud_xTrain), numTotalAud)
		aud_yTrain = array(aud_yTrain).reshape(len(aud_yTrain), numLabel)

		xTrain = array(xTrain).reshape(len(xTrain), numTotalAcc + numTotalAud)
		yTrain = array(yTrain).reshape(len(yTrain), numLabel)

		bf.mLog("training Start", logPath)
		cTime = time.localtime()
		f = open("%02d%02d-%02d:%02d.txt" % (cTime.tm_mon, cTime.tm_mday, cTime.tm_hour, cTime.tm_min), 'a')
		for j in range(20001):
			batch_X, batch_Y = bf.getBatchData(BATCHSIZE, xTrain, yTrain)
			batch_XA = batch_X[:,0:numTotalAcc]
			batch_XB = batch_X[:,numTotalAcc : numTotalAcc + numTotalAud]
			train_step3.run(session=sess, feed_dict={acc_inputX: batch_XA, acc_outputY: batch_Y, aud_inputX:batch_XB, aud_outputY:batch_Y, keep_prob:0.5})

			if j % BATCHSIZE == 0:
				train_accuracy3 = accuracy3.eval(session=sess, feed_dict={acc_inputX: batch_XA, acc_outputY: batch_Y, aud_inputX: batch_XB, aud_outputY: batch_Y, keep_prob:1.0})	
				bf.mLog("step %d, accuracy %g" % (j, train_accuracy3), logPath)
				yPreTmp = tf.argmax(y_conv4, 1)
				test_accuracy = accuracy3.eval(feed_dict={acc_inputX: acc_xTest, acc_outputY: acc_yTest, aud_inputX: aud_xTest, aud_outputY: aud_yTest, keep_prob: 1.0})
				bf.mLog("test accuracy %g" % test_accuracy, logPath)
				f.write(str(test_accuracy)+'\n')
				#bf.mLog("AUD step %d, All accuracy %g" % (j, train_accuracy3), logPath)
				#h1 = sess.run(y_conv1, feed_dict={acc_inputX: batch_XA, acc_outputY: batch_Y, aud_inputX: batch_XB, aud_outputY: batch_Y, keep_prob:1.0}) 
				#h2 = sess.run(y_conv2, feed_dict={acc_inputX: batch_XA, acc_outputY: batch_Y, aud_inputX: batch_XB, aud_outputY: batch_Y, keep_prob:1.0}) 
				#h3 = sess.run(y_conv3, feed_dict={acc_inputX: batch_XA, acc_outputY: batch_Y, aud_inputX: batch_XB, aud_outputY: batch_Y, keep_prob:1.0}) 
				#print('h1 : ',h1)
				#print('h2 : ',h2)
				#print('h3 : ',h3)
		f.close()
		bf.mLog("training Finish", logPath)

		bf.mLog("test Start", logPath)
		
		yPreTmp = tf.argmax(y_conv4, 1)
		val_acc, yPred = sess.run([accuracy3, yPreTmp], feed_dict={acc_inputX: acc_xTest, acc_outputY: acc_yTest, aud_inputX: aud_xTest, aud_outputY: aud_yTest, keep_prob: 1.0})
		yTrue = np.argmax(acc_yTest, 1)
		bf.mLog("test finish", logPath)

		result_accuracy[count] = accuracy_score(yTrue, yPred)
		result_precision[count] = precision_score(yTrue, yPred, average = 'macro')
		result_recall[count] = recall_score(yTrue, yPred, average = 'macro')
		result_f1Score[count] = f1_score(yTrue, yPred, average = 'macro')
		result_confusion = str(confusion_matrix(yTrue, yPred))
		bf.mLog("%d-fold %dth" % (KFOLD, count+1), logPath)
		bf.mLog("accuracy : " + str(result_accuracy[count]), logPath)
		bf.mLog("precision : " + str(result_precision[count]), logPath)
		bf.mLog("recall : " + str(result_recall[count]), logPath)
		bf.mLog("f1 Score : " + str(result_f1Score[count]), logPath)
		bf.mLog("confusion matrix\n" + result_confusion, logPath)
		count = count + 1

		sess.close()

	modelAccuracy = 0
	modelPrecision = 0
	modelRecall = 0
	modelF1Score = 0

	for j in range(0, count):
		modelAccuracy = modelAccuracy + result_accuracy[j]
		modelPrecision = modelPrecision + result_precision[j]
		modelRecall = modelRecall + result_recall[j]
		modelF1Score = modelF1Score + result_f1Score[j]

	modelAccuracy = modelAccuracy / KFOLD
	modelPrecision = modelPrecision /KFOLD
	modelRecall = modelRecall / KFOLD
	modelF1Score = modelF1Score / KFOLD

	bf.mLog("total Accuracy : " + str(modelAccuracy), logPath)
	bf.mLog("total Precision : " + str(modelPrecision), logPath)
	bf.mLog("total Recall : " + str(modelRecall), logPath)
	bf.mLog("total f1 score : " + str(modelF1Score), logPath)
