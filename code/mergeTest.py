import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))) #상위 폴더 import
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))) # 상위 상위 폴더 import
import basicFunction as bf
import numpy as np
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

def max_pool_2x1(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME')


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
    #filePath = '../data/20190816/Acceloremeter/'
    patternName = ['1','2','3','4','5','6','7','8','9','a']

    numLabel = len(patternName)
    bf.allNumber = 0
    data = []
    for fileIndex in range(1,51):
        for patternIndex in range(10):
            fileName = patternName[patternIndex]+"/"+patternName[patternIndex]+"_"+str(fileIndex)+".csv"
            if (patternName[patternIndex] == 'a'):
                label = '0'
            else:
                label = patternName[patternIndex]
            tmp = bf.onlyFileRead(filePath, fileName, label)
            data.append(tmp)
            #print('tmp : ', tmp)
            #print('data : ', data)

    np.random.shuffle(data)

    result_accuracy = np.zeros(shape=(KFOLD))
    result_precision = np.zeros(shape=(KFOLD))
    result_recall = np.zeros(shape=(KFOLD))
    result_f1Score = np.zeros(shape=(KFOLD))
    count = 0

    kf = KFold(n_splits = KFOLD)
    kf.get_n_splits(data)
    data = np.array(data)
    for train_index, test_index in kf.split(data):
        dataTrain, dataTest = data[train_index], data[test_index]

        dTrain = bf.onlySampleSize(dataTrain, 1)

        acc_xTrain = []
        acc_yTrain = []
        for d in dTrain:
            acc_xTrain.append(d[0:300])
            acc_yTrain.append(bf.oneHotLabel(int(d[300]), numLabel))

        #       on Imac the GPU is not working. so
        with tf.device('/gpu:3'):
            acc_inputX = tf.placeholder(tf.float32, [None, 300])
            acc_outputY = tf.placeholder(tf.float32, [None, 10])

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

            #  h2 = tf.nn.dropout(h1_pool2,0.5)

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
            #  h5 = tf.nn.dropout(h1_pool5,0.5)
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
            #  keep_prob = tf.placeholder(tf.float32)
            h_fc11_drop2 = tf.nn.dropout(h_fc12, keep_prob)

            W_fc13 = weight_variable([144, 10])
            b_fc13 = bias_variable([10])
            y_conv1 = tf.nn.softmax(tf.matmul(h_fc11_drop2, W_fc13) + b_fc13)

            # Define loss and optimizer
            cross_entropy = -tf.reduce_sum(acc_outputY * tf.log(tf.clip_by_value(y_conv1, 1e-10, 1.0)))
            train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
            correct_prediction = tf.equal(tf.argmax(y_conv1, 1), tf.argmax(acc_outputY, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#        saver = tf.train.Saver()
        sess = tf.InteractiveSession()

        sess.run(tf.global_variables_initializer())

        acc_xTrain = array(acc_xTrain).reshape(len(acc_xTrain), 300)
        acc_yTrain = array(acc_yTrain).reshape(len(acc_yTrain), 10)

        bf.mLog("training Start", logPath)
        
        for j in range(301):

            batch_X, batch_Y = bf.getBatchData(BATCHSIZE, acc_xTrain, acc_yTrain)
            train_step.run(feed_dict={acc_inputX: batch_X, acc_outputY: batch_Y, keep_prob:0.5})
            if j % BATCHSIZE == 0:
                train_accuracy = accuracy.eval(feed_dict={acc_inputX: batch_X, acc_outputY: batch_Y, keep_prob:1.0})
                bf.mLog("step %d, accuracy %g" % (j, train_accuracy), logPath)
        bf.mLog("training Finish", logPath)

        #savePath = saver.save(sess, "./model/model_" + str(KFOLD) +  ".ckpt")
        #bf.mLog("save path"+savePath, logPath)

        dTest = bf.onlySampleSize(dataTest, 1)

        acc_xTest = []
        acc_yTest = []
        for d in dTest:
            acc_xTest.append(d[0:300])
            acc_yTest.append(bf.oneHotLabel(int(d[300]), numLabel))

        acc_xTest = array(acc_xTest).reshape(len(acc_xTest), 300)
        bf.mLog("test Start", logPath)
        yPreTmp = tf.argmax(y_conv1, 1)
        val_acc, yPred = sess.run([accuracy, yPreTmp], feed_dict={acc_inputX: acc_xTest, acc_outputY: acc_yTest, keep_prob: 1.0})
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
        bf.mLog("f1 Score : " +str(result_f1Score[count]), logPath)
        bf.mLog("confution matrix" + result_confusion, logPath)
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
