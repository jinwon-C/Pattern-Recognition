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

def max_pool_2x2(x):
    # return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')
    return tf.nn.max_pool(x, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='SAME')


if __name__ == "__main__":
    cTime = time.localtime()
    KFOLD = 5
    logPath = "./paperModel_CNN_%04d%02d%02d_" % (cTime.tm_year, cTime.tm_mon, cTime.tm_mday) + str(KFOLD) + "fold.log"
    BATCHSIZE = 10
    filePath = '../data/'
    #patternName = ['1','2','3','4','5','6','7','8','9','a']
    patternName = ['1','2','3','4','5']

    numLabel = 5
    #for patternIndex in range(1):
    bf.allNumber = 0
    data = []
    #for fileIndex in range(1,501):
    for fileIndex in range(1,11):
        #    for fileIndex in range(1,2):
        for patternIndex in range(5):
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
    count = 0

    kf = KFold(n_splits = KFOLD)
    kf.get_n_splits(data)
    data = np.array(data)
    for train_index, test_index in kf.split(data):
        dataTrain, dataTest = data[train_index], data[test_index]

        dTrain = bf.onlySampleSize(dataTrain, 1)

        xTrain = []
        yTrain = []
        for d in dTrain:
            xTrain.append(d[0:300])
            yTrain.append(bf.oneHotLabel(int(d[300]), numLabel))

        #       on Imac the GPU is not working. so
        with tf.device('/gpu:2'):
            inputX = tf.placeholder(tf.float32, [None, 300])
            outputY = tf.placeholder(tf.float32, [None, 5])

            #1 * 100 * 3
            W_conv1 = weight_variable([3, 1, 1, 9]) #width, height, channel input, channel output
            b_conv1 = bias_variable([9])
            x_image = tf.reshape(inputX, [-1, 100, 3, 1]) # -1, width, height, channel)
            h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
            h_pool1 = max_pool_2x2(h_conv1)
            #1 * 150 * 9
            W_conv2 = weight_variable([3, 1, 9, 18])
            b_conv2 = bias_variable([18])
            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
            h_pool2 = max_pool_2x2(h_conv2)
            # 1* 75 * 18

            #  h2 = tf.nn.dropout(h1_pool2,0.5)

            W_conv3 = weight_variable([3, 1, 18, 36])
            b_conv3 = bias_variable([36])
            h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
            h_pool3 = max_pool_2x2(h_conv3)

            # 1* 13 * 36
            W_conv4 = weight_variable([3, 1, 36, 72])
            b_conv4 = bias_variable([72])
            h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
            h_pool4 = max_pool_2x2(h_conv4)

            # 1* 7 * 72
            W_conv5 = weight_variable([3, 1, 72, 144])
            b_conv5 = bias_variable([144])
            h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)
            h_pool5 = max_pool_2x2(h_conv5)

            # 1* 4 * 144
            #  h5 = tf.nn.dropout(h1_pool5,0.5)
            W_conv6 = weight_variable([3, 1, 144, 288])
            b_conv6 = bias_variable([288])
            h_conv6 = tf.nn.relu(conv2d(h_pool5, W_conv6) + b_conv6)
            h_pool6 = max_pool_2x2(h_conv6)

            # 1* 2 * 288
            W_fc1 = weight_variable([2*3*288, 576])
            b_fc1 = bias_variable([576])
            h_pool2_flat = tf.reshape(h_pool6, [-1, 2 * 3 * 288])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

            keep_prob = tf.placeholder(tf.float32)
            h_fc1_drop1 = tf.nn.dropout(h_fc1, keep_prob)

            W_fc2 = weight_variable([576, 144])
            b_fc2 = bias_variable([144])
            h_pool3_flat = tf.reshape(h_fc1_drop1, [-1, 576])
            h_fc2 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc2) + b_fc2)

            # drop out 연산의 결과를 담을 변수
            #  keep_prob = tf.placeholder(tf.float32)
            h_fc1_drop2 = tf.nn.dropout(h_fc2, keep_prob)

            W_fc3 = weight_variable([144, 5])
            b_fc3 = bias_variable([5])
            y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop2, W_fc3) + b_fc3)

            # Define loss and optimizer
            cross_entropy = -tf.reduce_sum(outputY * tf.log(tf.clip_by_value(y_conv, 1e-10, 1.0)))
            train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
            correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(outputY, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        saver = tf.train.Saver()
        sess = tf.InteractiveSession()

        sess.run(tf.global_variables_initializer())

        xTrain = array(xTrain).reshape(len(xTrain), 300)
        yTrain = array(yTrain).reshape(len(yTrain), 5)

        bf.mLog("training Start", logPath)
        for j in range(15001):
            batch_X, batch_Y = bf.getBatchData(BATCHSIZE, xTrain, yTrain)
            train_step.run(feed_dict={inputX: batch_X, outputY: batch_Y, keep_prob:0.5})
            if j % BATCHSIZE == 0:
                train_accuracy = accuracy.eval(feed_dict={inputX: batch_X, outputY: batch_Y, keep_prob:1.0})
                bf.mLog("step %d, accuracy %g" % (j, train_accuracy), logPath)
        bf.mLog("training Finish", logPath)

        #savePath = saver.save(sess, "./model/model_" + str(KFOLD) +  ".ckpt")
        #bf.mLog("save path"+savePath, logPath)

        dTest = bf.onlySampleSize(dataTest, 1)

        xTest = []
        yTest = []
        for d in dTest:
            xTest.append(d[0:300])
            yTest.append(bf.oneHotLabel(int(d[300]), numLabel))

        xTest = array(xTest).reshape(len(xTest), 300)
        bf.mLog("test Start", logPath)
        yPreTmp = tf.argmax(y_conv, 1)
        val_acc, yPred = sess.run([accuracy, yPreTmp], feed_dict={inputX: xTest, outputY: yTest, keep_prob: 1.0})
        yTrue = np.argmax(yTest, 1)
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
