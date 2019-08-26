import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import time
import numpy as np
import tensorflow as tf
import loadData_jh as ld

DirName = "/home/cjw/data/wintest/win5000/"
Kind = ["nothing","left-right","up-down","circle","block","right-left","down-up","triangle"]
xData, yData = ld.load_data_jh(DirName, Kind)

KFOLD = 5
num_data = 344
freq_slice = 20
window_size = 5000
num_gest = len(Kind)
date = time.strftime('%Y%m%d')

sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, num_data, freq_slice])
y_= tf.placeholder(tf.float32, [None, num_gest])
W = tf.Variable(tf.zeros([num_data, freq_slice, num_gest]))
b = tf.Variable(tf.zeros([num_gest]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 원하는 행렬 사이즈로 초기 값을 만들어서 리턴하는 메서드
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)
# 0.1 로 초기값 지정하여 원하는 사이즈로 리턴
def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_1x7(x):
	return tf.nn.max_pool(x, ksize=[1, 1, 7, 1],strides=[1, 1, 7, 1], padding='SAME')

def max_pool_1x5(x):
	return tf.nn.max_pool(x, ksize=[1, 1, 5, 1],strides=[1, 1, 5, 1], padding='SAME')

def max_pool_1x2(x):
	return tf.nn.max_pool(x, ksize=[1, 1, 2, 1],strides=[1, 1, 2, 1], padding='SAME')

def max_pool_1x4(x):
	return tf.nn.max_pool(x, ksize=[1, 1, 4, 1],strides=[1, 1, 4, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

def aver_pool(x):
	return tf.nn.avg_pool(x, ksize=[1, 3, 2, 1],strides=[1, 3, 2, 1], padding='SAME')

with tf.device('/gpu:2'):
	#커널 size = weight 갯수, 입력 갯수, 출력 갯수
	W_conv1 = weight_variable([1, 3, 1, 4])
	#bias = 출력 갯수= kernel 수= filter 수
	b_conv1 = bias_variable([4])
	x_image = tf.reshape(x, [-1,freq_slice, num_data,1])
	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	h_pool1 = max_pool_1x2(h_conv1)
									
	W_conv2 = weight_variable([1, 3, 4, 8])
	b_conv2 = bias_variable([8])
	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_1x2(h_conv2)

#h2 = tf.nn.dropout(h_pool2,0.5)

	W_conv3 = weight_variable([1, 3, 8, 16])
	b_conv3 = bias_variable([16])
	h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
	h_pool3 = max_pool_1x2(h_conv3)

	W_conv4 = weight_variable([1, 3, 16, 32])
	b_conv4 = bias_variable([32])
	h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
	h_pool4 = max_pool_1x2(h_conv4)

	W_conv5 = weight_variable([1, 3, 32, 64])
	b_conv5 = bias_variable([64])
	h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)
	h_pool5 = max_pool_1x2(h_conv5)

#h5 = tf.nn.dropout(h_pool5,0.5)
	
	W_conv6 = weight_variable([1, 3, 64, 128])
	b_conv6 = bias_variable([128])
	h_conv6 = tf.nn.relu(conv2d(h_pool5, W_conv6) + b_conv6)
	h_pool6 = max_pool_2x2(h_conv6)
	
	
	W_conv7 = weight_variable([1, 3, 128, 256])
	b_conv7 = bias_variable([256])
	h_conv7 = tf.nn.relu(conv2d(h_pool6, W_conv7) + b_conv7)
	h_pool7 = max_pool_2x2(h_conv7)

	W_conv8 = weight_variable([1, 3, 256, 512])
	b_conv8 = bias_variable([512])
	h_conv8 = tf.nn.relu(conv2d(h_pool7, W_conv8) + b_conv8)
	h_pool8 = max_pool_2x2(h_conv8)


	h_pool_aver = aver_pool(h_pool8)
	h_pool2_flat = tf.reshape(h_pool_aver, [-1, 1* 1 * 512])

# drop out 연산의 결과를 담을 변수
	keep_prob = tf.placeholder(tf.float32)
	#h_fc1_drop2 = tf.nn.dropout(h_pool2_flat, keep_prob)
									
	W_fc3 = weight_variable([512, num_gest])
	b_fc3 = bias_variable([num_gest])
	y_conv=tf.nn.softmax(tf.matmul(h_pool2_flat, W_fc3) + b_fc3)
									
# Define loss and optimizer
	cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv,1e-10,1.0)))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
									
saver = tf.train.Saver()

for j in range(0, KFOLD):
	sess.run(tf.initialize_all_variables())
	xTest=[]
	yTest=[]
	xTrain=[]
	yTrain=[]
	xTrain, yTrain, xTest, yTest = ld.Kfold_jh(xData, yData,Kind, KFOLD)
	for i in range(16000):
		batch_X,batch_Y = ld.batchTrain_jh(xTrain, yTrain, Kind)
		batch_X = np.asarray(batch_X)
		batch_Y = np.asarray(batch_Y)
		batch_X = batch_X.reshape(80,num_data, freq_slice)
		batch_Y = ld.OneAndHot(batch_Y,Kind)
		if i%100 ==0:
			train_accuracy = accuracy.eval(feed_dict={x:batch_X, y_: batch_Y, keep_prob: 1.0})
			print("step %d, training accuracy %g"%(i, train_accuracy))
		
		train_step.run(feed_dict={x: batch_X, y_: batch_Y, keep_prob: 0.5})

	print("N-Fold Data :",j)
	result_dir = '/home/cjw/result/win'+str(window_size)
	if not os.path.isdir(result_dir):
		os.mkdir(result_dir)
	f_val_result = open(result_dir+'/val_result3_'+str(date)+'_'+str(j)+'.txt','w')
	f_pre_result = open(result_dir+'/pre_result3_'+str(date)+'_'+str(j)+'.txt','w')
	
	xTest = np.asarray(xTest)
	xTest = xTest.reshape(160,num_data, freq_slice)
	yTest = np.asarray(yTest)
	yTest = ld.OneAndHot(yTest, Kind)

	print("test accuracy %g"%accuracy.eval(feed_dict={x:xTest, y_: yTest, keep_prob: 1.0}))

	ckpt_path = saver.save(sess, "20190527_stft_2/train", j)
	print("save ckpt file : ", ckpt_path)

	y = sess.run(y_conv, feed_dict={x:xTest,keep_prob:1.0})
	for k in range(160):
		f_val_result.write(str(np.argmax(yTest[k])))
		f_val_result.write('\n')
		f_pre_result.write(str(np.argmax(y[k])))
		f_pre_result.write('\n')

sess.close()
