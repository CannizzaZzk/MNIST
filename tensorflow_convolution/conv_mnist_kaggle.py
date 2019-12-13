#coding: utf-8
import tensorflow as tf
import numpy as np
import pandas as pd

#定义超参数
INPUT_LAYER = 7*7*64
HIDDEN_LAYER = 1024
OUTPUT_LAYER = 10
BATCH_SIZE = 128
EPOCH_NUM = 256
VALIDATION_SIZE = 2000

#常用函数
def get_weight(shape):
	initial =  tf.truncated_normal(shape,stddev=0.1)
	return tf.Variable(initial)

def get_bias(shape):
	initial = tf.zeros(shape)
	return tf.Variable(initial)

def conv2d(x,W):
	return tf.nn.conv2d(x,W,strides = [1,1,1,1],padding = 'SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x,ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')\

def one_hot(labels,num_of_classes):
	num_of_labels = labels.shape[0]
	labels_one_hot = np.zeros((num_of_labels,num_of_classes))
	row_hd = np.arange(num_of_labels)*num_of_classes
	labels_one_hot.flat[row_hd+labels.ravel()] = 1
	return labels_one_hot

#导入数据
data = pd.read_csv('train.csv')

images = data.iloc[:,1:].values
images = images.astype(np.float)

images = np.multiply(images, 1.0 / 255.0)
image_size = images.shape[1]
image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)

labels_flat = data['label']
labels_count = 10
labels = one_hot(labels_flat, labels_count)
labels = labels.astype(np.uint8)

validation_images = images[:VALIDATION_SIZE]
validation_labels = labels[:VALIDATION_SIZE]

train_images = images[VALIDATION_SIZE:]
train_labels = labels[VALIDATION_SIZE:]


x_train =  tf.placeholder(tf.float32,[None,784])
y_train = tf.placeholder(tf.float32,[None,10])

x_image = tf.reshape(x_train,[-1,28,28,1])

#定义网络结构
#第一卷积层
W_conv1 = get_weight([5,5,1,32])
b_conv1 = get_bias([32])
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#第二卷积层
W_conv2 = get_weight([5,5,32,64])
b_conv2 = get_bias([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#全连接层
W_connect1 = get_weight([INPUT_LAYER,HIDDEN_LAYER])
b_connect1 = get_bias([HIDDEN_LAYER])
h_pool2_reshape = tf.reshape(h_pool2,[-1,7*7*64])
h_connect1 = tf.nn.sigmoid(tf.matmul(h_pool2_reshape,W_connect1)+b_connect1)
drop_rate = tf.placeholder(tf.float32)
h_connect1_drop = tf.nn.dropout(h_connect1,drop_rate)

W_connect2 = get_weight([HIDDEN_LAYER,OUTPUT_LAYER])
b_connect2 = get_bias([OUTPUT_LAYER])
y_out = tf.matmul(h_connect1_drop,W_connect2) + b_connect2

#计算交叉熵
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_train, logits=y_out))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#定义测试的准确率
correct_prediction = tf.equal(tf.argmax(y_out, 1), tf.argmax(y_train, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

n_batch = len(train_images)/BATCH_SIZE
n_batch = int(n_batch)

#运行
with tf.Session() as sess:
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	for epoch in range(1,EPOCH_NUM):
		for batch in range(n_batch):
			batch_x = train_images[(batch)*BATCH_SIZE:(batch+1)*BATCH_SIZE]
			batch_y = train_labels[(batch)*BATCH_SIZE:(batch+1)*BATCH_SIZE]
			
			train_accuracy = accuracy.eval(feed_dict={x_train:validation_images, y_train: validation_labels, drop_rate: 1.0})
			print("epoch %d,batch %d, training accuracy %g" % (epoch,batch, train_accuracy))
			
			train_step.run(feed_dict={x_train: batch_x, y_train: batch_y, drop_rate: 0.5})

		#保存模型
	#	saver.save(sess,'model.ckpt',global_step = epoch)
	
	#生成结果
	test_images = pd.read_csv('test.csv').values
	test_images = test_images.astype(np.float)

	test_images = np.multiply(test_images, 1.0 / 255.0)

	predict = tf.argmax(y_out, 1)
	predicted_lables = np.zeros(test_images.shape[0])
	for i in range(0,test_images.shape[0]//BATCH_SIZE):
	    predicted_lables[i*BATCH_SIZE : (i+1)*BATCH_SIZE] = predict.eval(feed_dict={x_train: test_images[i*BATCH_SIZE : (i+1)*BATCH_SIZE], drop_rate: 1.0})


	# save results
	np.savetxt('submission.csv', 
           np.c_[range(1,len(test_images)+1),predicted_lables], 
		              delimiter=',', 
					             header = 'ImageId,Label', 
								            comments = '', 
											           fmt='%d')

sess.close()


