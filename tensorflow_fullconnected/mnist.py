import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

#prepare data
BATCH_SIZE = 200
INPUT_LAYER = 784
HIDDEN_LAYER = 400
OUTPUT_LAYER = 10

mnist = input_data.read_data_sets('./data/',one_hot=True)

#build network
x_in = tf.placeholder(tf.float32,[None,784])
y_label = tf.placeholder(tf.float32,[None,10])

w1 = tf.Variable(tf.random_normal([INPUT_LAYER,HIDDEN_LAYER],stddev=1))
w2 = tf.Variable(tf.random_normal([HIDDEN_LAYER,OUTPUT_LAYER],stddev=1))
b1 = tf.Variable(tf.zeros([HIDDEN_LAYER]))
b2 = tf.Variable(tf.zeros([OUTPUT_LAYER]))

a =tf.nn.relu(tf.matmul(x_in,w1)+b1)
a =tf.nn.sigmoid(tf.matmul(x_in,w1)+b1)
#a =tf.matmul(x_in,w1)+b1
#y_out =tf.nn.sigmoid(tf.matmul(a,w2)+b2)
#y_out =tf.nn.relu(tf.matmul(a,w2)+b2)
y_out =tf.matmul(a,w2)+b2

#define loss and training method
predictions = tf.nn.softmax(y_out)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(predictions),reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#accuracy 
result = tf.equal(tf.argmax(predictions,1),tf.argmax(y_label,1))
accuracy = tf.reduce_mean(tf.cast(result,tf.float32))

#run
with tf.Session() as sess:
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	print('b1:',sess.run(b1))
	for i in range(5000):
#		print('batch %d'%i)
		batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)
		sess.run(train_step, feed_dict={x_in: batch_x,y_label: batch_y })
		if i%100 == 0:
			print('The accuracy is :')
			print(sess.run(accuracy,feed_dict={x_in:mnist.test.images,y_label:mnist.test.labels}))
#	print('b1:',sess.run(b1))
