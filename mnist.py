from __future__ import print_function
import tensorflow as tf
import numpy
import time

import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Create the model
x = tf.placeholder("float", [None, 784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y_ = tf.placeholder("float", [None,10])
sess = tf.InteractiveSession()


def weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)
def bias_variable(shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)
def conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')
def max_pool_3x3(x):
      return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],strides=[1, 1, 1, 1], padding='SAME')
# 1.concolution layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1,28,28,1])
#h_conv1= tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_conv1=tf.nn.sigmoid((conv2d(x_image, W_conv1) + b_conv1))
#h_conv1=numpy.square((conv2d(x_image, W_conv1) + b_conv1))
# 2.square actication layer
h_square1=numpy.square(h_conv1)
# 3.pool layer
h_pool1 = max_pool_3x3(h_conv1)
# 4.convolution layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
#h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_conv2=tf.nn.sigmoid((conv2d(h_pool1, W_conv2) + b_conv2))
#h_conv2=numpy.square((conv2d(h_pool1, W_conv2) + b_conv2))
# 5.pool layer
h_pool2 = max_pool_3x3(h_conv2)
# 6.fully connected layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
# 7.square activation layer
h_square2=numpy.square(h_fc1)
# Dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
# 9.output layer
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
start=time.time() 

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.global_variables_initializer())
for i in range(10000):
      batch = mnist.train.next_batch(50)
      if i%100 == 0:
                train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
                print("step %d training accuracy %g"%(i,train_accuracy))
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
print("test accuracy %g"%accuracy.eval(feed_dict={  x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
#saver_path=saver.save(sess, "/Users/rat_racer/Desktop/tensorflow/saved models/mnist_relu/save_net.ckpt")
saver_path=saver.save(sess, "/Users/rat_racer/Desktop/tensorflow/saved models/mnist_sigmoid/save_net.ckpt")
#saver_path=saver.save(sess, "/Users/rat_racer/Desktop/tensorflow/saved models/mnist_square/save_net.ckpt")
end=time.time()  
print('Runing time.%s Seconds'%(end-start))  
print("Model saved in file:", saver_path)
