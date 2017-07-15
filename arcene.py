#!/bin/bash/python

import tensorflow as tf
import numpy as np
import time

#BINARY CLASSIFICATION CASE
def read_data(Xname, Yname):
	X = np.loadtxt(Xname, dtype=np.float32);
	Y = np.loadtxt(Yname, dtype=np.float32);
	return X, Y;

Xd, Yd = read_data('/Users/rat_racer/Desktop/tensorflow/Acrene/arcene_train.data', '/Users/rat_racer/Desktop/tensorflow/Acrene/arcene_train.labels');

X = Xd;
Y = Yd;

k = 256;

xShape = list(X.shape);
XinputShape = list(X.shape);
YinputShape = list(Y.shape);
xShape.reverse();
xShape.pop();
xShape.insert(1, k);
xShape = np.array(xShape);
yShape = list(Y.shape);
yShape.insert(1, k);
print(yShape);
reshapeofY = np.transpose(np.tile(Y, (k, 1)));
#Create Variable in TensorFlowGraph
Weights = tf.Variable(tf.ones(xShape));
B = tf.Variable(tf.ones(yShape));

def inputs():
	return tf.to_float(Xd), tf.to_float(Yd);

def handleOnGraph():
	return tf.get_default_graph();

def combine(input_data):
	return tf.matmul(input_data, Weights) + B;

def inference(input_data):
	return tf.tanh(tf.nn.relu(combine(input_data)));

def total_loss(input_data, input_labels):
	return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(combine(input_data), reshapeofY));
	#return -tf.reduce_mean(float(input_labels-log(combine(input_data))));

def train(total_loss):
	learning_rate = 0.02;
	return tf.train.AdagradOptimizer(learning_rate).minimize(total_loss);
#MULTICLASS CASE
'''
class MultiClass:
	MultiGraph = tf.Graph();
	with MultiGraph.as_default():

		def __init__(self,data, labels):
			self._X = data;
			self._Y = labels;
			pass;

		def returnDimensions(self):
			return X.shape;

		pass;
'''

with tf.Session() as sess:
	

	print(Xd.shape);
	print(Yd.shape);

	input_data, input_labels = inputs();

	loss = total_loss(input_data, input_labels);
	train_op = train(loss);
	tf.initialize_all_variables().run();
	print(sess.run(inference(input_data)));
	
	coord = tf.train.Coordinator();
	threads = tf.train.start_queue_runners(sess=sess, coord=coord);
	saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
	start=time.time() 
	keep_prob = tf.placeholder("float")

	for i in range(700):
		sess.run([train_op]);
		#for debugging
		if i % 10== 0:
			print "loss: ", sess.run([loss]);
	for i in range(700,706):
		sess.run([train_op]);
		#for debugging
		print "loss: ", sess.run([loss]);

	print(sess.run(inference(input_data)));
	print(sess.run(Weights));
	correct_prediction = tf.equal(tf.argmax(inference(input_data),1), tf.argmax(Y,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


	coord.request_stop();
	coord.join(threads);
	saver_path=saver.save(sess, "/Users/rat_racer/Desktop/tensorflow/saved models/arcene/save_net.ckpt")
	sess.close();

end=time.time()  
print('Runing time.%s Seconds'%(end-start))  
print("Model saved in file:", saver_path)


		
