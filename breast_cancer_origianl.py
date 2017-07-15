from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import urllib
import os
import time

# Data sets
DATA_TRAINING = "cancer_training.csv"
DATA_TEST = "cancer_test.csv"




# Load datasets.
training_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename=DATA_TRAINING,
                                                                   target_dtype=np.int,
                                                                   features_dtype=np.float32
                                                                  )

test_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename=DATA_TEST,
                                                               target_dtype=np.int,
                                                               features_dtype=np.float32
                                                           )



# Specify that all features have real-value data
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=9)]

# Build 3 layer DNN with 10, 20, 10 units respectively.
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 20, 10],
                                            n_classes=3,
                                            model_dir="/tmp/cancer_model")





# Fit model
train_start_time=time.time()
def get_train_inputs():
    x = tf.constant(training_set.data)
    y = tf.constant(training_set.target)
    return x, y
classifier.fit(input_fn=get_train_inputs,steps=20000)
train_end_time=time.time()

#eval
def get_test_inputs():
    x = tf.constant(test_set.data)
    y = tf.constant(test_set.target)
    return x, y

test_start_time=time.time()
accuracy_score = classifier.evaluate(input_fn=get_test_inputs, steps=1)["accuracy"]
print('Accuracy: {0:f}'.format(accuracy_score))
test_end_time=time.time()

print("training time:%d seconds"%(train_end_time-train_start_time))
print("testing time:%d seconds"%(test_end_time-test_start_time))



# Classify new samples.
#sample1 = [14.58,21.53,97.41,644.8,0.1054,0.1868,0.1425,0.08783,0.2252,0.06924,0.2545,0.9832,2.11,21.05,0.004452,0.03055,0.02681,0.01352,0.01454,0.003711,17.62,33.21,122.4,896.9,0.1525,0.6643,0.5539,0.2701,0.4264,0.1275]
#sample2 = [10.95,21.35,71.9,371.1,0.1227,0.1218,0.1044,0.05669,0.1895,0.0687,0.2366,1.428,1.822,16.97,0.008064,0.01764,0.02595,0.01037,0.01357,0.00304,12.84,35.34,87.22,514,0.1909,0.2698,0.4023,0.1424,0.2964,0.09606]
#new_samples = np.array([sample1, sample2], dtype=float)
#y = classifier.predict(new_samples)
