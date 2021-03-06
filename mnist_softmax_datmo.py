# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import json

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None

#################### DATMO LOCATION FOR TASK FILES ##########################

# Location to store information for each experiment provided by Datmo 
# to run multiple experiments at once without overwriting any files. 
SAFE_SAVE_DIR = os.environ['DATMO_TASK_DIR']

#################### DATMO LOCATION FOR TASK FILES ##########################

def basic_classifier(x, num_classes):
  # More info on tf.Variable: https://www.tensorflow.org/api_docs/python/tf/Variable
  W = tf.Variable(tf.zeros([784, num_classes]))
  b = tf.Variable(tf.zeros([num_classes]))
  
  #################### TENSORBOARD ########################

  # Add summary ops to collect data to visualize on tensorboard
  w_h = tf.summary.histogram("weights", W)
  b_h = tf.summary.histogram("biases", b)

  ################### TENSORBOARD ########################

  # More info on tf.matmul: https://www.tensorflow.org/api_docs/python/tf/matmul
  y = tf.matmul(x, W) + b
  return y

def main(_):
  # Import MNIST data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
  num_classes = 10

  # Set hyperparameters
  hyperparameters = {
    "learning_rate": 0.01,
    "training_iterations": 30,
    "batch_size": 100,
    "display_step": 2
  }

  #################### DATMO SNAPSHOT CONFIG ##########################

  # Store hyperparameters in the config.json for Datmo to easily compare experiments
  with open(os.path.join(SAFE_SAVE_DIR, 'config.json'), 'wb') as f:
    f.write(json.dumps(hyperparameters))

  #################### DATMO SNAPSHOT CONFIG ##########################

  # Create the model
  # More info on tf.placeholder: https://www.tensorflow.org/api_docs/python/tf/placeholder
  x = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784

  # Define output placeholder vector
  y_ = tf.placeholder(tf.float32, [None, num_classes])

  # Build the graph for the basic classification
  with tf.name_scope("model") as scope:
    model = basic_classifier(x, num_classes)

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  with tf.name_scope("cost_function") as scope: 
    # Minimize error using cross entropy
    cost_function = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=model))

    ################### TENSORBOARD ########################

    # Create a summary to monitor the cost function on tensorboard
    tf.summary.scalar("cost_function", cost_function)

    ################### TENSORBOARD ########################


  with tf.name_scope("train") as scope:
    # Gradient descent
    train_step = tf.train.GradientDescentOptimizer(hyperparameters['learning_rate']).minimize(cost_function)

  # Initialize all variables
  init = tf.global_variables_initializer()

  ################### TENSORBOARD ########################

  # Merge all summaries into single operator for Tensorboard
  merged_summary_ops = tf.summary.merge_all()

  ################### TENSORBOARD ########################

  # Add ops to save your trained model to a file
  saver = tf.train.Saver()

  with tf.Session() as sess:
    sess.run(init)

    # Set the logs writer to the folder 
    file_writer = tf.summary.FileWriter(os.path.join(SAFE_SAVE_DIR, 'logs'))
    file_writer.add_graph(tf.get_default_graph())

    # Training iterations 
    for iteration in range(hyperparameters['training_iterations']):
      avg_cost = 0.
      total_batch = int(mnist.train.num_examples/hyperparameters['batch_size'])
      # Train
      for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(hyperparameters['batch_size'])
        # Fit training using batch data
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        # Calculate the average cost for the training iteration
        avg_cost += sess.run(cost_function, feed_dict={x: batch_xs, y_: batch_ys})/total_batch

        ################### TENSORBOARD ########################

        # Write logs for each batch 
        summary_str = sess.run(merged_summary_ops, feed_dict={x: batch_xs, y_:batch_ys})
        file_writer.add_summary(summary_str, iteration*total_batch + i)

        ################### TENSORBOARD ########################
      # Display iteration and cost for each display step 
      if iteration % hyperparameters['display_step'] == 0:
        print("Iteration:", '%04d' % (iteration + 1), "cost=", "{:.9f}".format(avg_cost))
    # Training complete!

    #################### DATMO SNAPSHOT FILES ##########################

    # Save all variables for the trained model to disk
    save_path = saver.save(sess, os.path.join(SAFE_SAVE_DIR, 'model.ckpt'))
    print("Model saved in file: %s" % save_path)

    #################### DATMO SNAPSHOT FILES ##########################

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(y_, 1))
    # Define the accuracy 
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # Calculate the accuracy 
    test_data_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images,
                                        y_: mnist.test.labels})
    print("Accuracy:", test_data_accuracy)

    #################### DATMO SNAPSHOT METRICS ##########################

    # Store stats.json for Datmo 
    with open(os.path.join(SAFE_SAVE_DIR, 'stats.json'), 'wb') as f:
      f.write(json.dumps({"test_data_accuracy": float(test_data_accuracy)}))

    #################### DATMO SNAPSHOT METRICS ##########################

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)