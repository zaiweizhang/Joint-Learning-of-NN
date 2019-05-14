"""A MNIST linear classifier using linear layers.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

import tensorflow as tf

from random import shuffle

#XXX
import time
import utils


import numpy as np
#XXX
from datagenerator import ImageDataGenerator

#from alexnet import AlexNet
#from inception_v4 import *
#from resnet_v1 import *
#from resnet_v2 import *
#from vgg import *
from nets import inception_v3
#from nets import resnet_v2
from tensorflow.contrib.data import Iterator

from tensorflow.python import pywrap_tensorflow

import os
import subprocess

slim = tf.contrib.slim

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
nvidia_output = subprocess.check_output('nvidia-smi')
if nvidia_output.split("\n")[-3].split(' ')[4] == '1':
    print('using GPU 2')
    os.environ["CUDA_VISIBLE_DEVICES"]="2"
else:
    print('using GPU 1')
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

FLAGS = None

def reinit_variables(sess, variable_list):
  for i in range(len(variable_list)):
    #vs = variable_list[i].get_shape()
    #if 'fc6' in variable_list[i].op.name or 'fc7' in variable_list[i].op.name or 'fc8' in variable_list[i].op.name:
    #if 'logits' in variable_list[i].op.name:
    if 'Logits' in variable_list[i].op.name:
      sess.run(variable_list[i].initializer)

def extract_vars(cur, post, checkpoint):
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint)
    var_to_shape_map = reader.get_variable_to_shape_map()
    var_list = []
    for var in var_to_shape_map:
        var_list.append(var.replace(cur, post))
    return var_list

def main(_):
  ## Some hyper parameters
  mu = 0.01/2.0 #for l2
  t_mu = 10
  #mu = 1 #for l1
  batch_size = 64#32
  display_step = 40
  global_step = 30000#15000
  save_model = 5000
  optimize_w = 5000
  w_lambda_update = mu
  initw_step = 1500
  #num_classes = 50
  num_classes = [200, 120, 102, 196, 100]#, 101]
  #num_classes = [200, 120, 196, 70]
  #num_classes = [50, 50, 50, 50, 50]
  #last_layer_name = "fc8"
  #last_layer_name = "logits"
  last_layer_name = "Logits"
  #var_in_checkpoint = extract_vars("resnet_v2_50", "model0", FLAGS.pre_train)
  var_in_checkpoint = extract_vars("InceptionV3", "model0", FLAGS.pre_train)
  #var_in_checkpoint = extract_vars("vgg_16", "model0", FLAGS.pre_train)
 
  # Import data
  # Multiple dataset
  data = []
  testdata = []
  train_init_op = []
  test_init_op = []
  iterator = []
  next_batch = []
  #XXX 
  with tf.device('/cpu:0'):
    for i in range(int(FLAGS.num_data)):
      data.append(ImageDataGenerator(FLAGS.data_dir+"ex2_"+str(i+1)+"_train.txt",
                                     mode='training',
                                     batch_size=batch_size,
                                     num_classes=num_classes[i],
                                     shuffle=True))
      testdata.append(ImageDataGenerator(FLAGS.data_dir+"ex2_"+str(i+1)+"_test.txt",
                                         mode='inference',
                                         batch_size=batch_size,
                                         num_classes=num_classes[i],
                                         shuffle=False))
  
      iterator.append(Iterator.from_structure(data[i].data.output_types, data[i].data.output_shapes))
      next_batch.append(iterator[i].get_next())
      train_init_op.append(iterator[i].make_initializer(data[i].data))
      test_init_op.append(iterator[i].make_initializer(testdata[i].data))

  # Create the model
  x_list = []
  #XXX
  for i in range(1):
    x_list.append(tf.placeholder(tf.float32, [None, 299, 299, 3], name="data"+str(i)))

  # Define loss and optimizer
  y_losses = []
  #XXX
  for i in range(int(FLAGS.num_data)):
    y_losses.append(tf.placeholder(tf.float32, [None, num_classes[i]], name="loss"+str(i)))

  # Build the graph for the deep net
  # Multiple networks
  variable_list = []
  layer_list = []
  test_layer_list = []
  paired_loss = []
  data_loss = []
  optimizer = []
  paired_optimizer = []
  accuracy_list = []
  test_accuracy_list = []
  joint_optimizer = []
  joint_loss = []
  isolated_optimizer = []
  naive_joint_optimizer = []
  naive_joint_loss = []
  winitial_optimizer = []
  model_list = []
  shared_variable_list = []
  stored_vars = []
  update_ops = []
  #XXX
  for i in range(1):
    #model_list.append(AlexNet(x_list[i], num_classes=num_classes, global_name='model'+str(i)))
    #with slim.arg_scope(resnet_v2.resnet_arg_scope()):
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
    #with slim.arg_scope(vgg_arg_scope()):
      #net,_ = resnet_v1_50(x_list[i], num_classes=num_classes[i], scope='model'+str(i))
      #net,_ = resnet_v2.resnet_v2_50(x_list[i], num_classes=num_classes[i], is_training=True, scope='model'+str(i))
      #testnet,_ = resnet_v2.resnet_v2_50(x_list[i], num_classes=num_classes[i], scope='model'+str(i), is_training=False, reuse=True)
      net,_ = inception_v3.inception_v3(x_list[i], num_classes=num_classes, scope='model'+str(i), create_aux_logits=False)
      testnet,_ = inception_v3.inception_v3(x_list[i], num_classes=num_classes, scope='model'+str(i), is_training=False, reuse=True, create_aux_logits=False)

      #variables_to_restore = slim.get_variables_to_restore()
      #print (variables_to_restore)
      #resnet_saver = tf.train.Saver()#var_list=variables_to_restore)
      #net,_ = vgg_16(x_list[i], num_classes=num_classes[i], scope='model'+str(i))
    #layer_list.append(net)
    layer_list = net
    #test_layer_list.append(testnet)
    test_layer_list = testnet

    ## Special for recnet to remove the bias layers
    templist = []
    org_vars = slim.get_trainable_variables(scope='model'+str(i))
    update_ops.append(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='model'+str(i)))
    print ("Global vars")
    for var in org_vars:
      print (var.op.name)
      if var.op.name in var_in_checkpoint:#not ("Conv2d" in var.op.name and "bias" in var.op.name):
        templist.append(var)
    print ("End vars")
    variable_list.append(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model'+str(i)))    
    stored_vars.append(templist)

    print (len(stored_vars[0]))
    #print (stored_vars)
    #sys.exit(0)

    temp = []
    for var in variable_list[i]:
      #if last_layer_name not in var.op.name and "beta" not in var.op.name and "gamma" not in var.op.name:
      if last_layer_name not in var.op.name and "BatchNorm" not in var.op.name:
        temp.append(var)
    shared_variable_list.append(temp)

  #XXX
  # claim just variables
  copy_to_model0_op = []
  copy_from_model0_op = []
  model0_vars = variable_list[0]#tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model0')
  print ("Shared variables")
  for i in range(1, int(FLAGS.num_data)+1):
    variable_list_i = []
    shared_variable_list_i = []
    copy_from_model0_op_i = []
    copy_to_model0_op_i = []
    for var in model0_vars:
        new_var_name = var.name.replace('model0', 'model%d' % i)
        if var in tf.trainable_variables():
            trainable = True
        else:
            trainable = False
        new_var = tf.get_variable(new_var_name.split(':')[0], shape=var.shape,
                dtype=var.dtype, trainable=trainable)
        #if last_layer_name not in new_var.op.name and "beta" not in var.op.name and "gamma" not in var.op.name:
        if last_layer_name not in new_var.op.name and "BatchNorm" not in var.op.name:
            if i == 1:
                print (new_var.op.name)
            shared_variable_list_i.append(new_var)
        variable_list_i.append(new_var)
        copy_from_model0_op_i.append(new_var.assign(var))
        copy_to_model0_op_i.append(var.assign(new_var))
    shared_variable_list.append(shared_variable_list_i)
    variable_list.append(variable_list_i)
    copy_to_model0_op.append(copy_to_model0_op_i)
    copy_from_model0_op.append(copy_from_model0_op_i)
  #config = tf.ConfigProto()
  #config.gpu_options.allow_growth=True
  #with tf.Session(config=config) as sess:
  #    sess.run(tf.global_variables_initializer())
  #    time.sleep(100.0) 
  var_loss = []
  for k in range(len(shared_variable_list[0])):
    temp1 = []
    #XXX
    for i in range(int(FLAGS.num_data)):
      temp2 = []
      #XXX
      for j in range(int(FLAGS.num_data)):
        temp2.append(0)
      temp1.append(temp2)
    var_loss.append(temp1)
  
  ## Savers
  saver = tf.train.Saver()
  ## For model1
  #???
  pre_saver = []
  pretrain = {}
  for i in range(len(stored_vars[0])):
    #if last_layer_name not in stored_vars[0][i].op.name and "BatchNorm" not in stored_vars[0][i].op.name:
    if last_layer_name not in stored_vars[0][i].op.name:
      org_name = stored_vars[0][i].op.name.replace("model0", "InceptionV3")
      #org_name = stored_vars[0][i].op.name.replace("model0", "vgg_16")
      #org_name = stored_vars[0][i].op.name.replace("model0", "resnet_v2_50")
      pretrain[org_name] = stored_vars[0][i]
      print (org_name)
  pre_saver = tf.train.Saver(pretrain)

  #XXX
  weight_graph = tf.placeholder(tf.float32, [len(shared_variable_list[0]), int(FLAGS.num_data)])
  weight_scale = tf.placeholder(tf.float32, [len(shared_variable_list[0])])
  w_lambda = tf.placeholder(tf.float32)

  #XXX
  for i in range(int(FLAGS.num_data)):
    with tf.variable_scope('data_loss'+str(i)):
      cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_losses[i], logits=layer_list[i])
      data_loss.append(tf.reduce_mean(cross_entropy))

  #XXX
  for i in range(int(FLAGS.num_data)):
    ## Add the pairwise training loss and optimizer
    #XXX
    for j in range(i+1, int(FLAGS.num_data)):
      with tf.name_scope('paired_weight_loss'+str(i)+str(j)):
        w_loss = 0
        for n_var in range(len(shared_variable_list[0])):## Same model
          #if FLAGS.norm == "l1":
          #  cur_var_loss = tf.reduce_mean(tf.abs(shared_variable_list[i][n_var] - shared_variable_list[j][n_var]))
          #else:
          #XXX
          cur_var_loss = tf.nn.l2_loss(shared_variable_list[i+1][n_var] - shared_variable_list[j+1][n_var])
          var_loss[n_var][i][j] = cur_var_loss
          var_loss[n_var][j][i] = cur_var_loss
          #w_loss += cur_var_loss

        #pair_loss = mu*w_loss + data_loss[i] + data_loss[j]
        #paired_loss.append(pair_loss)

        #with tf.name_scope('adam_optimizer'):
        #  paired_optimizer.append(tf.train.AdamOptimizer(1e-4).minimize(pair_loss, var_list=variable_list[i]+variable_list[j]))

  ## Add the joint loss here
  winit_losses = []
  for i in range(1):
    with tf.name_scope('joint'+str(i)):
      w_loss = 0
      naive_joint = 0
      for j in range(1, int(FLAGS.num_data)+1):
        if i != j:
          for n_var in range(len(shared_variable_list[0])):## Same model
            if FLAGS.norm == "l1":
              w_loss += weight_graph[n_var][j-1]*tf.reduce_mean(tf.abs(shared_variable_list[i][n_var] - shared_variable_list[j][n_var]))
              naive_joint += tf.reduce_mean(tf.abs(shared_variable_list[i][n_var] - shared_variable_list[j][n_var]))
            else:
              #if n_var <= 9:
              w_loss += weight_graph[n_var][j-1]*tf.nn.l2_loss(shared_variable_list[i][n_var] - shared_variable_list[j][n_var]) * (1.0 / weight_scale[n_var])
              naive_joint += tf.nn.l2_loss(shared_variable_list[i][n_var] - shared_variable_list[j][n_var]) * (1.0 / weight_scale[n_var])
              #else:
              #  w_loss += weight_graph[n_var][i][j]*tf.nn.l2_loss(shared_variable_list[i][n_var] - shared_variable_list[j][n_var])*t_mu
              #  naive_joint += tf.nn.l2_loss(shared_variable_list[i][n_var] - shared_variable_list[j][n_var])*t_mu
                
      w_loss *= w_lambda
      winit = naive_joint * w_lambda
      naive_joint *= w_lambda
      winit_losses.append(winit)
      for k in range(int(FLAGS.num_data)):
        joint_loss.append(data_loss[k]+w_loss)
        naive_joint_loss.append(data_loss[k]+naive_joint)
      #joint_optimizer.append(tf.train.AdamOptimizer(1e-4).minimize(joint_loss[i], var_list=variable_list[i], name='joint'))
      #isolated_optimizer.append(tf.train.AdamOptimizer(1e-4).minimize(data_loss[i], var_list=variable_list[i], name='isolate'))
      #naive_joint_optimizer.append(tf.train.AdamOptimizer(1e-4).minimize(naive_joint_loss[i], var_list=variable_list[i], name='naive_joint'))
      #winitial_optimizer.append(tf.train.AdamOptimizer(1e-4).minimize(data_loss[i]+winit, var_list=variable_list[i], name='w_init'))
      #for k in range(int(FLAGS.num_data)):

      #with tf.variable_scope('Adam_joint'):
      #    joint_optimizer.append(tf.train.AdamOptimizer(1e-4).minimize(joint_loss[i], var_list=variable_list[i]))
      #shallow_joint = []
      #for k in range(int(FLAGS.num_data)):
      #    shallow_joint_k = []
      #    for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Adam_joint'):
      #        print(var.name, var.shape)
      #        #with tf.device('/cpu:0'):
      #        new_var = tf.get_variable(name=var.name.replace('Adam_joint', 'Adam_shallow_joint%d' % k).split(':')[0], shape=var.shape, dtype=var.dtype, trainable=False)
      #        shallow_joint_k.append(new_var)
      #    shallow_joint.append(shallow_joint_k)
            
      #with tf.variable_scope('Adam_isolated'):
      #    isolated_optimizer.append(tf.train.AdamOptimizer(1e-4).minimize(data_loss[i], var_list=variable_list[i]))
      #shallow_isolated = []
      #for k in range(int(FLAGS.num_data)):
      #    shallow_isolated_k = []
      #    for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Adam_isolated'):
      #        new_var = tf.get_variable(name=var.name.replace('Adam_isolated', 'Adam_shallow_isolated%d' % k).split(':')[0], shape=var.shape, dtype=var.dtype, trainable=False)
      #        shallow_isolated_k.append(new_var)
      #    shallow_isolated.append(shallow_isolated_k)

      #with tf.variable_scope('Adam_naive'):
      #    naive_joint_optimizer.append(tf.train.AdamOptimizer(1e-4).minimize(naive_joint_loss[i], var_list=variable_list[i]))
      #shallow_naive = []
      #for k in range(int(FLAGS.num_data)):
      #    shallow_naive_k = []
      #    for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Adam_naive'):
      #        new_var = tf.get_variable(name=var.name.replace('Adam_naive', 'Adam_shallow_naive%d' % k).split(':')[0], shape=var.shape, dtype=var.dtype, trainable=False)
      #        shallow_naive_k.append(new_var)
      #    shallow_naive.append(shallow_naive_k)
     
      if FLAGS.stage == 0:
          for k in range(int(FLAGS.num_data)):
              #with tf.variable_scope('Adam_winitial%d' % k):
              #    winitial_optimizer.append(tf.train.AdamOptimizer(1e-4).minimize(data_loss[k]+winit, var_list=variable_list[i]))
              with tf.variable_scope('Moment_winitial%d' % k):
                  with tf.control_dependencies(update_ops[0]):
                      winitial_optimizer.append(tf.train.MomentumOptimizer(1e-2, momentum=0.9).minimize(data_loss[k]+winit, var_list=variable_list[i]))
      if FLAGS.stage == 1:
          for k in range(int(FLAGS.num_data)):
              #with tf.variable_scope('Adam_joint%d' % k):
              #    joint_optimizer.append(tf.train.AdamOptimizer(1e-4).minimize(joint_loss[k], var_list=variable_list[i]))
              num_train = int(np.floor(data[k].data_size / batch_size))
              with tf.variable_scope('Moment_joint%d' % k):
                  global_step_iso = tf.Variable(0, trainable=False)
                  starter_learning_rate = 0.01
                  learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step_iso, num_train*60, 0.1, staircase=True)
                  #learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step_iso, 10000, staircase=True)
                  #learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step_iso, num_train*2, 0.94, staircase=True)
                  with tf.control_dependencies(update_ops[0]):
                      joint_optimizer.append(tf.train.MomentumOptimizer(learning_rate, momentum=0.9).minimize(joint_loss[k], var_list=variable_list[i], global_step=global_step_iso))
      if FLAGS.stage == 2:
          for k in range(int(FLAGS.num_data)):
              #with tf.variable_scope('Adam_isolated%d' % k):
              #    isolated_optimizer.append(tf.train.AdamOptimizer(1e-4).minimize(data_loss[k], var_list=variable_list[i]))
              num_train = int(np.floor(data[k].data_size / batch_size))
              with tf.variable_scope('Moment_isolated%d' % k):
                  global_step_iso = tf.Variable(0, trainable=False)
                  starter_learning_rate = 0.01
                  learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step_iso, num_train*60, 0.1, staircase=True)
                  #learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step_iso, num_train*2, 0.94, staircase=True)
                  with tf.control_dependencies(update_ops[0]):
                      isolated_optimizer.append(tf.train.MomentumOptimizer(learning_rate, momentum=0.9).minimize(data_loss[k], var_list=variable_list[i], global_step=global_step_iso))
                  #isolated_optimizer.append(tf.train.AdamOptimizer(learning_rate).minimize(data_loss[k], var_list=variable_list[i], global_step=global_step_iso))
      if FLAGS.stage == 3:
          for k in range(int(FLAGS.num_data)):
              #with tf.variable_scope('Adam_naive%d' % k):
              #    naive_joint_optimizer.append(tf.train.AdamOptimizer(1e-4).minimize(naive_joint_loss[k], var_list=variable_list[i]))
              num_train = int(np.floor(data[k].data_size / batch_size))
              with tf.variable_scope('Moment_naive%d' % k):
                  global_step_iso = tf.Variable(0, trainable=False)
                  starter_learning_rate = 0.01
                  learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step_iso, num_train*60, 0.1, staircase=True)
                  #learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step_iso, num_train*2, 0.94, staircase=True)
                  with tf.control_dependencies(update_ops[0]):
                      naive_joint_optimizer.append(tf.train.MomentumOptimizer(learning_rate, momentum=0.9).minimize(naive_joint_loss[k], var_list=variable_list[i], global_step=global_step_iso))
      #shallow_joint = []
      #for k in range(int(FLAGS.num_data)):
      #    shallow_joint_k = []
      #    for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Adam_joint'):
      #        print(var.name, var.shape)
      #        #with tf.device('/cpu:0'):
      #        new_var = tf.get_variable(name=var.name.replace('Adam_joint', 'Adam_shallow_joint%d' % k).split(':')[0], shape=var.shape, dtype=var.dtype, trainable=False)
      #        shallow_joint_k.append(new_var)
      #    shallow_joint.append(shallow_joint_k)

      #shallow_winitial = []
      #copy_to_main_op = []
      #copy_from_main_op = []
      #for k in range(int(FLAGS.num_data)):
      #    shallow_winitial_k = []
      #    copy_to_main_op_k = []
      #    copy_from_main_op_k = []
      #    for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Adam_winitial'):
      #        new_var = tf.get_variable(name=var.name.replace('Adam_winitial', 'Adam_shallow_winitial%d' % k).split(':')[0], shape=var.shape, dtype=var.dtype, trainable=False)
      #        shallow_winitial_k.append(new_var)
      #        copy_from_main_op_k.append(new_var.assign(var))
      #        copy_to_main_op_k.append(var.assign(new_var))
      #    copy_to_main_op.append(copy_to_main_op_k)
      #    copy_from_main_op.append(copy_from_main_op_k)
      #    shallow_winitial.append(shallow_winitial_k)
        
  for i in range(int(FLAGS.num_data)):
    with tf.name_scope('accuracy'+str(i)):
      correct_prediction = tf.equal(tf.argmax(layer_list[i], 1), tf.argmax(y_losses[i], 1))
      correct_prediction = tf.cast(correct_prediction, tf.float32)
      accuracy = tf.reduce_mean(correct_prediction)
      accuracy_list.append(accuracy)
    
      test_correct_prediction = tf.equal(tf.argmax(test_layer_list[i], 1), tf.argmax(y_losses[i], 1))
      test_correct_prediction = tf.cast(test_correct_prediction, tf.float32)
      test_accuracy = tf.reduce_mean(test_correct_prediction)
      test_accuracy_list.append(test_accuracy)
    
  graph_location = FLAGS.log_location + "log_graph"
  print('Saving graph to: %s' % graph_location)
  train_writer = tf.summary.FileWriter(graph_location)
  train_writer.add_graph(tf.get_default_graph())
  print('done')
  """ 
  training
  """
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    #XXX
    """
    for i in range(1, int(FLAGS.num_data)+1):
      #???
      #model_list[0].load_initial_weights(sess)
      pre_saver.restore(sess, FLAGS.pre_train)
      #print (sess.run(variable_list[0][0]))
      sess.run(copy_from_model0_op[i-1])
      reinit_variables(sess, variable_list[i])
    """
    w_scale = []
    for i in range(len(shared_variable_list[0])): #get the weight scale
      #w_scale.append(1.0 / sess.run(tf.nn.l2_loss(shared_variable_list[0][i])))
      w_scale.append(1.0)
    print (w_scale)

    distMat = []
    for k in range(len(shared_variable_list[0])):
      temp1 = []
      for i in range(int(FLAGS.num_data)):
        temp2 = []
        for j in range(int(FLAGS.num_data)):
          temp2.append(0)
        temp1.append(temp2)
      distMat.append(temp1)
    distMat_test = []
    for i in range(int(FLAGS.num_data)):
      temp2 = []
      for j in range(int(FLAGS.num_data)):
        temp2.append(0)
      distMat_test.append(temp2)
  
    ##First we initialize the bias
    ##Each train 1000 epochs
    data_sum = 0
    w_sum = 0
    w_lambda_update = mu
    """ ==== """
    if FLAGS.stage == 0:
        print ('Initializeing w')
        for i in range(initw_step):
          for j in range(int(FLAGS.num_data)):
            """ get data"""
            #XXX
            x = data[j]
            numtrain = int(np.floor(x.data_size / batch_size))
            sess.run(copy_to_model0_op[j])
            #numtrain = 1
            if i % numtrain == 0:
              sess.run(train_init_op[j])
            #XXX
            x_batch_images, x_batch_labels = sess.run(next_batch[j])
            #np.random.shuffle(x_batch_labels)
            #x_batch_images = np.zeros((10, 227, 227, 3))
            #x_batch_labels = np.zeros((10, num_classes[j]))
            
            """ get weight """
            winitial_optimizer[j].run(feed_dict={x_list[0]: x_batch_images,
                y_losses[j]: x_batch_labels, w_lambda: w_lambda_update, weight_scale: w_scale})
            if i % display_step == 0:
              train_accuracy = accuracy_list[j].eval(feed_dict={x_list[0]: x_batch_images, y_losses[j]: x_batch_labels})
              dataloss = data_loss[j].eval(feed_dict={x_list[0]: x_batch_images, y_losses[j]: x_batch_labels})
              ylabel = layer_list[j].eval(feed_dict={x_list[0]: x_batch_images})
              wloss = winit_losses[0].eval(feed_dict={w_lambda: w_lambda_update, weight_scale: w_scale})
              print('Epoch %g, dataset %g, training accuracy %g, data loss %g, wloss %g' % (i, j+1, train_accuracy, dataloss, wloss))
              sys.stdout.flush()
            if (i+1) % numtrain == 0:
              test_accuracy = 0
              sess.run(test_init_op[j])
              numtest = int(np.floor(testdata[j].data_size / batch_size))
              for iter1 in range(numtest):
                #XXX
                x_batch_images, x_batch_labels = sess.run(next_batch[j])
                #x_batch_images = np.zeros((10, 227, 227, 3))
                #x_batch_labels = np.zeros((10, num_classes[j]))
                
                test_accuracy += test_accuracy_list[j].eval(feed_dict={
                  x_list[0]: x_batch_images, y_losses[j]: x_batch_labels})
              test_accuracy /= numtest
              print('Epoch %g, dataset %g, test accuracy %g' % (i, j+1, test_accuracy))
              sys.stdout.flush()
            sess.run(copy_from_model0_op[j])
          #if (i+1) % save_model == 0:
          #  saver.save(sess, FLAGS.log_location+"naive_joint_model"+str(i+1)+".ckpt")
        saver.save(sess, FLAGS.log_location+"pre_joint_model"+str(i+1)+".ckpt")

        ## Fill in the data into distMat
        """
        for i in range(int(FLAGS.num_data)):
        ## Add the pairwise training loss and optimizer
          for j in range(i+1, int(FLAGS.num_data)):
            for n_var in range(len(shared_variable_list[i])):
              print ("for var "+shared_variable_list[i][n_var].op.name)
              sys.stdout.flush()
              distMat[n_var][i][j] = var_loss[n_var][i][j].eval()
              print ("we get first "+str(var_loss[n_var][j][i].eval()))
              sys.stdout.flush()
              distMat[n_var][j][i] = var_loss[n_var][j][i].eval()
              print ("we get second "+str(var_loss[n_var][j][i].eval()))
              sys.stdout.flush()

        ##Optimize the dist matrix here
        print (distMat)
        np.save("tempdist", distMat)
        #print (distMat_test)
        #???
        w_opt = utils.optimizeW(distMat, int(FLAGS.num_data))
        np.save("tempw", w_opt)
        """
        #w_opt = np.ones([14, 5, 5])
    else:
        """ Simply load bias and scale here, not implemented yet"""
        w_opt = np.load("tempw.npy")

    ##Alternating minization here
    if FLAGS.stage == 1:
        ## Clear the dataset
        for i in range(1, int(FLAGS.num_data)+1):
          #model_list[i].load_trained_weights(sess, pre_saver[i], FLAGS.pre_train)
          #model_list[0].load_initial_weights(sess)
          pre_saver.restore(sess, FLAGS.pre_train)
          sess.run(copy_from_model0_op[i-1])
          reinit_variables(sess, variable_list[i])

        print('Begin joint training!')
        data_sum = 0
        w_sum = 0
        w_lambda_update = mu
        for i in range(global_step):
          for j in range(int(FLAGS.num_data)):
            x = data[j]
            numtrain = int(np.floor(x.data_size / batch_size))
            sess.run(copy_to_model0_op[j])
            if i % numtrain == 0:
              sess.run(train_init_op[j])
            #XXX
            x_batch_images, x_batch_labels = sess.run(next_batch[j])
            #x_batch_images = np.zeros(x_list[0].shape.as_list())
            #x_batch_labels = np.zeros(y_losses[0].shape.as_list())
            #w_opt = np.ones(weight_graph.shape.as_list())
            if i % display_step == 0:
              train_accuracy = accuracy_list[j].eval(feed_dict={x_list[0]: x_batch_images, y_losses[j]: x_batch_labels})
              jloss = joint_loss[j].eval(feed_dict={x_list[0]: x_batch_images,
                  y_losses[j]: x_batch_labels, weight_graph: w_opt[:, j, :], w_lambda: w_lambda_update, weight_scale: w_scale})
              dataloss = data_loss[j].eval(feed_dict={x_list[0]: x_batch_images, y_losses[j]: x_batch_labels})
              w_sum += jloss - dataloss
              data_sum += dataloss
              if j == int(FLAGS.num_data) - 1:
                if w_sum > data_sum:
                  print ("wloss is higher, reducing lambda %g" % (w_lambda_update))
                  #w_lambda_update /= 2.0
                data_sum = 0
                w_sum = 0
              print('Epoch %g, dataset %g, training accuracy %g, joint_loss %g, data loss %g' % (i, j+1, train_accuracy, jloss, dataloss))
              sys.stdout.flush()
            joint_optimizer[j].run(feed_dict={x_list[0]: x_batch_images, y_losses[j]: x_batch_labels, weight_graph: w_opt[:, j, :], w_lambda: w_lambda_update, weight_scale: w_scale})
            if (i+1) % numtrain == 0:
              test_accuracy = 0
              sess.run(test_init_op[j])
              numtest = int(np.floor(testdata[j].data_size / batch_size))
              for iter1 in range(numtest):
                x_batch_images, x_batch_labels = sess.run(next_batch[j])
                test_accuracy += test_accuracy_list[j].eval(feed_dict={
                  x_list[0]: x_batch_images, y_losses[j]: x_batch_labels})
              test_accuracy /= numtest
              print('Epoch %g, dataset %g, test accuracy %g' % (i, j+1, test_accuracy))
              sys.stdout.flush()
            sess.run(copy_from_model0_op[j])
          if (i+1) % optimize_w == 0:
            ## Fill in the data into distMat
            """
            for i in range(int(FLAGS.num_data)):
              ## Add the pairwise training loss and optimizer
              for j in range(i+1, int(FLAGS.num_data)):
                for n_var in range(len(shared_variable_list[i])):
                  distMat[n_var][i][j] = var_loss[n_var][i][j].eval()
                  distMat[n_var][j][i] = var_loss[n_var][j][i].eval()
            w_opt = utils.optimizeW(distMat, int(FLAGS.num_data))
            """
            print (w_opt)
          if (i+1) % save_model == 0:
            saver.save(sess, FLAGS.log_location+"joint_model"+str(i+1)+".ckpt")
    
    ## Begin isolated training
    """ ==== """
    if FLAGS.stage == 2:
        print ('Isolated training begin')
        data_sum = 0
        w_sum = 0
        w_lambda_update = mu

        for i in range(1, int(FLAGS.num_data)+1):
          #model_list[i].load_trained_weights(sess, pre_saver[i], FLAGS.pre_train)
          #model_list[0].load_initial_weights(sess)
          #saver = tf.train.Saver(var_list=shared_variable_list[0])
          #saver.restore(sess, 'Shareall.model/isolate_model15000.ckpt')
          pre_saver.restore(sess, FLAGS.pre_train)
          #resnet_saver.restore(sess, FLAGS.pre_train)
          sess.run(copy_from_model0_op[i-1])
          #print (variable_list[i])
          reinit_variables(sess, variable_list[i])
          #print (variable_list[0][3].op.name)
          #print (sess.run(variable_list[0][3]))
          #sys.exit(0)
        train_accuracy = [0]*int(FLAGS.num_data)
        for i in range(global_step):
          for j in range(int(FLAGS.num_data)):
            x = data[j]
            numtrain = int(np.floor(x.data_size / batch_size))
            if i % numtrain == 0:
              sess.run(train_init_op[j])
            sess.run(copy_to_model0_op[j])
            x_batch_images, x_batch_labels = sess.run(next_batch[j])
            #if i % display_step == 0:
            train_accuracy[j] += accuracy_list[j].eval(feed_dict={x_list[0]: x_batch_images, y_losses[j]: x_batch_labels})
            dataloss = data_loss[j].eval(feed_dict={x_list[0]: x_batch_images, y_losses[j]: x_batch_labels})
            isolated_optimizer[j].run(feed_dict={x_list[0]: x_batch_images, y_losses[j]: x_batch_labels})
            if (i+1) % numtrain == 0:
              test_accuracy = 0
              print('Epoch %g, dataset %g, training accuracy %g, data loss %g' % (i, j+1, train_accuracy[j]/numtrain, dataloss))
              sys.stdout.flush()
              train_accuracy[j] = 0
              sess.run(test_init_op[j])
              numtest = int(np.floor(testdata[j].data_size / batch_size))
              for iter1 in range(numtest):
                x_batch_images, x_batch_labels = sess.run(next_batch[j])
                test_accuracy += test_accuracy_list[j].eval(feed_dict={
                  x_list[0]: x_batch_images, y_losses[j]: x_batch_labels})
              test_accuracy /= numtest
              print('Epoch %g, dataset %g, test accuracy %g' % (i, j+1, test_accuracy))
              sys.stdout.flush()
            sess.run(copy_from_model0_op[j])

          if (i+1) % save_model == 0:
            saver.save(sess, FLAGS.log_location+"isolate_model"+str(i+1)+".ckpt")
        #saver.save(sess, FLAGS.log_location+"isolate_model.ckpt")
      
    ### naive joint training
    """ ==== """
    if FLAGS.stage == 3:
        print ('Naive joint training begin')
        data_sum = 0
        w_sum = 0
        w_lambda_update = mu
        for i in range(1, int(FLAGS.num_data)+1):
          #model_list[i].load_trained_weights(sess, pre_saver[i], FLAGS.pre_train)
          #model_list[0].load_initial_weights(sess)
          pre_saver.restore(sess, FLAGS.pre_train)
          sess.run(copy_from_model0_op[i-1])
          reinit_variables(sess, variable_list[i])

        for i in range(global_step):
          for j in range(int(FLAGS.num_data)):
            x = data[j]
            numtrain = int(np.floor(x.data_size / batch_size))
            if i % numtrain == 0:
              sess.run(train_init_op[j])
            x_batch_images, x_batch_labels = sess.run(next_batch[j])
            naive_joint_optimizer[j].run(feed_dict={x_list[0]: x_batch_images, y_losses[j]: x_batch_labels, w_lambda: w_lambda_update, weight_scale: w_scale})
            if i % display_step == 0:
              train_accuracy = accuracy_list[j].eval(feed_dict={x_list[0]: x_batch_images, y_losses[j]: x_batch_labels})
              dataloss = data_loss[j].eval(feed_dict={x_list[0]: x_batch_images, y_losses[j]: x_batch_labels})
              print('Epoch %g, dataset %g, training accuracy %g, data loss %g' % (i, j+1, train_accuracy, dataloss))
              sys.stdout.flush()
            if (i+1) % numtrain == 0:
              test_accuracy = 0
              sess.run(test_init_op[j])
              numtest = int(np.floor(testdata[j].data_size / batch_size))
              for iter1 in range(numtest):
                x_batch_images, x_batch_labels = sess.run(next_batch[j])
                test_accuracy += test_accuracy_list[j].eval(feed_dict={
                  x_list[0]: x_batch_images, y_losses[j]: x_batch_labels})
              test_accuracy /= numtest
              print('Epoch %g, dataset %g, test accuracy %g' % (i, j+1, test_accuracy))
              sys.stdout.flush()
          if (i+1) % save_model == 0:
            saver.save(sess, FLAGS.log_location+"naive_joint_model"+str(i+1)+".ckpt")
    

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  parser.add_argument('--num_data', type=str,
                      default='5',
                      help='Number of datasets')
  parser.add_argument('--log_location', type=str,
                      default='logs',
                      help='Directory for storing the log files')
  parser.add_argument('--norm', type=str,
                      default='l2',
                      help='Type of norm between variables')
  parser.add_argument('--pre_train', type=str,
                      default='logs/checkpoint',
                      help='checkpoint path')
  parser.add_argument('--stage', type=int,
                      default='0',
                      help='0:winitial, 1:joint, 2:isolated, 3:naive')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
