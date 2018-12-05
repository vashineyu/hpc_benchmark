from __future__ import print_function

import os
import sys

import numpy as np
import pandas as pd
import time
import glob
import re
import random
import datetime
import pickle
#import cv2
import tensorflow as tf
import argparse
#####################
####### Setup #######
#####################
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default = 60, type = int)
parser.add_argument('--result_dir', default = './', type = str)
parser.add_argument('--test_mode', default = 0, type = int)
FLAGS = parser.parse_args()

bin_root = "/home/aetherai2018/benchmark/"
data_dir = "/work1/aetherai2018/data/"
model_collection_path = "./tf-pretrain/"

sys.path.append(os.path.join(bin_root, "model_resnet"))
sys.path.append(os.path.join(bin_root, "model_inception_resnet"))
from resnet import *

pretrain_model = "resnet_50"
num_epochs = 100

#####################
PRETRAIN_DICT = {'resnet_50': model_collection_path + '/resnet_v2_50/model.ckpt',
                 'resnet_101': model_collection_path + '/resnet_v2_101/model.ckpt',
                 'resnet_152': model_collection_path + '/resnet_v2_152/model.ckpt',
                 'inception_resnet': model_collection_path + '/inception_resnet_v2/model.ckpt'
                }
    
# image preprocessing function corresponding to each pre-train model
CORRESPONDING_PREPROC = {
    'resnet_50': tf.keras.applications.resnet50.preprocess_input,
    'resnet_101': tf.keras.applications.resnet50.preprocess_input,
    'resnet_152': tf.keras.applications.resnet50.preprocess_input,
    'inception_resnet': tf.keras.applications.inception_resnet_v2.preprocess_input
    }

# If you want to do grad-cam, responding layers should be sent
CORRESPONDING_LAYERS = {
    'resnet_50': 'resnet_v2_50/block4/unit_3/bottleneck_v2/add:0',
    'resnet_101': 'resnet_v2_101/block4/unit_3/bottleneck_v2/add:0',
    'resnet_152': 'resnet_v2_152/block4/unit_3/bottleneck_v2/add:0'
}

try:
    os.mkdir(FLAGS.result_dir)
except:
    pass

#if os.path.exists(FLAGS.result_dir):
#    pass
#else:
#    os.makedirs(FLAGS.result_dir)

#####################
### Data Pipeline ###
#####################
train_list = [os.path.join(data_dir+'/train',i) for i in os.listdir(os.path.join(data_dir, 'train'))]
valid_list = [os.path.join(data_dir+'/test',i) for i in os.listdir(os.path.join(data_dir, 'test'))]
print('Numbers of training pickle: %i, test pickle: %i' % (len(train_list), len(valid_list)))

import queue
from threading import Thread, Event, Timer
class DataGenerator():
    def __init__(self, batch_size, data_path_list, n_class = 3, f_preproc = None, aug_params = None):
        """
        Init with
        - batch size
        - data path list (point to pickle file)
        """
        self.batch_size = batch_size
        self.dlist = data_path_list
        self.n_class = n_class
        self.preproc = f_preproc
        self.aug = aug_params
        
        ## init operations ##
        random.shuffle(self.dlist)
        self._check_numbers_of_elements()
    
    def get_total_steps(self):
        return self.n_steps_per_epoch
    
    def get_data(self):
        while True:
            item = self.train_queue.get()
            x, y = item
            yield x,y
    
    def start_workers(self, workers = 4):
        
        self.train_queue = queue.Queue(maxsize = 50)
        self.pickle_queue = queue.Queue(maxsize = 10) # open a pickle file and put them into queue
        
        self.events = list()
        self.threading = list()
        
        event = Event()
        thread = Thread(target = enqueue, args = (self.pickle_queue, 
                                                  event, 
                                                  self._open_pickle))
        thread.daemon = True
        thread.start()
        self.events.append(event)
        self.threading.append(thread)
        
        for i in range(workers):
            event = Event()
            thread = Thread(target = enqueue, args = (self.train_queue, 
                                                      event, 
                                                      self._get_train_data))
            thread.daemon = True 
            thread.start()
            self.events.append(event)
            self.threading.append(thread)
        
    def stop_workers(self):
        for t in self.events:
            t.set()
        for i, t in enumerate(self.threading):
            t.join(timeout = 1)
        print("All Threads were stopped")
    
    def _check_numbers_of_elements(self):
        with open(self.dlist[0], 'rb') as f:
            tmp_x, tmp_y = pickle.load(f)
        print("Image per pickle: %i" % len(tmp_x))
        self.n_steps_per_epoch = len(self.dlist) * len(tmp_x) // self.batch_size # This will be used to control "StopAtStepHook": total_batch / bz * epoch / hvd.size
        
    def _open_pickle(self):
        while True:
            for this_file in self.dlist:
                try:
                    with open(this_file, 'rb') as f:
                        obj = pickle.load(f)
                    yield obj
                except:
                    # conflict with multi-accessing, ignore now
                    pass
                
            # After loop all files, shuffle the list
            random.shuffle(self.dlist)
            
    def _get_train_data(self):
        while True:
            item = self.pickle_queue.get()
            x_, y_ = item
            num_iters = int(np.ceil(len(x_) / self.batch_size))
            for i in range(num_iters):
                
                this_x = x_[(i*self.batch_size):((i+1)*self.batch_size)]
                this_y = y_[(i*self.batch_size):((i+1)*self.batch_size)]
                
                if self.aug:
                    this_x = self.aug.augment_images(this_x)
                    
                this_x = this_x.astype(np.float32)
                
                if self.preproc:
                    this_x = self.preproc(this_x)
                    
                this_y = tf.keras.utils.to_categorical(this_y, num_classes=self.n_class)

                yield this_x, this_y
        
def enqueue(queue, stop, gen_func):
    gen = gen_func()
    while True:
        if stop.is_set():
            return
        queue.put(next(gen))
        
try:
    import imgaug as ia
    from imgaug import augmenters as iaa
except:
    print("Import Error, Please make sure you have imgaug")
    
sometimes = lambda aug: iaa.Sometimes(0.5, aug)
lesstimes = lambda aug: iaa.Sometimes(0.2, aug)
augmentation = iaa.Sequential([
    iaa.Fliplr(0.5, name="FlipLR"),
    iaa.Flipud(0.5, name="FlipUD"),
    iaa.ContrastNormalization((0.8, 1.2), name = "Contrast"),
    iaa.Add((-15, 15), per_channel = 0.5),
    iaa.OneOf([iaa.Multiply((0.8, 1.2), per_channel = 0.5, name = "Multiply"),
               iaa.AddToHueAndSaturation((-15,30),name = "Hue"),
              ]),
    sometimes(iaa.GaussianBlur((0, 1.0), name="GaussianBlur")),
    iaa.OneOf([iaa.Affine(rotate = 90),
               iaa.Affine(rotate = 180),
               iaa.Affine(rotate = 270)]),
    sometimes(iaa.Affine(
                scale = (0.6,1.2),
                mode = 'wrap'
                )),
    iaa.OneOf([iaa.AdditiveGaussianNoise(scale=0.05*255, name="Noise"),
           iaa.CoarseDropout((0.05, 0.15), size_percent=(0.01, 0.05), name = 'Cdrop')
           ]),
])

data_generator = DataGenerator(batch_size= FLAGS.batch_size, 
                               data_path_list=train_list,
                               n_class=3, 
                               f_preproc=CORRESPONDING_PREPROC[pretrain_model], aug_params=augmentation
                               )
print("Estimate steps per epoch: %i" % (data_generator.get_total_steps() ))
data_generator.start_workers(workers=4)
train_gen = data_generator.get_data()

### Get Validation data ###
x_test, y_test = [], []
for file in valid_list:
    with open(file, 'rb') as f:
        xx, yy = pickle.load(f)
    x_test.append(xx)
    y_test.append(yy)
x_test = np.concatenate(x_test)
y_test = np.concatenate(y_test)

x_test = CORRESPONDING_PREPROC[pretrain_model](x_test.astype('float32'))
y_test = tf.keras.utils.to_categorical(y_test, 3)
print("Numbers of Validation data: %i" % (len(x_test)))

#####################
#### Model struct ###
#####################
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as slimNet
import horovod.tensorflow as hvd

hvd.init()
tf.logging.set_verbosity(tf.logging.INFO)
config = tf.ConfigProto()
config.gpu_options.visible_device_list = str(hvd.local_rank())
config.gpu_options.allow_growth = True

tf.reset_default_graph()

inputs = tf.placeholder(shape = (None, 128, 128, 3), dtype=tf.float32, name = 'x_input')
y_true = tf.placeholder(shape = (None, 3), dtype = tf.float32, name = 'y_true')
is_training = tf.placeholder(dtype=tf.bool, shape=[])
global_step = tf.train.get_or_create_global_step()
#lr = tf.placeholder(tf.float32, shape = [])
lr = tf.train.exponential_decay(0.01, global_step, 10000, 0.97, staircase = True)


with slim.arg_scope(resnet_utils.resnet_arg_scope(batch_norm_decay=0.95)):
    _, layers_dict = resnet_v2_50(inputs=inputs, is_training=is_training)
exclude = []
var_list = slim.get_variables_to_restore(exclude = exclude)
var_list = [i for i in var_list if 'squeeze_and_excitation' not in i.name]

gap = layers_dict['global_pool']
if len(gap.shape) == 4:
    gap = tf.reduce_mean(gap, [1,2], name = "GAP_layer") # global_averaing pooling

with tf.variable_scope('classifier'):
    logits = tf.layers.dense(inputs=gap, units=3, name = 'logits')
    prediction = tf.nn.softmax(logits)

with tf.variable_scope('Compute_loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = y_true, name = 'loss'))
    correct_pred = tf.equal(tf.argmax(prediction, axis = 1), 
                            tf.argmax(y_true, axis = 1))
    accuracy_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
      
#optimizer = tf.train.MomentumOptimizer(lr * hvd.size(), momentum = 0.95, use_nesterov = True)
optimizer = tf.train.AdamOptimizer(lr * hvd.size())
optimizer = hvd.DistributedOptimizer(optimizer)
train_op = optimizer.minimize(loss, global_step=global_step)

###############
### Summary ###
###############
saver = tf.train.Saver() if hvd.rank() == 0 else None
tf.summary.scalar('accuracy', accuracy_op)
tf.summary.scalar('loss', loss)
tf.summary.scalar('learning_rate', lr)
merged_summary = tf.summary.merge_all()

##
################
### HVD hook ###
################
total_steps = data_generator.get_total_steps()*num_epochs // hvd.size() if not FLAGS.test_mode else data_generator.get_total_steps() // hvd.size() 
hooks = [
    hvd.BroadcastGlobalVariablesHook(0),
    # Horovod: adjust number of steps based on number of GPUs.
    tf.train.StopAtStepHook(last_step=total_steps),
    tf.train.LoggingTensorHook(tensors={'step': global_step, 'loss': loss, 'accuracy': accuracy_op},
                               every_n_iter=1000),
]
checkpoint_dir = os.path.join(FLAGS.result_dir, 'checkpoints') if hvd.rank() == 0 else None

#####################
### Training Loop ###
#####################
A = np.arange(len(x_test))
with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir,
                                       hooks=hooks,
                                       config=config,
                                       log_step_count_steps = 500, # print every 500 global steps (default = 100)
                                       save_checkpoint_secs = 1800, # save model per 0.5 hour (default = 600)
                                      ) as mon_sess:
    train_writer = tf.summary.FileWriter(os.path.join(FLAGS.result_dir, 'train'),  mon_sess.graph) if hvd.rank() == 0 else None
    test_writer = tf.summary.FileWriter(os.path.join(FLAGS.result_dir, 'test')) if hvd.rank() == 0 else None
    idx = 0
    while not mon_sess.should_stop():
        x_, y_ = next(train_gen)
        this_summary, _ = mon_sess.run([merged_summary, train_op], feed_dict = {inputs: x_,
                                                                                y_true: y_,
                                                                                is_training: True,
                                                                                #lr: 0.0001
                                                                               })
        if (hvd.rank() == 0) & (idx % 1000 == 0):
            train_writer.add_summary(this_summary, idx)
        
        """
        if (hvd.rank() == 0) & (idx % 10000 == 0):
            pick_idx = np.random.choice(A, 7200)
            x_test_sample, y_test_sample = x_test[pick_idx], y_test[pick_idx]
            
            this_summary = mon_sess.run(merged_summary, feed_dict = {inputs: x_test_sample,
                                                                     y_true: y_test_sample,
                                                                     is_training: False})
            test_writer.add_summary(this_summary, idx)
        """
        idx += 1
        
data_generator.stop_workers()
