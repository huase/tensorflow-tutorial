import os, datetime
import numpy as np
import tensorflow as tf
from DataLoader import *
import heapq
import itertools

import inception_resnet_v2

# Dataset Parameters
batch_size = 80#200
load_size = 256
fine_size = 224
c = 3
data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])

# Training Parameters
learning_rate = 0.0003#0.001
dropout = 0.5 # Dropout, probability to keep units
training_iters = 100000
step_display = 50
step_save = 10000
#path_save = 'alexnet'
start_from = 'inception-90000'


# Construct dataloader
opt_data_train = {
    'data_h5': 'miniplaces_256_train.h5',
    'data_root': 'images/',   # MODIFY PATH ACCORDINGLY
    'data_list': 'development_kit/data/train.txt', # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': True
    }
opt_data_val = {
    'data_h5': 'miniplaces_256_val.h5',
    'data_root': 'images/',   # MODIFY PATH ACCORDINGLY
    'data_list': 'development_kit/data/val.txt',   # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': False
    }

opt_data_test = {
    'data_h5': 'miniplaces_256_test.h5',
    'data_root': 'images/',   # MODIFY PATH ACCORDINGLY
    'data_list': 'development_kit/data/test_new.txt',   # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': False
    }

# loader_train = DataLoaderDisk(**opt_data_train)
# loader_val = DataLoaderDisk(**opt_data_val)

#loader_train = DataLoaderH5(**opt_data_train)
#loader_val = DataLoaderH5(**opt_data_val)
#loader_test = DataLoaderH5(**opt_data_test)

datatype = "test"
opt_data = opt_data_test
loader = DataLoaderH5(**opt_data)


# tf Graph input
x = tf.placeholder(tf.float32, [None, fine_size, fine_size, c])
y = tf.placeholder(tf.int64, None)
keep_dropout = tf.placeholder(tf.float32)

# Construct model
logits,end_points = inception_resnet_v2.inception_resnet_v2(x, num_classes=100,dropout_keep_prob=keep_dropout)

# Define loss and optimizer
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y))
train_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Evaluate model
accuracy1 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, y, 1), tf.float32))
accuracy5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, y, 5), tf.float32))

# define initialization
init = tf.initialize_all_variables()

# define saver
saver = tf.train.Saver()

# define summary writer
#writer = tf.train.SummaryWriter('.', graph=tf.get_default_graph())

# Launch the graph

config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )

def make_filename(n):
    n=str(int(n))
    return datatype+"/"+"".join(["0"]*(8-len(n)))+n+".jpg"

with tf.Session(config=config) as sess:
    # Initialization
    if len(start_from)>1:
        saver.restore(sess, start_from)
    else:
        sess.run(init)
    
    step = 0

    # Evaluate
    print 'Evaluation...'
    num_batch = loader.size()/batch_size
    loader.reset()
    
    with open(datatype+"_results_"+start_from,"w") as f:
        for i in range(num_batch):
       	    images_batch, labels_batch = loader.next_batch(batch_size)    
            ypred = sess.run(logits, feed_dict={x: images_batch, keep_dropout: 1.})
            for xp,yp in zip(labels_batch, ypred):
                f.write(make_filename(xp)+" "+" ".join([str(k[1]) for k in heapq.nlargest(5,zip(yp,itertools.count()))])+"\n")
            print "batch",i,"of",num_batch,"done"
