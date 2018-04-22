import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
from random import shuffle
import csv
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider
import tf_util

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_cls', help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--log_dir', default='log/log_field', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=250, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=64, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
FLAGS = parser.parse_args()

NUM_PROBE = 2048
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

MAX_NUM_POINT = 2048
NUM_CLASSES = 40

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()
d_dir = open(os.getenv("HOME")+'/Data_dir', 'r')
data_dir = d_dir.read()[:-1]
d_dir.close()
num_demo = 0

# ModelNet40 official train/test split
#TRAIN_FILES = provider.getDataFiles( \
#    os.path.join(BASE_DIR, 'data/h5/train_files.txt'))
#TEST_FILES = provider.getDataFiles(\
#    os.path.join(BASE_DIR, 'data/h5/test_files.txt'))

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def half(shape):
    norm = np.random.rand(3, 1) - 0.5*np.ones(shape=(3, 1))
    norm = norm/abs(norm)
    shape_distance = np.matmul(shape, norm)
    index = np.argsort(shape_distance, axis=0)
    output = np.empty(shape=(1024, 3), dtype=np.float32)
    ind = 0
    for i in index:
        output[ind, : ] = shape[i, :]
        ind += 1
        if ind == 1023:
            break
    return output


def provide_data(is_train):
    if is_train:
        file = open(data_dir+'/03001627train.txt', 'r')
    else:
        file = open(data_dir+'/03001627val.txt', 'r')
    lines = file.read().split('\n')
    shuffle(lines)
    file.close()

    output_pointcloud = np.zeros(shape=(BATCH_SIZE, NUM_POINT, 3), dtype=np.float32)
    output_probepoint = np.zeros(shape=(BATCH_SIZE, NUM_PROBE, 3), dtype=np.float32)
    output_label = np.zeros(shape=(BATCH_SIZE, NUM_PROBE), dtype=np.int32)
    output_weight = np.zeros(shape=(BATCH_SIZE, 1024), dtype=np.float32)
    filled = 0
    for id in lines:
        try:
            h5f = h5py.File(data_dir+'/03001627/'+id+"/model1points.h5", 'r')
            h5f2 = h5py.File(data_dir+'/03001627/'+id+"/model1pointsweight.h5", 'r')
        except:
            continue
        else:
            #output_pointcloud[filled, :, :] = half(h5f['points_on'][:])
            output_pointcloud[filled, :, :] = h5f['points_on'][0:NUM_POINT, :]
            output_probepoint[filled, 0:int(3*NUM_PROBE/4), :] = h5f['points_in_out'][0:int(3*NUM_PROBE/4), :]
            output_probepoint[filled, int(3*NUM_PROBE/4):, :] = h5f['points_in_out'][6144:int(6144+NUM_PROBE/4), :]
            output_weight[filled, :] = h5f2['weight'][:]
            tmp1 = np.zeros_like(output_label[filled, 0:int(3*NUM_PROBE/4)])
            tmp2 = np.zeros_like(output_label[filled, int(3*NUM_PROBE/4):])
            tmp1[h5f['in_out_lable'][:int(3*NUM_PROBE/4)]] = 1
            tmp2[h5f['in_out_lable'][6144:int(6144+NUM_PROBE/4)]] = 1
            #tmp = np.less(output_probepoint[filled, :, 0], np.zeros_like(output_probepoint[filled, :, 0]))
            output_label[filled, :] = np.concatenate((tmp1.astype(int), tmp2.astype(int)), axis=0)

            filled += 1
            h5f.close()
            h5f2.close()
            if filled == BATCH_SIZE:
                filled = 0
                yield output_pointcloud, output_probepoint, output_label, output_weight



######################################### main #############################################
######################################### main #############################################
######################################### main #############################################
######################################### main #############################################
######################################### main #############################################
def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, probe_points_pl, labels_pl, elm_weight = MODEL.placeholder_inputs_field(BATCH_SIZE, NUM_POINT, NUM_PROBE)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl)

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss
            h5f = h5py.File(data_dir + "/03001627/random_weight.h5", 'r')
            random_weights = h5f['randomweight'][:]
            h5f.close()
            net1 = tf.constant(random_weights, dtype=tf.float32)
            #net1 = tf.constant(np.random.normal(size=(3, 3072)), dtype=tf.float32)
            pred, end_points, G_features, pred_elm_weight = MODEL.get_model_field(pointclouds_pl, probe_points_pl, is_training_pl, net1, bn_decay=bn_decay)
            #loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=labels_pl)
            pred = tf.squeeze(pred, axis=2)
            loss1 = tf.reduce_mean(tf.losses.mean_squared_error(labels=labels_pl, predictions=pred))
            loss2 = tf.reduce_mean(tf.losses.mean_squared_error(labels=elm_weight, predictions=pred_elm_weight))
            rate = 0
            loss = loss1 + rate * loss2# - 0.06*tf.reduce_mean(pred)
            tf.summary.scalar('loss', loss)
            loss_rate = tf.divide(loss1, rate * loss2)

            #correct = tf.equal(tf.argmax(pred, 2), tf.to_int64(labels_pl))
            correct = tf.equal(tf.cast(tf.greater(pred, tf.constant(0.5*np.ones(shape=(BATCH_SIZE, NUM_PROBE)), dtype=np.float32)), tf.int32), labels_pl)
            ones = tf.reduce_sum(tf.cast(tf.greater(pred, tf.constant(0.5 * np.ones(shape=(BATCH_SIZE, NUM_PROBE)), dtype=np.float32)), tf.int32))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE) / float(NUM_PROBE)
            tf.summary.scalar('accuracy', accuracy)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()


        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        # merged = tf.merge_all_summaries()
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                             sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init, {is_training_pl: True})

        ops = {'pointclouds_pl': pointclouds_pl,
               'probe_points_pl': probe_points_pl,
               'labels_pl': labels_pl,
               'elm_weight_pl': elm_weight,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'accuracy': accuracy,
               'loss_rate': loss_rate,
               'ones': ones}

        builder = tf.saved_model.builder.SavedModelBuilder(LOG_DIR + '/model_elm_incomplete')
        builder.add_meta_graph_and_variables(sess, 'feature_net')

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            train_one_epoch(sess, ops, train_writer)
            eval_accu = eval_one_epoch(sess, ops, test_writer)

            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)
            if eval_accu > 0.98:
                builder.save()



def train_one_epoch(sess, ops, train_writer):
    is_training = True
    log_string('train_one_epoch*****************************************')
    train_generator = provide_data(is_training)
    loss_sum = 0
    acc_sum = 0
    loss_rate_sum = 0
    num = 0
    for data in train_generator:
        num += 1
        feed_dict = {ops['pointclouds_pl']: data[0],
                     ops['probe_points_pl']: data[1],
                     ops['labels_pl']: data[2],
                     ops['elm_weight_pl']: data[3],
                     ops['is_training_pl']: is_training,}
        summary, step, _, loss, acc, pred, loss_rate = sess.run([ops['merged'],
                                                                 ops['step'],
                                                                 ops['train_op'],
                                                                 ops['loss'],
                                                                 ops['accuracy'],
                                                                 ops['pred'],
                                                                 ops['loss_rate']],
                                                                feed_dict = feed_dict)
        loss_rate_sum += loss_rate
        acc_sum += acc
        loss_sum += loss
        train_writer.add_summary(summary, step)

    log_string('train loss: %f' % (loss_sum/num))
    log_string('loss_rate: %f' % (loss_rate_sum/num))
    log_string('accuracy: %f' % (acc_sum/num))



def eval_one_epoch(sess, ops, train_writer):
    is_training = False
    log_string('evaluate_one_epoch')
    test_generator = provide_data(is_training)
    loss_sum = 0
    acc_sum = 0
    loss_rate_sum = 0
    ones_sum = 0
    num = 0
    for data in test_generator:
        num += 1
        feed_dict = {ops['pointclouds_pl']: data[0],
                     ops['probe_points_pl']: data[1],
                     ops['labels_pl']: data[2],
                     ops['elm_weight_pl']: data[3],
                     ops['is_training_pl']: is_training, }
        summary, step, _, loss, acc, pred, loss_rate, ones = sess.run([ops['merged'],
                                                                       ops['step'],
                                                                       ops['train_op'],
                                                                       ops['loss'],
                                                                       ops['accuracy'],
                                                                       ops['pred'],
                                                                       ops['loss_rate'],
                                                                       ops['ones']],
                                                                      feed_dict = feed_dict)
        loss_rate_sum += loss_rate
        acc_sum += acc
        loss_sum += loss
        ones_sum += ones/NUM_PROBE/BATCH_SIZE
        train_writer.add_summary(summary, step)
        if np.random.rand(1) < 0.01:
            #    print(np.mean(np.mean(np.argmax(pred, axis=2))))
            h5f = h5py.File(os.path.join(LOG_DIR, "Demo.h5"), 'w')
            h5f.create_dataset('points_on', data=data[0])
            h5f.create_dataset('points_in_out', data=data[1])
            h5f.create_dataset('points_label', data=data[2])
            h5f.create_dataset('points_pred', data=pred)
            h5f.close()

    log_string('eval loss: %f' % (loss_sum/num))
    log_string('loss_rate: %f' % (loss_rate_sum/num))
    log_string('eval accuracy: %f' % (acc_sum / num))
    log_string('ones_rate: %f' % (ones_sum / num))

    return acc_sum / num




















if __name__ == "__main__":
    train()
    LOG_FOUT.close()