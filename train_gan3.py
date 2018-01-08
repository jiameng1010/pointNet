import tensorflow as tf
tfgan = tf.contrib.gan
layers = tf.contrib.layers
import numpy as np
import argparse
import provider
import importlib
import h5py
import os
import sys
from tensorflow.python.ops import variable_scope, embedding_ops
from tensorflow.contrib.gan.python.losses.python import tuple_losses_impl as tfgan_losses
from tensorflow.contrib.data import Dataset, Iterator

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import tf_util
from transform_nets import input_transform_net_no_bn, feature_transform_net_no_bn

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_cls', help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=250, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.00001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
FLAGS = parser.parse_args()
FLAGS.noise_dims = 32
FLAGS.max_number_of_steps = 120
FLAGS.embeding_dim = 32

MAX_NUM_POINT = 2048
NUM_CLASSES = 40
DECAY_STEP = FLAGS.decay_step
BATCH_SIZE = FLAGS.batch_size
BASE_LEARNING_RATE = FLAGS.learning_rate
NUM_POINT = FLAGS.num_point
BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

MODEL = importlib.import_module(FLAGS.model)

LOG_DIR = './log/gan_log_12'
LOG_FOUT = open(os.path.join('./log/gan_log_12', 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rateG(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.0005) # CLIP THE LEARNING RATE!
    return learning_rate

def get_learning_rateD(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate

def provide_data(sess2):
    BATCH_SIZE = FLAGS.batch_size
    current_data, current_label = provider.loadDataFile('./data/h5/traincompleteall.h5')
    current_data, current_label, _ = provider.shuffle_data(current_data, np.squeeze(current_label))
    current_label = np.squeeze(current_label.astype(int))

    partial_data, partial_label = provider.loadDataFile('./data/h5/trainall.h5')
    partial_data, partial_label, _ = provider.shuffle_data(partial_data, np.squeeze(partial_label))
    partial_label = np.squeeze(partial_label.astype(int))


    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE

    pointclouds_pl = sess2.graph.get_tensor_by_name('Placeholder:0')
    labels_pl = sess2.graph.get_tensor_by_name('Placeholder_1:0')
    is_train_pl = sess2.graph.get_tensor_by_name('Placeholder_2:0')

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE

        # mantipulation data
        rotated_data = provider.rotate_point_cloud(current_data[start_idx:end_idx, :, :])
        jittered_data = provider.jitter_point_cloud(rotated_data)
        # mantipulate labe
        one_hot_labe1 = np.zeros((BATCH_SIZE, 41))
        one_hot_labe1[np.arange(BATCH_SIZE), current_label[start_idx:end_idx]] = 1
        one_hot_labe2 = np.zeros((BATCH_SIZE, 41))
        one_hot_labe2[np.arange(BATCH_SIZE), partial_label[start_idx:end_idx]] = 1

        is_train = False
        feed_dict = {pointclouds_pl: provider.rotate_point_cloud(partial_data[start_idx:end_idx, :, :]),
                     labels_pl: partial_label[start_idx:end_idx],
                     is_train_pl: is_train}
        G_features = sess2.run(sess2.graph.get_tensor_by_name('maxpool/maxpool:0'), feed_dict=feed_dict)
        #out['data'] = jittered_data
        #out['labe'] = one_hot_labe
        yield jittered_data, one_hot_labe1, current_label[start_idx:end_idx], np.squeeze(G_features), one_hot_labe2, partial_label[start_idx:end_idx]

def get_model(point_cloud, embedded_label, is_training, bn_decay=None,):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}

    with tf.variable_scope('transform_net1', reuse=tf.AUTO_REUSE) as sc:
        transform = input_transform_net_no_bn(point_cloud, is_training, bn_decay, K=3)

    point_cloud_transformed = tf.matmul(point_cloud, transform)
    input_image = tf.expand_dims(point_cloud_transformed, -1)

    net = tf_util.conv2d(input_image, 64, [1,3],
                         padding='VALID', stride=[1,1],
                         bn=False, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay, activation_fn=tf.nn.leaky_relu)
    net = tf_util.conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=False, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay, activation_fn=tf.nn.leaky_relu)

    with tf.variable_scope('transform_net2', reuse=tf.AUTO_REUSE) as sc:
        transform = feature_transform_net_no_bn(net, is_training, bn_decay, K=64)

    end_points['transform'] = transform
    net_transformed = tf.matmul(tf.squeeze(net, axis=[2]), transform)
    net_transformed = tf.expand_dims(net_transformed, [2])

    net = tf_util.conv2d(net_transformed, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=False, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay, activation_fn=tf.nn.leaky_relu)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=False, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay, activation_fn=tf.nn.leaky_relu)
    net = tf_util.conv2d(net, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=False, is_training=is_training,
                         scope='conv5', bn_decay=bn_decay, activation_fn=tf.nn.leaky_relu)

    # Symmetric function: max pooling
    net = tf_util.max_pool2d(net, [num_point,1],
                             padding='VALID', scope='maxpool')

    net = tf.reshape(net, [batch_size, -1])
    #with tf.variable_scope('embeding_condition', reuse=tf.AUTO_REUSE):
    #    net = tfgan.features.condition_tensor(net, embedded_label)####################
    net = tf_util.fully_connected(net, 512, bn=False, is_training=is_training,
                                  scope='fc1', bn_decay=bn_decay, activation_fn=tf.nn.leaky_relu)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp1')
    net = tf_util.fully_connected(net, 256, bn=False, is_training=is_training,
                                  scope='fc2', bn_decay=bn_decay, activation_fn=tf.nn.leaky_relu)
    net_branch = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp2')
    net = tf_util.fully_connected(net_branch, 41, activation_fn=None, scope='fc3')

    return net, end_points

def conditional_discriminator(point_clouds, embedded_label):
    is_training_pl = tf.constant([True])
    # Get model and loss
    #with tf.variable_scope('Discriminator', reuse=tf.AUTO_REUSE):
    return get_model(point_clouds, embedded_label, tf.squeeze(is_training_pl))


def generate_cloud(feature, noise):
    feature = tf.concat([feature, feature], axis=1)#2
    feature = tf.concat([feature, feature], axis=1)#4
    feature = tf.concat([feature, feature], axis=1)#8
    feature = tf.concat([feature, feature], axis=1)#16
    feature = tf.concat([feature, feature], axis=1)#32
    feature = tf.concat([feature, feature], axis=1)#64
    feature = tf.concat([feature, feature], axis=1)#128
    feature = tf.concat([feature, feature], axis=1)#256
    feature = tf.concat([feature, feature], axis=1)#512
    feature = tf.concat([feature, feature], axis=1)#1024

    #noise = tf.concat([noise, noise], axis=1)#2
    #noise = tf.concat([noise, noise], axis=1)#4
    #noise = tf.concat([noise, noise], axis=1)#8
    #noise = tf.concat([noise, noise], axis=1)#16
    #noise = tf.concat([noise, noise], axis=1)#32
    #noise = tf.concat([noise, noise], axis=1)#64
    #noise = tf.concat([noise, noise], axis=1)#128
    #noise = tf.concat([noise, noise], axis=1)#256
    #noise = tf.concat([noise, noise], axis=1)#512
    #noise = tf.concat([noise, noise], axis=1)#1024

    feature = tf.concat([feature, noise], axis=2)
    point = layers.fully_connected(feature, 256, activation_fn=tf.nn.leaky_relu)
    point = layers.dropout(point, keep_prob=0.8)
    point = layers.fully_connected(point, 128, activation_fn=tf.nn.leaky_relu)
    point = layers.dropout(point, keep_prob=0.8)
    point = layers.fully_connected(point, 32, activation_fn=tf.nn.leaky_relu)
    point = layers.fully_connected(point, 16, activation_fn=tf.nn.leaky_relu)
    point = layers.fully_connected(point, 3, activation_fn=tf.nn.tanh)

    return point


def conditional_generator(inputs):
    noise, cloud_labels, partial_feature = inputs
    #with tf.variable_scope('Generator', reuse=tf.AUTO_REUSE):
    with tf.contrib.framework.arg_scope(
            [layers.fully_connected, layers.conv2d_transpose],
            activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm,
            weights_regularizer=layers.l2_regularizer(2.5e-5)):

        with tf.variable_scope('conditioning1'):
            net = tfgan.features.condition_tensor(noise, cloud_labels)
        net = layers.fully_connected(noise, 64, activation_fn=tf.nn.leaky_relu)
        net = layers.fully_connected(net, 128, activation_fn=tf.nn.leaky_relu)
        #with tf.variable_scope('conditioning2'):
        #    net = tfgan.features.condition_tensor(net, cloud_labels)
        net = layers.fully_connected(net, 256, activation_fn=tf.nn.leaky_relu)
        #with tf.variable_scope('conditioning3'):
        #    net = tfgan.features.condition_tensor(net, cloud_labels)
        net = layers.fully_connected(net, 512, activation_fn=tf.nn.leaky_relu)
        net = tfgan.features.condition_tensor(net, partial_feature)
        feature = layers.fully_connected(net, 1024)

    noise2 = tf.random_normal([32, 1024, 16])
    #noise2 = tf.to_float(tf.constant(np.zeros([32, 1024, 128])))
    cloud = generate_cloud(tf.expand_dims(feature, axis=1), noise2)

    return cloud

def density_penalty_for_one(G_output):
    shuffled = tf.random_shuffle(G_output)
    stoped_shuffled = tf.stop_gradient(shuffled)
    return tf.norm(stoped_shuffled-G_output)

def density_penalty(G_output):
    #with tf.variable_scope('NO_TRAINING'):
    loss = tf.constant([0.0])
    for i in range(BATCH_SIZE):
        loss += density_penalty_for_one(G_output[i, :, :])
    return loss


######################################### main #############################################
######################################### main #############################################
######################################### main #############################################
######################################### main #############################################
######################################### main #############################################
def train():
    ## global steps
    stepsG = tf.Variable(0)
    stepsD = tf.Variable(0)
    global_step = stepsD + stepsG

    ## setup input data
    partial_featureG = tf.placeholder(dtype=tf.float32, shape=(BATCH_SIZE, 1024))
    onehot_labelsG = tf.placeholder(dtype=tf.float32, shape=(BATCH_SIZE, 41))
    cloud_labelsG = tf.placeholder(dtype=tf.int32, shape=(BATCH_SIZE))
    #
    point_cloudsD = tf.placeholder(dtype=tf.float32, shape=(BATCH_SIZE, 1024, 3))
    onehot_labelsD = tf.placeholder(dtype=tf.float32, shape=(BATCH_SIZE, 41))
    cloud_labelsD = tf.placeholder(dtype=tf.int32, shape=(BATCH_SIZE))
    #
    noise = tf.random_normal((FLAGS.batch_size, FLAGS.noise_dims))
    gt_zeros = tf.constant(np.zeros((32)), dtype=tf.int32)
    gt_forty = tf.constant(40*np.ones((32)), dtype=tf.int32)

    ## setup models
    incomplete_features = tf.Graph()
    #with tf.variable_scope('Embedding', reuse=tf.AUTO_REUSE):
    #    embedding = variable_scope.get_variable('embedding', [40, FLAGS.embeding_dim])
    #    embedded_label_D = embedding_ops.embedding_lookup(embedding, cloud_labelsD)
    #    embedded_label_G = embedding_ops.embedding_lookup(embedding, cloud_labelsG)
    with tf.variable_scope('Generator', reuse=tf.AUTO_REUSE):
        embeddingG = variable_scope.get_variable('embedding', [40, FLAGS.embeding_dim])
        embedded_label_G = embedding_ops.embedding_lookup(embeddingG, cloud_labelsG)
        G_input = noise, embedded_label_G, partial_featureG
        G_output = conditional_generator(G_input)
    with tf.variable_scope('Discriminator') as sc:
        embeddingD = variable_scope.get_variable('embedding', [40, FLAGS.embeding_dim])
        embedded_label_D = embedding_ops.embedding_lookup(embeddingD, cloud_labelsD)
        D_output_trainG = conditional_discriminator(G_output, embedded_label_D)
        #D_input1_trainD = tf.concat([point_cloudsD, G_output], axis=0)
        #D_input2_trainD = tf.concat([cloud_labelsD, cloud_labelsG], axis=0)
        sc.reuse_variables()
        D_output_trainD = conditional_discriminator(point_cloudsD, embedded_label_D)


    ## setup loss
    lossD1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=D_output_trainG[0], labels=gt_forty))
    lossD2 = MODEL.get_loss(D_output_trainD[0], cloud_labelsD, D_output_trainD[1])
    lossD = lossD1/16 + lossD2
    lossG1 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=D_output_trainG[0], labels=cloud_labelsG)
    lossG2 = density_penalty(G_output)
    lossG = tf.reduce_mean(lossG1) # - tf.reduce_mean(tf.reduce_mean((1e-5)*lossG3))
    tf.summary.scalar('lossD', lossD)
    tf.summary.scalar('lossG', lossG)

    correct_trainD = tf.less(tf.argmax(D_output_trainD[0], 1), tf.to_int64(gt_forty))
    correct_trainG = tf.equal(tf.argmax(D_output_trainG[0], 1), tf.to_int64(gt_forty))
    accuracy_classification_trainD = (tf.reduce_sum(tf.cast((correct_trainD), tf.float32))) / float(BATCH_SIZE)
    accuracy_classification_trainG = tf.reduce_sum(tf.cast(correct_trainG, tf.float32)) / float(BATCH_SIZE)
    forty = tf.constant(40 * np.ones(shape=(BATCH_SIZE), dtype=int))
    tf.summary.scalar('accuracy_classification_trainD', accuracy_classification_trainD)
    tf.summary.scalar('accuracy_classification_trainG', accuracy_classification_trainG)
    correct_label_trainD = tf.equal(tf.argmax(D_output_trainD[0], 1), tf.to_int64(cloud_labelsD))
    accuracy_label_trainD = tf.reduce_sum(tf.cast((correct_label_trainD), tf.float32)) / float(BATCH_SIZE)
    correct_label_trainG = tf.equal(tf.argmax(D_output_trainG[0], 1), tf.to_int64(cloud_labelsG))
    accuracy_label_trainG = tf.reduce_sum(tf.cast((correct_label_trainG), tf.float32)) / float(BATCH_SIZE)
    tf.summary.scalar('accuracy_label_trainD', accuracy_label_trainD)
    tf.summary.scalar('accuracy_label_trainG', accuracy_label_trainG)

    ## setup optimizor
    learning_rateG = get_learning_rateG(stepsG)
    learning_rateD = get_learning_rateD(stepsD)
    #E_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Embedding')
    optimizerG = tf.train.AdamOptimizer(learning_rateG)
    G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Generator')
    train_opG = optimizerG.minimize(lossG, global_step=stepsG, var_list=[G_vars])
    optimizerD = tf.train.AdamOptimizer(learning_rateD)
    D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Discriminator')
    train_opD = optimizerD.minimize(lossD, global_step=stepsD, var_list=[D_vars])

    with tf.device('/gpu:0'):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess2 = tf.Session(graph=incomplete_features, config=config)
        tf.saved_model.loader.load(sess2, 'feature_net', './log/model')
    with tf.Session() as sess:
        ## summary writer
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        ops = {'partial_featureG': partial_featureG,
                'predG': G_output,
                'predD': D_output_trainD,
                'predDG': D_output_trainG,
                'lossG': lossG,
                'lossD': lossD,
                'train_opG': train_opG,
                'train_opD': train_opD,
                'merged': merged,
                'stepG': stepsG,
                'stepD': stepsD,
                'global_step': global_step,
                'cloud_labelsG': cloud_labelsG,
                'point_cloudsD': point_cloudsD,
                'cloud_labelsD': cloud_labelsD,
               'accuracy_classification_trainD': accuracy_classification_trainD,
               'accuracy_classification_trainG': accuracy_classification_trainG,
               'accuracy_label_trainD': accuracy_label_trainD,
               'accuracy_label_trainG': accuracy_label_trainG}
        # initialize the iterator and variable on the training data
        init = tf.global_variables_initializer()
        sess.run(init)

        for epoch in range(501):
            log_string('******************************* EPOCH %03d ******************************' % (epoch))
            #trainD(sess, sess2, ops, train_writer)
            #trainG(sess, sess2, ops, train_writer)
            #repeat = 0
            #while (True):
            #    repeat += 1
            #    acc = trainG(sess, sess2, ops, train_writer)
            #    if (acc > 0.5) or (repeat == 6):
            #        break
            if epoch == 0:
                for i in range(10):
                    trainD(sess, sess2, ops, train_writer)
                else:
                    train_joint(sess, sess2, ops, train_writer)

            if epoch % 100 == 0:
                builder = tf.saved_model.builder.SavedModelBuilder(LOG_DIR + '/model_in_epoch_' + str(epoch))
                builder.add_meta_graph_and_variables(sess, 'GAN')
                builder.save()
                del builder
        print('Done!')

def train_joint(sess, sess2, ops, train_writer):
    log_string('train JJJJJJJJJJJJJJJJJJJJJJJJJJJJJ')
    generator = provide_data(sess2)
    loss_sumG = 0
    loss_sumD = 0
    AcDsum = 0
    AcGsum = 0
    AlDsum = 0
    AlGsum = 0
    num = 0
    for data in generator:
        num += 1
        feed_dict = {ops['cloud_labelsG']: data[5],
                     ops['cloud_labelsD']: data[2],
                     ops['point_cloudsD']: data[0],
                     ops['partial_featureG']: data[3]}
        if num%2 == 0:
            summary, step, _, lossG, lossD, pred_val, AcD, AcG, AlD, AlG = sess.run([ops['merged'], ops['global_step'],
                                                                                    ops['train_opG'], ops['lossG'],
                                                                                    ops['lossD'], ops['predG'],
                                                                                    ops['accuracy_classification_trainD'],
                                                                                    ops['accuracy_classification_trainG'],
                                                                                    ops['accuracy_label_trainD'],
                                                                                    ops['accuracy_label_trainG']],
                                                                                    feed_dict=feed_dict)
        else:
            summary, step, _, lossG, lossD, pred_val, AcD, AcG, AlD, AlG = sess.run([ops['merged'], ops['global_step'],
                                                                                    ops['train_opD'], ops['lossG'],
                                                                                    ops['lossD'], ops['predG'],
                                                                                    ops['accuracy_classification_trainD'],
                                                                                    ops['accuracy_classification_trainG'],
                                                                                    ops['accuracy_label_trainD'],
                                                                                    ops['accuracy_label_trainG']],
                                                                                    feed_dict=feed_dict)
        AcDsum += AcD
        AcGsum += AcG
        AlDsum += AlD
        AlGsum += AlG
        train_writer.add_summary(summary, step)
        loss_sumG += lossG
        loss_sumD += lossD
        if np.random.rand() <= 0.001:
            h5r = h5py.File((LOG_DIR + '/demo/demo' + str(step).zfill(8) + '.h5'), 'w')
            h5r.create_dataset('data', data=pred_val)
            h5r.create_dataset('label', data=data[5])
            h5r.close()
    log_string('total lossG: %f' % loss_sumG)
    log_string('total lossD: %f' % loss_sumD)
    log_string('accuracy_label_trainD: %f' % (AlDsum/num))
    log_string('accuracy_label_trainG: %f' % (AlGsum/num))
    log_string('accuracy_classification_trainD: %f' % (AcDsum/num))
    log_string('accuracy_classification_trainG: %f' % (AcGsum/num))
    return (AcGsum/num)

def trainG(sess, sess2, ops, train_writer):
    log_string('train GGGGGGGGGGGGGGGGGGGGG')
    generator = provide_data(sess2)
    loss_sumG = 0
    loss_sumD = 0
    AcDsum = 0
    AcGsum = 0
    AlDsum = 0
    AlGsum = 0
    num = 0
    for data in generator:
        num += 1
        feed_dict = {ops['cloud_labelsG']: data[5],
                     ops['cloud_labelsD']: data[2],
                     ops['point_cloudsD']: data[0],
                     ops['partial_featureG']: data[3]}
        summary, step, _, lossG, lossD, pred_val, AcD, AcG, AlD, AlG = sess.run([ops['merged'], ops['global_step'],
                                                                                 ops['train_opG'], ops['lossG'],
                                                                                 ops['lossD'], ops['predG'],
                                                                                 ops['accuracy_classification_trainD'],
                                                                                 ops['accuracy_classification_trainG'],
                                                                                 ops['accuracy_label_trainD'],
                                                                                 ops['accuracy_label_trainG']],
                                                                                feed_dict=feed_dict)
        AcDsum += AcD
        AcGsum += AcG
        AlDsum += AlD
        AlGsum += AlG
        train_writer.add_summary(summary, step)
        loss_sumG += lossG
        loss_sumD += lossD
        if np.random.rand() <= 0.001:
            h5r = h5py.File((LOG_DIR + '/demo/demo' + str(step).zfill(8) + '.h5'), 'w')
            h5r.create_dataset('data', data=pred_val)
            h5r.create_dataset('label', data=data[5])
            h5r.close()
    log_string('total lossG: %f' % loss_sumG)
    log_string('total lossD: %f' % loss_sumD)
    log_string('accuracy_label_trainD: %f' % (AlDsum/num))
    log_string('accuracy_label_trainG: %f' % (AlGsum/num))
    log_string('accuracy_classification_trainD: %f' % (AcDsum/num))
    log_string('accuracy_classification_trainG: %f' % (AcGsum/num))
    return (AcGsum/num)


def trainD(sess, sess2, ops, train_writer):
    log_string('train DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD')
    generator = provide_data(sess2)
    loss_sumG = 0
    loss_sumD = 0
    AcDsum = 0
    AcGsum = 0
    AlDsum = 0
    AlGsum = 0
    num = 0
    for data in generator:
        num += 1
        feed_dict = {ops['cloud_labelsG']: data[5],
                     ops['cloud_labelsD']: data[2],
                     ops['point_cloudsD']: data[0],
                     ops['partial_featureG']: data[3]}
        summary, step, _, lossG, lossD, pred_val, AcD, AcG, AlD, AlG = sess.run([ops['merged'], ops['global_step'],
                                                                                 ops['train_opD'], ops['lossG'],
                                                                                 ops['lossD'], ops['predG'],
                                                                                 ops['accuracy_classification_trainD'],
                                                                                 ops['accuracy_classification_trainG'],
                                                                                 ops['accuracy_label_trainD'],
                                                                                 ops['accuracy_label_trainG']],
                                                                                feed_dict=feed_dict)
        AcDsum += AcD
        AcGsum += AcG
        AlDsum += AlD
        AlGsum += AlG
        train_writer.add_summary(summary, step)
        loss_sumG += lossG
        loss_sumD += lossD
        if np.random.rand() <= 0.001:
            h5r = h5py.File((LOG_DIR + '/demo/demo' + str(step).zfill(8) + '.h5'), 'w')
            h5r.create_dataset('data', data=pred_val)
            h5r.create_dataset('label', data=data[5])
            h5r.close()
    log_string('total lossG: %f' % loss_sumG)
    log_string('total lossD: %f' % loss_sumD)
    log_string('accuracy_label_trainD: %f' % (AlDsum/num))
    log_string('accuracy_label_trainG: %f' % (AlGsum/num))
    log_string('accuracy_classification_trainD: %f' % (AcDsum/num))
    log_string('accuracy_classification_trainG: %f' % (AcGsum/num))
    return (AcGsum/num)



def trainD_bound(sess, sess2, ops, train_writer):
    log_string('train DDDDDDDDDDDDDDDD')
    generator = provide_data(sess2)
    loss_sumG = 0
    loss_sumD = 0
    AcDsum = 0
    AcGsum = 0
    AgDsum = 0
    AgGsum = 0
    num = 0
    for data in generator:
        num += 1
        feed_dict = {ops['labels_plG']: data[5],
                     ops['labels_plD']: np.concatenate((data[2], 40*np.ones(shape=(BATCH_SIZE), dtype=float)), axis=0),
                     ops['cloud_labelsG']: data[4],
                     ops['cloud_labelsD']: data[1],
                     ops['point_cloudsD']: data[0],
                     ops['partial_featureG']: data[3]}
        summary, step, _, lossG, lossD, pred_val, AcD, AcG, AgD, AgG = sess.run([ops['merged'], ops['stepD'],
                                                                                 ops['train_opD'], ops['lossG'],
                                                                                 ops['lossD'], ops['predD'],
                                                                                 ops['accuracy_classification_trainD'],
                                                                                 ops['accuracy_classification_trainG'],
                                                                                 ops['accuracy_gan_trainD'],
                                                                                 ops['accuracy_gan_trainG']],
                                                                                feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        loss_sumG += lossG
        loss_sumD += lossD
        AcDsum += AcD
        AcGsum += AcG
        AgDsum += AgD
        AgGsum += AgG
        if (AgGsum/num) > 0.9:
            break
    log_string('total lossG: %f' % loss_sumG)
    log_string('total lossD: %f' % loss_sumD)
    log_string('accuracy_classification_trainD: %f' % (AcDsum/num))
    log_string('accuracy_classification_trainG: %f' % (AcGsum/num))
    log_string('accuracy_gan_trainD: %f' % (AgDsum/num))
    log_string('accuracy_gan_trainG: %f' % (AgGsum/num))


def tester(sess, sess2, ops, train_writer):
    log_string('tester')
    generator = provide_data(sess2)
    for data in generator:
        normal_noise = np.random.randn(FLAGS.batch_size, FLAGS.noise_dims)
        feed_dict = {ops['labels_plG']: data[2],
                     ops['labels_plD']: np.concatenate((data[2], 40 * np.ones(shape=(BATCH_SIZE), dtype=float)), axis=0),
                     ops['cloud_labelsG']: data[1],
                     ops['cloud_labelsD']: data[1],
                     ops['point_cloudsD']: data[0],
                     ops['noise']: normal_noise}
        summary, step, lossG, lossD, pred_val, predDG, predD = sess.run([ops['merged'], ops['stepD'],
                                                                            ops['lossG'],
                                                                            ops['lossD'], ops['predD'],
                                                                            ops['predDG'], ops['predD']],
                                                                           feed_dict=feed_dict)
        summary, step, lossG, lossD, pred_val, predDG1, predD1 = sess.run([ops['merged'], ops['stepD'],
                                                                            ops['lossG'],
                                                                            ops['lossD'], ops['predD'],
                                                                            ops['predDG'], ops['predD']],
                                                                           feed_dict=feed_dict)
        print('%.10f' % np.mean(predDG[0]))
        print('%.10f' % np.mean(predD[0][32:, :] - predDG[0]))
        print('%.10f' % np.mean(predD[0][:32, :] - predDG[0]))
        print('%.10f' % np.mean(predD1[0] - predD[0]))
        print('%.10f' % np.mean(predDG1[0] - predDG[0]))
        break


os.system('cp train_gan3.py %s' % (LOG_DIR))
if __name__ == "__main__":
    train()
    #LOG_FOUT.close()