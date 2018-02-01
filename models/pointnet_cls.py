import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util
from transform_nets import input_transform_net, feature_transform_net, input_transform_net_no_bn

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl

def get_model_rbf0(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}

    with tf.variable_scope('transform_net1', reuse=tf.AUTO_REUSE) as sc:
        transform = input_transform_net(point_cloud, is_training, bn_decay, K=3)
    point_cloud_transformed = tf.matmul(point_cloud, transform)
    point_cloud_transformed = tf.expand_dims(point_cloud_transformed, 3)

    #centroids = tf.constant(np.random.randn(1, 1, 3, 1024), dtype=tf.float32)
    centroids = tf.get_variable('centroids',
                                [1, 1, 3, 1024],
                                initializer=tf.constant_initializer(0.2*np.random.randn(1, 1, 3, 1024)),
                                dtype=tf.float32)

    feature = tf.tile(point_cloud_transformed, [1, 1, 1, 1024])

    bias = tf.tile(centroids, [batch_size, 1024, 1, 1])

    net = tf.subtract(feature, bias)
    net = tf.norm(net, axis=2, keep_dims=True)
    net = tf.exp(-net)

    # Symmetric function: max pooling
    features = tf_util.max_pool2d(net, [num_point,1],
                             padding='VALID', scope='maxpool')

    net = tf.reshape(features, [batch_size, -1])
    #net = tf_util.fully_connected(net, 1024, bn=True, is_training=is_training,
    #                              scope='fc0', bn_decay=bn_decay)
    #net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
    #                      scope='dp1')
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp1')
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp2')
    net = tf_util.fully_connected(net, 40, activation_fn=None, scope='fc3')

    return net, end_points, features, centroids

def get_model_rbf0_gan(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}

    with tf.variable_scope('transform_net1', reuse=tf.AUTO_REUSE) as sc:
        transform = input_transform_net_no_bn(point_cloud, is_training, bn_decay, K=3)
    point_cloud_transformed = tf.matmul(point_cloud, transform)
    point_cloud_transformed = tf.expand_dims(point_cloud_transformed, 3)

    #centroids = tf.constant(np.random.randn(1, 1, 3, 1024), dtype=tf.float32)
    centroids = tf.get_variable('centroids',
                                [1, 1, 3, 1024],
                                initializer=tf.constant_initializer(0.2*np.random.randn(1, 1, 3, 1024)),
                                dtype=tf.float32)

    feature = tf.tile(point_cloud_transformed, [1, 1, 1, 1024])

    bias = tf.tile(centroids, [batch_size, 1024, 1, 1])

    net = tf.subtract(feature, bias)
    net = tf.norm(net, axis=2, keep_dims=True)
    net = tf.exp(-net)

    # Symmetric function: max pooling
    features = tf_util.max_pool2d(net, [num_point,1],
                             padding='VALID', scope='maxpool')

    net = tf.reshape(features, [batch_size, -1])
    #net = tf_util.fully_connected(net, 1024, bn=True, is_training=is_training,
    #                              scope='fc0', bn_decay=bn_decay)
    #net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
    #                      scope='dp1')
    net = tf_util.fully_connected(net, 512, bn=False, is_training=is_training,
                                  scope='fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp1')
    net = tf_util.fully_connected(net, 256, bn=False, is_training=is_training,
                                  scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp2')
    net = tf_util.fully_connected(net, 41, activation_fn=None, scope='fc3')

    return net, end_points, features, centroids

def get_model_rbf(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}

    with tf.variable_scope('transform_net1', reuse=tf.AUTO_REUSE) as sc:
        transform = input_transform_net(point_cloud, is_training, bn_decay, K=3)
    point_cloud_transformed = tf.matmul(point_cloud, transform)
    point_cloud_transformed = tf.expand_dims(point_cloud_transformed, 3)

    c1 = 1024
    #centroids = tf.constant(np.random.randn(1, 1, 3, 1024), dtype=tf.float32)
    centroids = tf.get_variable('centroids',
                                [1, 1, 3, c1],
                                initializer=tf.constant_initializer(0.5*np.random.randn(1, 1, 3, c1)),
                                dtype=tf.float32)
    #the per-centroids weights to change the shape of the multi-norm
    weights = tf.get_variable('weights',
                              [1, 1, 4, c1],
                              initializer=tf.constant_initializer(0.01 * np.random.randn(1, 1, 3, c1)),)

    feature = tf.tile(point_cloud_transformed, [1, 1, 1, c1])

    bias = tf.tile(centroids, [batch_size, num_point, 1, 1])

    net = tf.subtract(feature, bias)
    net = tf.exp(net)
    net = tf.exp(-tf.concat([tf.norm(net, ord=0.5, axis=2, keep_dims=True),
                             #tf.norm(net, ord=0.8, axis=2, keep_dims=True),
                             tf.norm(net, ord=1, axis=2, keep_dims=True),
                             #tf.norm(net, ord=1.5, axis=2, keep_dims=True),
                             tf.norm(net, ord=2, axis=2, keep_dims=True),
                             #tf.norm(net, ord=3, axis=2, keep_dims=True),
                             #tf.norm(net, ord=4, axis=2, keep_dims=True),
                             tf.norm(net, ord=np.inf, axis=2, keep_dims=True),
                            ], axis=2))
    net = tf.multiply(net, tf.tile(weights, [batch_size, num_point, 1, 1]))
    #net = tf.exp(-net)
    # Symmetric function: max pooling
    features = tf_util.max_pool2d(net, [num_point,1],
                             padding='VALID', scope='maxpool')
    net = tf.transpose(features, perm=[0, 1, 3, 2])
    net = tf_util.conv2d(net, 3, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='mini_conv1', bn_decay=bn_decay)
    net = tf.reshape(net, [batch_size, -1])
    #net = tf_util.fully_connected(net, 1024, bn=True, is_training=is_training,
    #                              scope='fc0', bn_decay=bn_decay)
    #net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
    #                      scope='dp1')
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp1')
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp2')
    net = tf_util.fully_connected(net, 40, activation_fn=None, scope='fc3')

    return net, end_points, features, centroids

def get_model_rbf2(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}

    with tf.variable_scope('transform_net1', reuse=tf.AUTO_REUSE) as sc:
        transform = input_transform_net(point_cloud, is_training, bn_decay, K=3)
    point_cloud_transformed = tf.matmul(point_cloud, transform)
    point_cloud_transformed = tf.expand_dims(point_cloud_transformed, 3)

    #centroids = tf.constant(np.random.randn(1, 1, 3, 1024), dtype=tf.float32)
    c1 = 128
    c2 = 32
    centroids = tf.get_variable('centroids',
                                [1, 1, 3, c1],
                                initializer=tf.constant_initializer(np.random.randn(1, 1, 3, c1)),
                                dtype=tf.float32)

    sub_centroids = tf.get_variable('sub_centroids',
                                    [1, 1, 3, c2],
                                    initializer=tf.constant_initializer(0.05*np.random.randn(1, 1, 3, c2)),
                                    dtype=tf.float32)
    #sub_centroids = tf.constant(0.05*np.random.randn(1, 1, 3, c2), dtype=tf.float32)
    sub_bias = tf.add(tf.tile(tf.expand_dims(sub_centroids, 4), [1, 1, 1, 1, c1]),
                      tf.tile(tf.expand_dims(centroids, 3), [1, 1, 1, c2, 1]))
    sub_bias = tf.tile(sub_bias, [32, 1024, 1, 1, 1])
    sub_feature = tf.tile(tf.expand_dims(point_cloud_transformed, 4), [1, 1, 1, c2, c1])

    #feature = tf.tile(point_cloud_transformed, [1, 1, 1, c2, c1])

    #bias = tf.tile(centroids, [32, 1024, 1, 1])

    sub_net = tf.subtract(sub_feature, sub_bias)
    sub_net = tf.norm(sub_net, axis=2, keep_dims=True)
    sub_net = tf.reshape(sub_net, [32, 1024, 1, -1])
    net = tf.exp(-tf.square(sub_net))

    # Symmetric function: max pooling
    features = tf_util.max_pool2d(net, [num_point,1],
                             padding='VALID', scope='maxpool')

    net = tf.reshape(features, [batch_size, -1])
    net = tf_util.fully_connected(net, 1024, bn=True, is_training=is_training,
                                  scope='fc0', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp1')
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp1')
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp2')
    net = tf_util.fully_connected(net, 40, activation_fn=None, scope='fc3')

    return net, end_points, features, centroids

def get_model_rbf3(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}

    with tf.variable_scope('transform_net1', reuse=tf.AUTO_REUSE) as sc:
        transform = input_transform_net(point_cloud, is_training, bn_decay, K=3)
    point_cloud_transformed = tf.matmul(point_cloud, transform)
    point_cloud_transformed = tf.expand_dims(point_cloud_transformed, 3)

    #centroids = tf.constant(np.random.randn(1, 1, 3, 1024), dtype=tf.float32)
    c1 = 512
    c2 = 8
    centroids = tf.get_variable('centroids',
                                [1, 1, 3, c1],
                                initializer=tf.constant_initializer(np.random.randn(1, 1, 3, c1)),
                                dtype=tf.float32)

    sub_centroids = tf.get_variable('sub_centroids',
                                    [1, 1, 3, c2],
                                    initializer=tf.constant_initializer(0.05*np.random.randn(1, 1, 3, c2)),
                                    dtype=tf.float32)
    #sub_centroids = tf.constant(0.05*np.random.randn(1, 1, 3, c2), dtype=tf.float32)

    sub_bias = tf.add(tf.tile(tf.expand_dims(sub_centroids, 4), [1, 1, 1, 1, c1]),
                      tf.tile(tf.expand_dims(centroids, 3), [1, 1, 1, c2, 1]))
    sub_bias = tf.tile(sub_bias, [batch_size, 1024, 1, 1, 1])
    sub_feature = tf.tile(tf.expand_dims(point_cloud_transformed, 4), [1, 1, 1, c2, c1])
    sub_net = tf.exp(-tf.norm(tf.exp(tf.subtract(sub_feature, sub_bias)), ord=3, axis=2, keep_dims=True))
    sub_net = tf.squeeze(sub_net)
    sub_net = tf.transpose(sub_net, perm=[0, 1, 3, 2])
    sub_net = tf_util.max_pool2d(sub_net, [num_point,1], stride=[1, 1],
                                padding='VALID', scope='maxpool')
    #sub_net = tf_util.conv2d(sub_net, 16, [1,1],
    #                                 padding='VALID', stride=[1,1],
    #                                 bn=True, is_training=is_training,
    #                                 scope='mini_conv1', bn_decay=bn_decay)
    sub_net = tf_util.conv2d(sub_net, 2, [1,1],
                                     padding='VALID', stride=[1,1],
                                     bn=True, is_training=is_training,
                                     scope='mini_conv2', bn_decay=bn_decay)
    sub_net = tf.squeeze(sub_net)


    feature = tf.tile(point_cloud_transformed, [1, 1, 1, c1])
    bias = tf.tile(centroids, [batch_size, 1024, 1, 1])
    net = tf.subtract(feature, bias)
    net = tf.exp(net)
    net = tf.norm(net, ord=3, axis=2, keep_dims=True)
    net = tf.exp(-net)
    net = tf_util.max_pool2d(net, [num_point,1],
                             padding='VALID', scope='maxpool')
    net = tf.expand_dims(tf.squeeze(net), 2)

    features = tf.concat([net, sub_net], axis=2)

    # Symmetric function: max pooling
    #features = tf_util.max_pool2d(net, [num_point,1],
    #                         padding='VALID', scope='maxpool')

    net = tf.reshape(features, [batch_size, -1])
    #net = tf_util.fully_connected(net, 1024, bn=True, is_training=is_training,
    #                              scope='fc0', bn_decay=bn_decay)
    #net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
    #                      scope='dp1')
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp1')
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp2')
    net = tf_util.fully_connected(net, 40, activation_fn=None, scope='fc3')

    return net, end_points, features, centroids

def get_model_elm(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}

    with tf.variable_scope('transform_net1', reuse=tf.AUTO_REUSE) as sc:
        transform = input_transform_net(point_cloud, is_training, bn_decay, K=3)
    point_cloud_transformed = tf.matmul(point_cloud, transform)
    input_image = tf.expand_dims(point_cloud_transformed, -1)

    random_weights = tf.constant(np.random.randn(3, 4096), dtype=tf.float32)
    random_weights1 = tf.expand_dims(random_weights, 0)
    random_weights1 = tf.concat([random_weights1, random_weights1], axis=0)#2
    random_weights1 = tf.concat([random_weights1, random_weights1], axis=0)#4
    random_weights1 = tf.concat([random_weights1, random_weights1], axis=0)#8
    random_weights1 = tf.concat([random_weights1, random_weights1], axis=0)#16
    random_weights1 = tf.concat([random_weights1, random_weights1], axis=0)#32

    net = tf.matmul(point_cloud, random_weights1)
    net = tf.expand_dims(net, 2)

    # Symmetric function: max pooling
    features = tf_util.max_pool2d(net, [num_point,1],
                             padding='VALID', scope='maxpool')

    net = tf.reshape(features, [batch_size, -1])
    net = tf_util.fully_connected(net, 1024, bn=True, is_training=is_training,
                                  scope='fc0', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp1')
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp1')
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp2')
    net = tf_util.fully_connected(net, 40, activation_fn=None, scope='fc3')

    return net, end_points, features

def get_model_half(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}

    with tf.variable_scope('transform_net1', reuse=tf.AUTO_REUSE) as sc:
        transform = input_transform_net(point_cloud, is_training, bn_decay, K=3)
    point_cloud_transformed = tf.matmul(point_cloud, transform)
    input_image = tf.expand_dims(point_cloud_transformed, -1)

    net = tf_util.conv2d(input_image, 64, [1,3],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)

    with tf.variable_scope('transform_net2', reuse=tf.AUTO_REUSE) as sc:
        transform = feature_transform_net(net, is_training, bn_decay, K=64)
    end_points['transform'] = transform
    net_transformed = tf.matmul(tf.squeeze(net, axis=[2]), transform)
    net_transformed = tf.expand_dims(net_transformed, [2])

    net = tf_util.conv2d(net_transformed, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 512, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv5', bn_decay=bn_decay)

    c1 = 512
    #centroids = tf.constant(np.random.randn(1, 1, 3, 1024), dtype=tf.float32)
    centroids = tf.get_variable('centroids',
                                [1, 1, 3, c1],
                                initializer=tf.constant_initializer(0.5*np.random.randn(1, 1, 3, c1)),
                                dtype=tf.float32)
    net2 = tf.subtract(tf.tile(tf.expand_dims(point_cloud_transformed, 3), [1, 1, 1, c1]), tf.tile(centroids, [batch_size, num_point, 1, 1]))
    net2 = tf.norm(net2, axis=2, keep_dims=True)
    net2 = tf.exp(-net2)
    net = tf.concat([net, net2], axis=2)

    # Symmetric function: max pooling
    features = tf_util.max_pool2d(net, [num_point,1],
                             padding='VALID', scope='maxpool')

    net = tf.reshape(features, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp1')
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp2')
    net = tf_util.fully_connected(net, 40, activation_fn=None, scope='fc3')

    return net, end_points, features, centroids


def get_model_rbf_transform(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}

    with tf.variable_scope('transform_net1', reuse=tf.AUTO_REUSE) as sc:
        transform = input_transform_net(point_cloud, is_training, bn_decay, K=3)
    point_cloud_transformed = tf.matmul(point_cloud, transform)
    input_image = tf.expand_dims(point_cloud_transformed, -1)

    c1 = 64
    #centroids = tf.constant(np.random.randn(1, 1, 3, 1024), dtype=tf.float32)
    centroids = tf.get_variable('centroids',
                                [1, 1, 3, c1],
                                initializer=tf.constant_initializer(0.5*np.random.randn(1, 1, 3, c1)),
                                dtype=tf.float32)
    net = tf.subtract(tf.tile(tf.expand_dims(point_cloud_transformed, 3), [1, 1, 1, c1]), tf.tile(centroids, [batch_size, num_point, 1, 1]))
    net = tf.norm(net, axis=2, keep_dims=True)
    net = tf.exp(-net)

    with tf.variable_scope('transform_net2', reuse=tf.AUTO_REUSE) as sc:
        transform = feature_transform_net(net, is_training, bn_decay, K=64)
    end_points['transform'] = transform
    net_transformed = tf.matmul(tf.squeeze(net, axis=[2]), transform)
    net_transformed = tf.expand_dims(net_transformed, [2])

    c2 = 256
    #centroids = tf.constant(np.random.randn(1, 1, 3, 1024), dtype=tf.float32)
    centroids2 = tf.get_variable('centroids2',
                                [1, 1, c2, 64],
                                initializer=tf.constant_initializer(0.5*np.random.randn(1, 1, c2, 64)),
                                dtype=tf.float32)
    net = tf.subtract(tf.tile(net_transformed, [1, 1, c2, 1]), tf.tile(centroids2, [batch_size, num_point, 1, 1]))
    net = tf.norm(net, axis=3, keep_dims=True)
    net = tf.exp(-net)

    # Symmetric function: max pooling
    features = tf_util.max_pool2d(net, [num_point,1],
                             padding='VALID', scope='maxpool')

    net = tf.reshape(features, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp1')
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp2')
    net = tf_util.fully_connected(net, 40, activation_fn=None, scope='fc3')

    return net, end_points, features, centroids

def get_model(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}

    with tf.variable_scope('transform_net1', reuse=tf.AUTO_REUSE) as sc:
        transform = input_transform_net(point_cloud, is_training, bn_decay, K=3)
    point_cloud_transformed = tf.matmul(point_cloud, transform)
    input_image = tf.expand_dims(point_cloud_transformed, -1)

    net = tf_util.conv2d(input_image, 64, [1,3],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)

    with tf.variable_scope('transform_net2', reuse=tf.AUTO_REUSE) as sc:
        transform = feature_transform_net(net, is_training, bn_decay, K=64)
    end_points['transform'] = transform
    net_transformed = tf.matmul(tf.squeeze(net, axis=[2]), transform)
    net_transformed = tf.expand_dims(net_transformed, [2])

    net = tf_util.conv2d(net_transformed, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv5', bn_decay=bn_decay)

    # Symmetric function: max pooling
    features = tf_util.max_pool2d(net, [num_point,1],
                             padding='VALID', scope='maxpool')

    net = tf.reshape(features, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp1')
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp2')
    net = tf_util.fully_connected(net, 40, activation_fn=None, scope='fc3')

    return net, end_points, features


def get_loss(pred, label, end_points, reg_weight=0.001):
    """ pred: B*NUM_CLASSES,
        label: B, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)

    # Enforce the transformation as orthogonal matrix
    transform = end_points['transform'] # BxKxK
    K = transform.get_shape()[1].value
    mat_diff = tf.matmul(transform, tf.transpose(transform, perm=[0,2,1]))
    mat_diff -= tf.constant(np.eye(K), dtype=tf.float32)
    mat_diff_loss = tf.nn.l2_loss(mat_diff) 
    tf.summary.scalar('mat loss', mat_diff_loss)

    return classify_loss + mat_diff_loss * reg_weight


if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,3))
        outputs = get_model(inputs, tf.constant(True))
        print(outputs)
