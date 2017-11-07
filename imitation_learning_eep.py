import tensorflow as tf
from tensorflow.contrib.layers.python import layers as tf_layers
from tensorflow.python.platform import flags
import tensorflow.contrib.slim as slim
import numpy as np
import os


class ImitationLearningModel:
    def __init__(self, vgg19_path, images=None, robot_configs=None):
        self.images = images
        self.robot_configs = robot_configs
        self.vgg_dict = np.load(os.path.join(vgg19_path, "vgg19.npy"), encoding='latin1').item()

        self.predicted_final_eep = None
        self.predicted_actions = None

    def build(self):
        with slim.arg_scope([slim.layers.conv2d, slim.layers.fully_connected, tf_layers.layer_norm]):

            layer1 = tf_layers.layer_norm(self.vgg_layer(self.images), scope='conv1_norm')
            layer2 = tf_layers.layer_norm(
                slim.layers.conv2d(layer1, 32, [3, 3], stride=2, scope='conv2'), scope='conv2_norm')

            layer3 = tf_layers.layer_norm(
                slim.layers.conv2d(layer2, 32, [3, 3], stride=2, scope='conv3'), scope='conv3_norm')

            batch_size, num_rows, num_cols, num_fp = layer3.get_shape()
            # print 'shape', layer3.get_shape
            num_rows, num_cols, num_fp = [int(x) for x in [num_rows, num_cols, num_fp]]

            x_map = np.empty([num_rows, num_cols], np.float32)
            y_map = np.empty([num_rows, num_cols], np.float32)

            for i in range(num_rows):
                for j in range(num_cols):
                    x_map[i, j] = (i - num_rows / 2.0) / num_rows
                    y_map[i, j] = (j - num_cols / 2.0) / num_cols

            x_map = tf.convert_to_tensor(x_map)
            y_map = tf.convert_to_tensor(y_map)

            x_map = tf.reshape(x_map, [num_rows * num_cols])
            y_map = tf.reshape(y_map, [num_rows * num_cols])

            features = tf.reshape(tf.transpose(layer3, [0, 3, 1, 2]), [-1, num_rows * num_cols])
            softmax = tf.nn.softmax(features)
            # print 'softmax', softmax

            fp_x = tf.reduce_sum(tf.multiply(x_map, softmax), [1], keep_dims=True)
            fp_y = tf.reduce_sum(tf.multiply(y_map, softmax), [1], keep_dims=True)
            self.fp_y = fp_y
            self.fp_x = fp_x
            # print 'fp_x', fp_x
            # print 'fp_y', fp_y
            fp_flat = tf.reshape(tf.concat([fp_x, fp_y], 1), [-1, num_fp * 2])
            # print 'fp_flat', fp_flat
            # print 'configs', self.robot_configs

            fc_layer1 = slim.layers.fully_connected(fp_flat, 50, scope='fc1')

            self.predicted_final_eep = slim.layers.fully_connected(fc_layer1, 3, scope='predicted_final_eep',
                                                                   activation_fn=None)  # dim of eeps: 3

            conv_out = tf.concat([fp_flat,
                                  self.robot_configs,  # dim of angles: 7, dim of eeps: 3
                                  self.predicted_final_eep],
                                 1)

            fc_layer2 = slim.layers.fully_connected(conv_out, 50, scope='fc2')

            self.predicted_actions = slim.layers.fully_connected(fc_layer2, 3, scope='predicted_action',
                                                             activation_fn=None)  # dim of velocities: 3

    # Source: https://github.com/machrisaa/tensorflow-vgg/blob/master/vgg19.py
    def vgg_layer(self, images):
        bgr_scaled = tf.to_float(images)

        vgg_mean = tf.convert_to_tensor(np.array([103.939, 116.779, 123.68], dtype=np.float32))
        # print 'images', images
        blue, green, red = tf.split(axis=3, num_or_size_splits=3, value=bgr_scaled)
        # print 'blue', blue
        # print 'green', green
        # print 'red', red

        bgr = tf.concat(axis=3, values=[
            blue - vgg_mean[0],
            green - vgg_mean[1],
            red - vgg_mean[2],
        ])

        name = "conv1_1"
        with tf.variable_scope(name):
            filt = tf.constant(self.vgg_dict[name][0], name="filter")

            conv = tf.nn.conv2d(bgr, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = tf.constant(self.vgg_dict[name][1], name="biases")
            bias = tf.nn.bias_add(conv, conv_biases)

            out = tf.nn.relu(bias)
        return out
