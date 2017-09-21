import tensorflow as tf
from tensorflow.contrib.layers.python import layers as tf_layers
from tensorflow.python.platform import flags
import tensorflow.contrib.slim as slim
import numpy as np
import os

FLAGS = flags.FLAGS
flags.DEFINE_string('vgg19_path', './', 'path to tfrecords file')


class ImitationLearningModel:

    def __init__(self, images, robot_configs, actions, vgg19_path):
        self.images = images
        self.robot_configs = robot_configs
        self.actions = actions

        self.data_dict = np.load(os.path.join(vgg19_path, "vgg19.npy"), encoding='latin1').item()

        self.predicted_actions = []

    def build(self):
        with slim.arg_scope([slim.layers.conv2d, slim.layers.fully_connected, tf_layers.layer_norm]):

            layer1 = tf_layers.layer_norm(self.build_vgg(self.images), scope='conv1_norm')

            layer2 = tf_layers.layer_norm(
                slim.layers.conv2d(layer1, 64, [3, 3], stride=2, scope='conv2'), scope='conv2_norm')

            layer3 = tf_layers.layer_norm(
                slim.layers.conv2d(layer2, 64, [3, 3], stride=2, scope='conv3'), scope='conv3_norm')

            batch_size, num_rows, num_cols, num_fp = layer3.get_shape()
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

            fp_x = tf.reduce_sum(tf.multiply(x_map, softmax), [1], keep_dims=True)
            fp_y = tf.reduce_sum(tf.multiply(y_map, softmax), [1], keep_dims=True)

            fp_flat = tf.reshape(tf.concat([fp_x, fp_y], 1), [-1, num_fp * 2])

            conv_out = tf.concat([fp_flat, tf.reshape(self.robot_configs, [batch_size, 10])], 1) # dim of angles: 7, dim of eepose: 3

            layer4 = slim.layers.fully_connected(conv_out, 100, scope='fc1')

            layer5 = slim.layers.fully_connected(layer4, 100, scope='fc2')

            layer6 = slim.layers.fully_connected(layer5, 100, scope='fc3')

            fc_out = slim.layers.fully_connected(layer6, 7, scope='fc4')

            self.predicted_actions.append(fc_out)

    # Source: https://github.com/machrisaa/tensorflow-vgg/blob/master/vgg19.py
    def build_vgg(self, images):
        rgb_scaled = tf.to_float(images) * 255

        vgg_mean = [tf.constant(103.939, dtype=tf.float32),
                    tf.constant(116.779, dtype=tf.float32),
                    tf.constant(123.68, dtype=tf.float32)]

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)

        bgr = tf.concat(axis=3, values=[
            blue - vgg_mean[0],
            green - vgg_mean[1],
            red - vgg_mean[2],
        ])

        conv1_1 = self.vgg_conv_layer(bgr, "conv1_1")
        conv1_2 = self.vgg_conv_layer(conv1_1, "conv1_2")
        pool1 = self.vgg_max_pool(conv1_2, 'pool1')

        conv2_1 = self.vgg_conv_layer(pool1, "conv2_1")
        conv2_2 = self.vgg_conv_layer(conv2_1, "conv2_2")
        pool2 = self.vgg_max_pool(conv2_2, 'pool2')

        conv3_1 = self.vgg_conv_layer(pool2, "conv3_1")
        conv3_2 = self.vgg_conv_layer(conv3_1, "conv3_2")
        conv3_3 = self.vgg_conv_layer(conv3_2, "conv3_3")
        conv3_4 = self.vgg_conv_layer(conv3_3, "conv3_4")
        pool3 = self.vgg_max_pool(conv3_4, 'pool3')

        conv4_1 = self.vgg_conv_layer(pool3, "conv4_1")
        conv4_2 = self.vgg_conv_layer(conv4_1, "conv4_2")
        conv4_3 = self.vgg_conv_layer(conv4_2, "conv4_3")
        conv4_4 = self.vgg_conv_layer(conv4_3, "conv4_4")
        pool4 = self.vgg_max_pool(conv4_4, 'pool4')

        conv5_1 = self.vgg_conv_layer(pool4, "conv5_1")
        conv5_2 = self.vgg_conv_layer(conv5_1, "conv5_2")
        conv5_3 = self.vgg_conv_layer(conv5_2, "conv5_3")
        conv5_4 = self.vgg_conv_layer(conv5_3, "conv5_4")
        pool5 = self.vgg_max_pool(conv5_4, 'pool5')

        fc6 = self.vgg_fc_layer(pool5, "fc6")
        relu6 = tf.nn.relu(fc6)

        fc7 = self.vgg_fc_layer(relu6, "fc7")
        relu7 = tf.nn.relu(fc7)

        fc8 = self.vgg_fc_layer(relu7, "fc8")

        return fc8

    def vgg_avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def vgg_max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def vgg_conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.vgg_get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.vgg_get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def vgg_fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.vgg_get_fc_weight(name)
            biases = self.vgg_get_bias(name)

            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def vgg_get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def vgg_get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def vgg_get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")


if __name__ == '__main__':
    vgg19_path = FLAGS.vgg19_path

    from read_tf_record import read_tf_record
    images_batch, angles_batch, velocities_batch, endeffector_poses_batch = read_tf_record('train/')

    robot_configs_batch = tf.concat([angles_batch, endeffector_poses_batch], 1)
    actions_batch = velocities_batch

    # images = tf.split(axis=0, num_or_size_splits=images_batch.get_shape()[0], value=images_batch)
    # images = [tf.squeeze(img) for img in images]
    #
    # robot_configs = tf.split(axis=0, num_or_size_splits=robot_configs_batch.get_shape()[0], value=robot_configs_batch)
    # robot_configs = [tf.squeeze(conf) for conf in robot_configs]
    #
    # actions = tf.split(axis=0, num_or_size_splits=actions_batch.get_shape()[0], value=actions_batch)
    # actions = [tf.squeeze(act) for act in actions]

    model = ImitationLearningModel(images_batch, robot_configs_batch, actions_batch, vgg19_path)
    model.build()
