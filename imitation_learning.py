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
        for image, robot_config, action in zip(self.images, self.robot_configs, self.actions):
            with slim.arg_scope([slim.layers.conv2d, slim.layers.fully_connected]):

                layer1 = tf.norm(self.build_vgg(image))

                layer2 = tf.norm(slim.layers.conv2d(layer1, 64, [3, 3], stride=2, scope='conv2'))

                layer3 = tf.norm(slim.layers.conv2d(layer2, 64, [3, 3], stride=2, scope='conv3'))

                _, num_rows, num_cols, num_fp = layer3.get_shape()
                x_map = np.empty([num_rows, num_cols], np.float32)
                y_map = np.empty([num_rows, num_cols], np.float32)

                features = tf.reshape(tf.transpose(layer3, [0, 3, 1, 2]),
                                      [-1, num_rows * num_cols])
                softmax = tf.nn.softmax(features)

                fp_x = tf.reduce_sum(tf.mul(x_map, softmax), [1], keep_dims=True)
                fp_y = tf.reduce_sum(tf.mul(y_map, softmax), [1], keep_dims=True)

                fp_flat = tf.reshape(tf.concat(1, [fp_x, fp_y]), [-1, num_fp * 2])

                conv_out = tf.concat(2, [fp_flat, tf.reshape(robot_config, [-1, 10])]) # dim of angles: 7, dim of eepose: 3

                layer4 = slim.layers.fully_connected(conv_out, 100, scope='fc1')

                layer5 = slim.layers.fully_connected(layer4, 100, scope='fc2')

                layer6 = slim.layers.fully_connected(layer5, 100, scope='fc3')

                fc_out = slim.layers.fully_connected(layer6, 7, scope='fc4')

                self.predicted_actions.append(fc_out)

    def build_vgg(self, image):
        rgb_scaled = image * 255.0

        vgg_mean = [103.939, 116.779, 123.68]

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(axis=3, values=[
            blue - vgg_mean[0],
            green - vgg_mean[1],
            red - vgg_mean[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

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
        assert fc6.get_shape().as_list()[1:] == [4096]
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

    robot_configs_batch = tf.concat(1, [angles_batch, endeffector_poses_batch])
    actions_batch = velocities_batch

    model = ImitationLearningModel(images_batch, robot_configs_batch, actions_batch, vgg19_path)