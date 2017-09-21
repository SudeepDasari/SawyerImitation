import tensorflow as tf
from tensorflow.contrib.layers.python import layers as tf_layers
import tensorflow.contrib.slim as slim
import numpy as np


class ImitationLearningModel:

    def __init__(self, images, robot_configs, actions):
        self.images = images
        self.robot_configs = robot_configs
        self.actions = actions

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

        conv1_1 = self.conv_layer(bgr, "conv1_1")
        conv1_2 = self.conv_layer(conv1_1, "conv1_2")
        pool1 = self.max_pool(conv1_2, 'pool1')

        conv2_1 = self.conv_layer(pool1, "conv2_1")
        conv2_2 = self.conv_layer(conv2_1, "conv2_2")
        pool2 = self.max_pool(conv2_2, 'pool2')

        conv3_1 = self.conv_layer(pool2, "conv3_1")
        conv3_2 = self.conv_layer(conv3_1, "conv3_2")
        conv3_3 = self.conv_layer(conv3_2, "conv3_3")
        conv3_4 = self.conv_layer(conv3_3, "conv3_4")
        pool3 = self.max_pool(conv3_4, 'pool3')

        conv4_1 = self.conv_layer(pool3, "conv4_1")
        conv4_2 = self.conv_layer(conv4_1, "conv4_2")
        conv4_3 = self.conv_layer(conv4_2, "conv4_3")
        conv4_4 = self.conv_layer(conv4_3, "conv4_4")
        pool4 = self.max_pool(conv4_4, 'pool4')

        conv5_1 = self.conv_layer(pool4, "conv5_1")
        conv5_2 = self.conv_layer(conv5_1, "conv5_2")
        conv5_3 = self.conv_layer(conv5_2, "conv5_3")
        conv5_4 = self.conv_layer(conv5_3, "conv5_4")
        pool5 = self.max_pool(conv5_4, 'pool5')

        fc6 = self.fc_layer(pool5, "fc6")
        assert fc6.get_shape().as_list()[1:] == [4096]
        relu6 = tf.nn.relu(fc6)

        fc7 = self.fc_layer(relu6, "fc7")
        relu7 = tf.nn.relu(fc7)

        fc8 = self.fc_layer(relu7, "fc8")

        return fc8

    def vgg_conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
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

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def vgg_get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def vgg_get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def vgg_get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")


if __name__ == '__main__':
    from read_tf_record import read_tf_record
    images, angles, velocities, endeffector_poses = read_tf_record('./')

    robot_configs = tf.concat(1, [angles, endeffector_poses])
    actions = velocities

    model = ImitationLearningModel(images, robot_configs, actions)