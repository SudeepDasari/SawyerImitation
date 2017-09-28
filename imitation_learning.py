import tensorflow as tf
from tensorflow.contrib.layers.python import layers as tf_layers
from tensorflow.python.platform import flags
import tensorflow.contrib.slim as slim
import numpy as np
import os



class ImitationLearningModel:

    def __init__(self, vgg19_path, images=None, robot_configs=None, actions=None):
        self.images = images
        self.robot_configs = robot_configs
        self.actions = actions

        self.vgg_dict = np.load(os.path.join(vgg19_path, "vgg19.npy"), encoding='latin1').item()

        self.predicted_actions = None
        self.predicted_eeps = None

    def build(self):
        with slim.arg_scope([slim.layers.conv2d, slim.layers.fully_connected, tf_layers.layer_norm]):

            layer1 = tf_layers.layer_norm(self.vgg_layer(self.images), scope='conv1_norm')
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

            conv_out = tf.concat([fp_flat,
                                  tf.reshape(self.robot_configs, [30, 10])],  # dim of angles: 7, dim of eeps: 3
                                 1)

            layer4 = slim.layers.fully_connected(conv_out, 100, scope='fc1')

            layer5 = slim.layers.fully_connected(layer4, 100, scope='fc2')

            layer6 = slim.layers.fully_connected(layer5, 100, scope='fc3')

            shifted_sigmoid = lambda x: 2 * tf.sigmoid(x) - 1

            fc_actions = slim.layers.fully_connected(layer6, 7, scope='fc4_1', activation_fn=shifted_sigmoid)  # dim of velocities: 7

            fc_eeps = slim.layers.fully_connected(layer6, 3, scope='fc4_2', activation_fn=shifted_sigmoid)  # dim of eeps: 3

            self.predicted_actions, self.predicted_eeps = fc_actions, fc_eeps

    # Source: https://github.com/machrisaa/tensorflow-vgg/blob/master/vgg19.py
    def vgg_layer(self, images):
        rgb_scaled = tf.to_float(images) * 255

        vgg_mean = tf.convert_to_tensor(np.array([103.939, 116.779, 123.68], dtype=np.float32))

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)

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


if __name__ == '__main__':
    FLAGS = flags.FLAGS
    flags.DEFINE_string('vgg19_path', './', 'path to npy file')
    flags.DEFINE_string('data_path', './', 'path to tfrecords file')

    vgg19_path = FLAGS.vgg19_path
    data_path = FLAGS.data_path

    from read_tf_record import read_tf_record

    images_batch, angles_batch, velocities_batch, endeffector_poses_batch = read_tf_record(data_path)
    if int(tf.__version__[0]) >= 1.0:
        robot_configs_batch = tf.concat([angles_batch, endeffector_poses_batch], 1)
    else:
        robot_configs_batch = tf.concat(1, [angles_batch, endeffector_poses_batch])

    actions_batch = velocities_batch

    model = ImitationLearningModel(vgg19_path, images_batch, robot_configs_batch, actions_batch)
    model.build()

    print(model.predicted_actions, model.predicted_eeps)
