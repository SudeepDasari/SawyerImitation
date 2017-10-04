import tensorflow as tf
import imp
import numpy as np
from imitation_learning import ImitationLearningModel
import os


def setup_predictor(model_path, vgg19_path):
    images_pl = tf.placeholder(tf.float32, name='images', shape=(32, 64, 64, 3))
    configs_pl = tf.placeholder(tf.float32, name='configs', shape=(32, 10))

    with tf.variable_scope('model', reuse=None):
        model = ImitationLearningModel(vgg19_path, images_pl, configs_pl)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)

    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    saver = tf.train.Saver(vars, max_to_keep=0)

    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    saver.restore(sess, model_path)

    def predictor_func(images=None, robot_configs=None):
        feed_dict = {
            images_pl: images,
            configs_pl: robot_configs,
        }
        predicted_actions, predicted_eeps = sess.run([model.predicted_actions, model.predicted_eeps], feed_dict)
        return predicted_actions[0], predicted_eeps[0]

    return predictor_func
