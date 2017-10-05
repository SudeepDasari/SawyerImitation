import tensorflow as tf
import imp
import numpy as np
from imitation_learning import ImitationLearningModel
import os
import cv2
import time

def setup_predictor(model_path, vgg19_path):
    images_pl = tf.placeholder(tf.float32, name='images', shape=(1, 64, 64, 3))
    configs_pl = tf.placeholder(tf.float32, name='configs', shape=(1, 10))

    with tf.variable_scope('model', reuse=None):
        model = ImitationLearningModel(vgg19_path, images_pl, configs_pl)
        model.build()

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)

    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    saver = tf.train.Saver(vars, max_to_keep=0)

    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    saver.restore(sess, model_path)

    def predictor_func(images=None, robot_configs=None):
        feed_image = images.astype(np.float32).reshape((1, 64, 64, 3))
        feed_config = robot_configs.astype(np.float32).reshape((1, 10))

        feed_dict = {
            images_pl: feed_image,
            configs_pl: feed_config,
        }
        predicted_actions, predicted_eeps = sess.run([model.predicted_actions, model.predicted_eeps], feed_dict)
        return predicted_actions[0], predicted_eeps[0]

    return predictor_func

if __name__ == '__main__':
    demo_image = cv2.imread('demo_frame.png')
    demo_action = np.zeros(10)
    pred = setup_predictor('modeldata_100_50/modelfinal', 'out/')
    pred_actions, pred_eep = pred(demo_image, demo_action)
    print 'predicted actions', pred_actions
    print 'predicted eep', pred_eep

    t0 = time.time()
    for i in range(1000):
        pred_actions, pred_eep = pred(demo_image, demo_action)
    t1 = time.time()
    print 'time:', (t1 - t0) / 1000