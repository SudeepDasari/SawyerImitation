import tensorflow as tf
import numpy as np
from imitation_learning import ImitationLearningModel
import os
import cv2
import time


def setup_predictor(model_path, vgg19_path):
    images_pl = tf.placeholder(tf.uint8, name='images', shape=(1, 224, 224, 3))
    configs_pl = tf.placeholder(tf.float32, name='configs', shape=(1, 10))

    with tf.variable_scope('model', reuse=None):
        model = ImitationLearningModel(vgg19_path, images=images_pl, robot_configs=configs_pl)
        model.build()
        print model.predicted_actions
        print model.predicted_eeps


    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)

    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    saver = tf.train.Saver(vars, max_to_keep=0)

    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    saver.restore(sess, model_path)

    def predictor_func(images=None, robot_configs=None):

        feed_image = images.astype(np.uint8).reshape((1, 224, 224, 3))
        feed_config = robot_configs.astype(np.float32).reshape((1, 10))

        feed_dict = {
            images_pl: feed_image,
            configs_pl: feed_config
        }


        predicted_actions, predicted_eeps = sess.run([model.predicted_actions, model.predicted_eeps], feed_dict)
        # print 'fp_x', fp_x
        # print 'fp_y', fp_y
        # height, width = images.shape[:2]
        # drawn = images.copy()
        # for i in range(32):
        #     cv2.circle(drawn, (int(height * (fp_x[i] + 0.5)) , int(width * (fp_y[i] + 0.5))), 5, (0, 0, 255), -1)

        return predicted_actions[0], predicted_eeps[0]

    return predictor_func

if __name__ == '__main__':
    demo_image = cv2.imread('../../test0.jpg')
    demo_action = np.zeros(10)
    demo_action[0] = 0.1
    pred = setup_predictor('../../single_lossrev_model/modelfinal', '../../out/')
    pred_actions, pred_eep = pred(demo_image, demo_action)
    print 'predicted actions', pred_actions
    print 'predicted eep', pred_eep

    t0 = time.time()
    for i in range(1000):

        pred_actions, pred_eep = pred(demo_image, demo_action)
    t1 = time.time()
    print 'time:', (t1 - t0) / 1000
