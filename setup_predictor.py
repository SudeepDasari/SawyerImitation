import tensorflow as tf
import numpy as np
from imitation_learning import ImitationLearningModel


def setup_predictor(model_path, vgg19_path):
    images_pl = tf.placeholder(tf.uint8, name='images', shape=(1, 224, 224, 3))
    configs_pl = tf.placeholder(tf.float32, name='configs', shape=(1, 10))

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
        feed_image = images.astype(np.uint8).reshape((1, 224, 224, 3))
        feed_config = robot_configs.astype(np.float32).reshape((1, 10))

        feed_dict = {
            images_pl: feed_image,
            configs_pl: feed_config,
        }
        predicted_actions, predicted_eeps = sess.run([model.predicted_actions, model.predicted_eeps], feed_dict)
        return predicted_actions[0], predicted_eeps[0]

    return predictor_func
