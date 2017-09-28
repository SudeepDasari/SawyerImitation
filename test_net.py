import tensorflow as tf
import numpy as np
import cPickle as pickle
from train_net import Model
from read_tf_record import read_tf_record

from tensorflow.python.platform import flags


NUM_FRAMES = 60
NUM_JOINTS = 7
STATE_DIM = 3
IMG_WIDTH = 224
IMG_HEIGHT = 224
COLOR_CHANNELS = 3



def mean_squared_error(true, pred):
    """L2 distance between tensors true and pred.

    Args:
      true: the ground truth image.
      pred: the predicted image.
    Returns:
      mean squared error between ground truth and predicted image.
    """
    return tf.reduce_sum(tf.square(true - pred)) / tf.to_float(tf.size(pred))



def main():
    vgg19_path = FLAGS.vgg19_path
    data_path = FLAGS.data_path
    output_dir = FLAGS.model_path

    with tf.variable_scope('model', reuse=None) as training_scope:
        images_batch, angles_batch, actions_batch, endeffector_poses_batch, use_frames_batch, \
            final_endeffector_poses_batch = read_tf_record(data_path + 'train.tfrecords', shuffle=False)

        robot_configs_batch = tf.concat([angles_batch, endeffector_poses_batch], 1)
        model = Model(vgg19_path, images_batch, robot_configs_batch, actions_batch, use_frames_batch,
                      final_endeffector_poses_batch)

    with tf.variable_scope('val_model', reuse=None):
        val_images_batch, val_angles_batch, val_actions_batch, val_endeffector_poses_batch, val_use_frames_batch, \
        val_final_endeffector_poses_batch = read_tf_record(data_path + 'test.tfrecords', d_append='test', shuffle=False)

        val_robot_configs_batch = tf.concat([val_angles_batch, val_endeffector_poses_batch], 1)
        val_model = Model(vgg19_path, val_images_batch, val_robot_configs_batch, val_actions_batch, val_use_frames_batch,
                      val_final_endeffector_poses_batch, training_scope)


    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    # Make training session.

    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    saver = tf.train.Saver(vars, max_to_keep=0)

    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
    summary_writer = tf.summary.FileWriter(output_dir, graph=sess.graph, flush_secs=10)

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    saver.restore(sess, output_dir)

    reals = []
    predictions = []
    for i in range(20):
        real_actions, predicted_actions, loss, summary_str = sess.run([val_model.m.actions, val_model.m.predicted_actions, val_model.loss, val_model.summ_op])
        print 'loss is', loss

        reals.append(real_actions)
        predictions.append(predicted_actions)

    pickle.dump({'reals':reals, 'predictions':predictions}, open('real_vs_pred.pkl', 'wb'))

    coord.request_stop()
    coord.join(threads)


if __name__ == '__main__':
    FLAGS = flags.FLAGS
    flags.DEFINE_string('vgg19_path', './', 'path to npy file')
    flags.DEFINE_string('data_path', './', 'path to tfrecords file')
    flags.DEFINE_string('model_path', './', 'path to output model/stats')
    main()