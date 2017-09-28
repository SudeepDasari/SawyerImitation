import tensorflow as tf
import numpy as np
from imitation_learning import ImitationLearningModel
from read_tf_record import read_tf_record
from tensorflow.python.platform import flags

def mean_squared_error(true, pred):
    """L2 distance between tensors true and pred.

    Args:
      true: the ground truth image.
      pred: the predicted image.
    Returns:
      mean squared error between ground truth and predicted image.
    """
    return tf.reduce_sum(tf.square(true - pred)) / tf.to_float(tf.size(pred))


class Model:
    def __init__(self,
                 vgg19_path,
                 images_batch,
                 robot_configs_batch,
                 actions_batch,
                 use_frames_batch,
                 final_endeffector_poses_batch,
                 reuse_train_scope = None):

        if reuse_train_scope is None:
            self.m = ImitationLearningModel(vgg19_path, images_batch, robot_configs_batch, actions_batch)
            self.m.build()
        else:
            with tf.variable_scope(reuse_train_scope, reuse=True):
                self.m = ImitationLearningModel(vgg19_path, images_batch, robot_configs_batch, actions_batch)
                self.m.build()

        action_loss = mean_squared_error(self.m.actions, self.m.predicted_actions)
        eep_loss = mean_squared_error(tf.multiply(use_frames_batch, final_endeffector_poses_batch),
                                      tf.multiply(use_frames_batch, self.m.predicted_eeps))
        self.eep_multiplier = 0.01
        loss = action_loss + self.eep_multiplier * eep_loss
        self.loss = loss
        self.lr = 0.001
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
        self.summ_op = tf.summary.merge([tf.summary.scalar('loss', loss),
                                         tf.summary.scalar('eep_loss', eep_loss),
                                         tf.summary.scalar('action_loss', action_loss)])




def main():
    vgg19_path = FLAGS.vgg19_path
    data_path = FLAGS.data_path
    output_dir = FLAGS.model_path

    NUM_ITERS = 10000

    with tf.variable_scope('model', reuse=None) as training_scope:
        images_batch, angles_batch, actions_batch, endeffector_poses_batch, use_frames_batch, \
            final_endeffector_poses_batch = read_tf_record(data_path + 'train.tfrecords')

        robot_configs_batch = tf.concat([angles_batch, endeffector_poses_batch], 1)
        model = Model(vgg19_path, images_batch, robot_configs_batch, actions_batch, use_frames_batch,
                      final_endeffector_poses_batch)

    with tf.variable_scope('val_model', reuse=None):
        val_images_batch, val_angles_batch, val_actions_batch, val_endeffector_poses_batch, val_use_frames_batch, \
        val_final_endeffector_poses_batch = read_tf_record(data_path + 'test.tfrecords', d_append='test')

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

    for itr in range(NUM_ITERS):
        if itr % 180 == 0:
            print 'Running Validation'
            val_loss = 0
            for t_itr in range(30):
                val_loss += sess.run([val_model.loss])[0] / 30.0

            val_summary = tf.Summary()
            val_summary.value.add(tag="val_model/val_loss", simple_value=val_loss)

            summary_writer.add_summary(val_summary, itr)
            print 'Validation Loss:', val_loss

        cost, _, summary_str = sess.run([model.loss, model.train_op, model.summ_op])
                                            #feed_dict)
        if itr % 10 == 0:
            summary_writer.add_summary(summary_str, itr)
        if itr % 100 == 0:
            print 'Cost', cost, 'On iter', itr
        if itr % 180 == 0 and itr > 0:
            saver.save(sess, output_dir + '/model'+ str(itr))

    saver.save(sess, output_dir + '/modelfinal')

    coord.request_stop()
    coord.join(threads)
    sess.close()


if __name__ == '__main__':
    FLAGS = flags.FLAGS
    flags.DEFINE_string('vgg19_path', './', 'path to npy file')
    flags.DEFINE_string('data_path', './', 'path to tfrecords file')
    flags.DEFINE_string('model_path', './', 'path to output model/stats')
    main()
