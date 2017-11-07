import tensorflow as tf
import numpy as np
from imitation_learning import ImitationLearningModel
from read_tf_record import read_tf_record
from tensorflow.python.platform import flags


def l2_loss(true, pred):
    """L2 distance between tensors true and pred.

    Args:
      true: the ground truth image.
      pred: the predicted image.
    Returns:
      mean squared error between ground truth and predicted image.
    """
    return tf.reduce_sum(tf.square(true - pred)) / pred.shape.as_list()[0]


def l1_loss(true, pred):
    return tf.reduce_sum(tf.abs(true - pred)) / pred.shape.as_list()[0]


class Model:
    def __init__(self,
                 vgg19_path,
                 images_batch,
                 robot_configs_batch,
                 actions_batch,
                 use_frames_batch,
                 final_endeffector_poses_batch,
                 reuse_train_scope=None):

        if reuse_train_scope is None:
            self.m = ImitationLearningModel(vgg19_path, images_batch, robot_configs_batch, actions_batch)
            self.m.build()
        else:
            with tf.variable_scope(reuse_train_scope, reuse=True):
                self.m = ImitationLearningModel(vgg19_path, images_batch, robot_configs_batch, actions_batch)
                self.m.build()


        action_loss = 0.01 * l2_loss(self.m.actions, self.m.predicted_actions) + l1_loss(self.m.actions, self.m.predicted_actions)
        eep_loss = tf.reduce_sum(tf.square(tf.multiply(use_frames_batch, final_endeffector_poses_batch) -
                                           tf.multiply(use_frames_batch, self.m.predicted_eeps)))
        eep_loss = tf.cond(eep_loss > 0,
                           lambda: tf.divide(eep_loss, tf.cast(tf.count_nonzero(use_frames_batch), tf.float32)),
                           lambda: eep_loss)

        self.eep_multiplier = 0.01
        loss = action_loss + self.eep_multiplier * eep_loss
        self.eep_loss = eep_loss
        self.action_loss = action_loss
        self.loss = loss
        self.lr = 0.001
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)


def main():
    vgg19_path = FLAGS.vgg19_path
    data_path = FLAGS.data_path
    output_dir = FLAGS.model_path

    NUM_ITERS = 100000

    with tf.variable_scope('model', reuse=None) as training_scope:


        images, angles, joint_velocity, endeffector_poses, end_effector_velocity, use_frames, \
            final_endeffector_poses = read_tf_record(data_path + 'train.tfrecords')

        robot_configs_batch = tf.concat([angles, endeffector_poses], 1)

        model = Model(vgg19_path, images, robot_configs_batch, end_effector_velocity, use_frames,
                      final_endeffector_poses)

    with tf.variable_scope('val_model', reuse=None):
        val_images, val_angles, val_joint_velocities, val_endeffector_poses, val_endeffector_vel, \
        val_use_frames, val_final_eeps = read_tf_record(data_path + 'test.tfrecords', d_append='test', rng=0)

        val_robot_configs_batch = tf.concat([val_angles, val_endeffector_poses], 1)

        val_model = Model(vgg19_path, val_images, val_robot_configs_batch, val_endeffector_vel, val_use_frames,
                          val_final_eeps, training_scope)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
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
            action_loss = 0
            eep_loss = 0
            count_first = 0
            for t_itr in range(30):
                curr_val_loss, curr_action_loss, curr_eep_loss = sess.run([val_model.loss,
                                                                           val_model.action_loss,
                                                                           val_model.eep_loss])
                val_loss += curr_val_loss / 30.0
                action_loss += curr_action_loss / 30.0
                if curr_eep_loss > 0:
                    eep_loss += curr_eep_loss
                    count_first += 1

            val_summary = tf.Summary()
            val_summary.value.add(tag="val_model/loss", simple_value=val_loss)
            val_summary.value.add(tag="val_model/action_loss", simple_value=action_loss)
            if eep_loss > 0:
                eep_loss /= count_first
                val_summary.value.add(tag="val_model/eep_loss", simple_value=eep_loss)

            summary_writer.add_summary(val_summary, itr)
            print 'Validation Loss:', val_loss

        train_loss, action_loss, eep_loss, _ = sess.run([model.loss, model.action_loss, model.eep_loss, model.train_op])

                                            #feed_dict)
        summary_str = tf.Summary()
        summary_str.value.add(tag="train_model/loss", simple_value=train_loss)
        summary_str.value.add(tag="train_model/action_loss", simple_value=action_loss)
        if eep_loss > 0:
            summary_str.value.add(tag="train_model/eep_loss", simple_value=eep_loss)

        if itr % 10 == 0:
            summary_writer.add_summary(summary_str, itr)
        if itr % 100 == 0:
            print 'Cost', train_loss, 'On iter', itr
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
