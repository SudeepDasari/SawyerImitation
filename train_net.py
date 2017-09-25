import tensorflow as tf
import numpy as np
from imitation_learning import ImitationLearningModel
from read_tf_record import read_tf_record
from tensorflow.python.platform import flags



FLAGS = flags.FLAGS
flags.DEFINE_string('vgg19_path', './', 'path to npy file')
flags.DEFINE_string('data_path', './', 'path to tfrecords file')
flags.DEFINE_string('model_path', './', 'path to output model/stats')
flags.DEFINE_bool('test', False, 'run trained model on test data')

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
    def __init__(self, vgg19_path, images_batch, robot_configs_batch, actions_batch, training=True):
        self.m =ImitationLearningModel(vgg19_path, images_batch, robot_configs_batch, actions_batch)
        self.m.build()

        loss = mean_squared_error(self.m.actions, self.m.predicted_actions)
        self.loss = loss
        self.lr = 0.001
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
        if training:
            self.summ_op = tf.summary.merge([tf.summary.scalar('loss', loss)])
        else:
            self.summ_op = tf.summary.merge([tf.summary.scalar('test_loss', loss)])

def main():
    vgg19_path = FLAGS.vgg19_path
    data_path = FLAGS.data_path
    output_dir = FLAGS.model_path

    NUM_ITERS = 18000

    if FLAGS.test:
        images_batch, angles_batch, velocities_batch, endeffector_poses_batch = read_tf_record(data_path, d_append='test')
    else:
        images_batch, angles_batch, velocities_batch, endeffector_poses_batch = read_tf_record(data_path)
    if int(tf.__version__[0]) >= 1.0:
        robot_configs_batch = tf.concat([angles_batch, endeffector_poses_batch], 1)
    else:
        robot_configs_batch = tf.concat(1, [angles_batch, endeffector_poses_batch])

    actions_batch = velocities_batch

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

    if FLAGS.test:
        model = Model(vgg19_path, images_batch, robot_configs_batch, actions_batch, training=False)
        ckpt = tf.train.get_checkpoint_state(output_dir + '/modelfinal')
        saver.restore(sess, ckpt.model_checkpoint_path)
        for i in range(20):
            loss, summary_str = sess.run([model.loss, model.summ_op])
            print 'loss is', loss
            summary_writer.add_summary(summary_str, i)

    else:


        model = Model(vgg19_path, images_batch, robot_configs_batch, actions_batch)

        for itr in range(NUM_ITERS):
            cost, _, summary_str = sess.run([model.loss, model.train_op, model.summ_op])
                                            #feed_dict)
            if itr % 100 == 0:
                print 'Cost', cost, 'On iter', itr

            if itr % 180 == 0 and itr > 0:

                saver.save(sess, output_dir + '/model'+ str(itr))

            summary_writer.add_summary(summary_str, itr)

        saver.save(sess, output_dir + '/modelfinal')

    coord.request_stop()
    coord.join(threads)



if __name__ == '__main__':
    main()