import tensorflow as tf
import numpy as np
from imitation_learning import ImitationLearningModel
from read_tf_record import read_tf_record
from tensorflow.python.platform import flags



FLAGS = flags.FLAGS
flags.DEFINE_string('vgg19_path', './', 'path to npy file')
flags.DEFINE_string('data_path', './', 'path to tfrecords file')
flags.DEFINE_string('model_path', './', 'path to output model/stats')

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
    def __init__(self, vgg19_path, images_batch, robot_configs_batch, actions_batch):
        self.m =ImitationLearningModel(vgg19_path, images_batch, robot_configs_batch, actions_batch)
        self.m.build()

        loss = mean_squared_error(self.m.actions, self.m.predicted_actions)
        self.loss = loss
        self.lr = 0.001
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
        self.summ_op = tf.summary.merge([tf.summary.scalar('loss', loss)])

def main():
    vgg19_path = FLAGS.vgg19_path
    data_path = FLAGS.data_path
    output_dir = FLAGS.model_path

    NUM_ITERS = 1000

    images_batch, angles_batch, velocities_batch, endeffector_poses_batch = read_tf_record(data_path)
    if int(tf.__version__[0]) >= 1.0:
        robot_configs_batch = tf.concat([angles_batch, endeffector_poses_batch], 1)
    else:
        robot_configs_batch = tf.concat(1, [angles_batch, endeffector_poses_batch])

    actions_batch = velocities_batch

    model = Model(vgg19_path, images_batch, robot_configs_batch, actions_batch)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    # Make training session.
    sess = tf.InteractiveSession(config=tf.ConfigProto(device_count = {'GPU': 0}))
    summary_writer = tf.summary.FileWriter(output_dir, graph=sess.graph, flush_secs=10)

    # feed_dict = {
    #     model.iter_num: np.float32(itr),
    #     model.lr: conf['learning_rate'],
    # }
    tf.train.start_queue_runners(sess)
    sess.run(tf.global_variables_initializer())
    print 'session begun'

    itr = 0

    cost, _, summary_str = sess.run([model.loss, model.train_op, model.summ_op])
                                    #feed_dict)
    print cost
    print summary_str


if __name__ == '__main__':
    main()