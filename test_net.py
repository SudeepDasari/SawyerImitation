import tensorflow as tf
import numpy as np
import cPickle as pickle
from imitation_learning import ImitationLearningModel
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS
flags.DEFINE_string('vgg19_path', './', 'path to npy file')
flags.DEFINE_string('data_path', './', 'path to tfrecords file')
flags.DEFINE_string('model_path', './', 'path to output model/stats')
NUM_FRAMES = 60
NUM_JOINTS = 7
STATE_DIM = 3
IMG_WIDTH = 224
IMG_HEIGHT = 224
COLOR_CHANNELS = 3


def read_tf_record(data_path, d_append = 'test'):
    #gets data path
    # data_path = FLAGS.data_path

    #setup tf session
    # with tf.Session() as sess:
        #feature dictionary definition
    feature = {
        d_append+'/image': tf.FixedLenFeature([IMG_WIDTH * IMG_HEIGHT * NUM_FRAMES * 3], tf.string),
        d_append+'/angle': tf.FixedLenFeature([NUM_JOINTS * NUM_FRAMES], tf.float32),
        d_append+'/velocity': tf.FixedLenFeature([NUM_JOINTS * NUM_FRAMES], tf.float32),
        d_append+'/endeffector_pos': tf.FixedLenFeature([STATE_DIM * NUM_FRAMES], tf.float32)
    }


    # Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer([data_path], shuffle = False)
    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)


    # Convert the image and robot data from string back to the numbers
    image = tf.decode_raw(features[d_append+'/image'], tf.uint8)
    angle = tf.reshape(features[d_append+'/angle'], shape=[NUM_FRAMES, NUM_JOINTS])
    velocity = tf.reshape(features[d_append+'/velocity'], shape=[NUM_FRAMES, NUM_JOINTS])
    endeffector_pos = tf.reshape(features[d_append+'/endeffector_pos'], shape=[NUM_FRAMES, STATE_DIM])

    # Reshape image data into original video
    image = tf.reshape(image, [NUM_FRAMES, IMG_HEIGHT, IMG_WIDTH, COLOR_CHANNELS])


    # Creates batches by randomly shuffling tensors. each training example is (image,velocity) pair

    images, angles, velocities, endeffector_poses = tf.train.batch([image, angle, velocity, endeffector_pos],
                                                                           batch_size=30, capacity=600, num_threads=30, enqueue_many=True)
    return images, angles, velocities, endeffector_poses


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
        self.summ_op = tf.summary.merge([tf.summary.scalar('test_loss', loss)])


def main():
    vgg19_path = FLAGS.vgg19_path
    data_path = FLAGS.data_path
    output_dir = FLAGS.model_path


    images_batch, angles_batch, velocities_batch, endeffector_poses_batch = read_tf_record(data_path)

    if int(tf.__version__[0]) >= 1.0:
        robot_configs_batch = tf.concat([angles_batch, endeffector_poses_batch], 1)
    else:
        robot_configs_batch = tf.concat(1, [angles_batch, endeffector_poses_batch])

    actions_batch = velocities_batch

    model = Model(vgg19_path, images_batch, robot_configs_batch, actions_batch)

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
        real_actions, predicted_actions, loss, summary_str = sess.run([model.m.actions, model.m.predicted_actions, model.loss, model.summ_op])
        print 'loss is', loss
        summary_writer.add_summary(summary_str, i)

        reals.append(real_actions)
        predictions.append(predicted_actions)

    pickle.dump({'reals':reals, 'predictions':predictions}, open('real_vs_pred.pkl', 'wb'))

    coord.request_stop()
    coord.join(threads)


if __name__ == '__main__':
    main()