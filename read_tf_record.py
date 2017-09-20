import tensorflow as tf
import numpy as np
from tensorflow.python.platform import gfile
from tensorflow.python.platform import flags
import os

FLAGS = flags.FLAGS
flags.DEFINE_string('data_path', './', 'path to tfrecords directory')

NUM_FRAMES = 60
NUM_JOINTS = 7
STATE_DIM = 3
IMG_WIDTH = 64
IMG_HEIGHT = 64

BATCH_SIZE = 32

def read_tf_record(training=True):
    file_names = gfile.Glob(os.path.join(FLAGS.data_path, '*'))
    if not file_names:
        raise RuntimeError('No data_files files found.')

    filename_queue = tf.train.string_input_producer(file_names, shuffle=True)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    image_seq, angles_seq, velocities_seq, endeffector_pos_seq = [], [], [], []

    for traj_iter in range(100):

    	image_name = str(traj_iter) + '/image'
    	angle_name = str(traj_iter) + '/angle'
    	velocity_name = str(traj_iter) + '/velocity'
    	endeffector_pos_name = str(traj_iter) + 'endeffector_pos'

    	features = {
            image_name: tf.FixedLenFeature([IMG_WIDTH * IMG_HEIGHT * NUM_FRAMES], tf.string),
            angle_name: tf.FixedLenFeature([NUM_JOINTS * NUM_FRAMES], tf.float32),
            velocity_name: tf.FixedLenFeature([NUM_JOINTS * NUM_FRAMES], tf.float32),
            endeffector_pos_name: tf.FixedLenFeature([STATE_DIM * NUM_FRAMES], tf.float32),
    	}

    	features = tf.parse_single_example(serialized_example, features=features)

	images = tf.decode_raw(features[image_name], tf.uint8)
	images = tf.reshape(images, shape=[NUM_FRAMES, IMG_HEIGHT, IMG_WIDTH])
	image_seq.append(images)

	angles = tf.reshape(features[angle_name], shape=[NUM_FRAMES, NUM_JOINTS])
	angles_seq.append(angles)

	velocities = tf.reshape(features[velocity_name], shape=[NUM_FRAMES, NUM_JOINTS])
	velocities_seq.append(velocities)

	endeffector_poses = tf.reshape(features[endeffector_pos_name], shape=[NUM_FRAMES, STATE_DIM])
	endeffector_pos_seq.append(endeffector_poses)

    image_seq = tf.concat(image_seq, axis=0)
    angles_seq = tf.concat(angles_seq, axis=0)
    velocities_seq = tf.concat(velocities_seq, axis=0)
    endeffector_pos_seq = tf.concat(endeffector_pos_seq, axis=0)
    print image_seq.shape, angles_seq.shape, velocities_seq.shape, endeffector_pos_seq.shape

    num_threads = np.min((BATCH_SIZE, 32))

    image_batch, angles_batch, velocities_batch, endeffector_pos_batch = tf.train.batch(
	[image_seq, angles_seq, velocities_seq, endeffector_pos_seq],
	BATCH_SIZE,
	num_threads=num_threads,
	capacity=100 * BATCH_SIZE,
	enqueue_many=True)
    
    print image_batch.shape, angles_batch.shape, velocities_batch.shape, endeffector_pos_batch.shape
    # return image_batch, angles_batch, velocities_batch, endeffector_pos_batch

if __name__ == '__main__':
    read_tf_record()
