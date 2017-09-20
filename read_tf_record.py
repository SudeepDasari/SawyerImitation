import tensorflow as tf
import numpy as np
from tensorflow.python.platform import gfile
from tensorflow.python.platform import flags
import os

FLAGS = flags.FLAGS
flags.DEFINE_string('data_path', './', 'path to tfrecords directory')

NUM_FRAMES = 60
NUM_JOINTS = 7


def read_tf_record(training=True):
    file_names = gfile.Glob(os.path.join(FLAGS.data_path, '*'))
    if not file_names:
        raise RuntimeError('No data_files files found.')

    filename_queue = tf.train.string_input_producer(file_names, shuffle=True)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    image_seq, angles_seq, velocities_seq, endeffector_pos_seq = [], [], [], []

    image_name = 'image'
    angle_name = 'angle'
    velocity_name = 'velocity'
    endeffector_pos_name = 'endeffector_pos'

    features = {
        image_name: tf.FixedLenFeature([NUM_FRAMES], tf.string),
        angle_name: tf.FixedLenFeature([NUM_JOINTS], tf.float32),
        velocity_name: tf.FixedLenFeature([NUM_JOINTS], tf.float32),
        endeffector_pos_name: tf.FixedLenFeature([3], tf.float32),
    }

    features = tf.parse_single_example(serialized_example, features=features)
    print features[image_name].shape
    print features[angle_name].shape
    print features[velocity_name].shape
    print features[endeffector_pos_name].shape


if __name__ == '__main__':
    read_tf_record()
