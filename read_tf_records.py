import tensorflow as tf
import numpy as np
from tensorflow.python.platform import gfile
import cv2
import os


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def write_tf_records(images, actions, states, filepath):
    filename = os.path.join(dir, filepath + '/tfrecords/demo_examples.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)

    feature = {}

    for ex in range(images.shape[0]):
        sequence_length = 60

        for tind in range(sequence_length):
            image_raw = (images[ex,tind]*255.).astype(np.uint8)
            image_raw = image_raw.tostring()

            feature[str(tind) + '/action'] = _float_feature(actions[ex, tind].tolist())
            feature[str(tind) + '/endeffector_pos'] = _float_feature(states[ex,tind].tolist())
            feature[str(tind) + '/image'] = _bytes_feature(image_raw)

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())

    writer.close()