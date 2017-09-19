import tensorflow as tf
import numpy as np
from tensorflow.python.platform import gfile
import cv2
import os
import cPickle
import glob


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def write_tf_records(images, angles, velocities, states, filepath):
    filename = os.path.join(dir, filepath + '/tfrecords/demo_examples.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)

    feature = {}

    for ex in range(images.shape[0]):
        image_raw = (images[ex]*255.).astype(np.uint8)
        image_raw = image_raw.tostring()

        feature['angle'] = _float_feature(angles[ex].tolist())
        feature['velocity'] = _float_feature(velocities[ex].tolist())
        feature['endeffector_pos'] = _float_feature(states[ex].tolist())
        feature['image'] = _bytes_feature(image_raw)

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())

    writer.close()


def main():
    images, angles, velocities, states = [], [], [], []

    # for i in range(100):
    #     for filename in glob.glob('traj{}/images/*.png'.format(i)):
    #         images += []
    #
    #     with open('joint_traj{}.pkl'.format(i), 'r') as f:
    #         data = cPickle.load(f)
    #     angles += data['jointangles']
    #     velocities += data['jointvelocities']
    #     states += data['endeffector_pos']
    #
    # filepath = ''
    # write_tf_records(images, angles, velocities, states, filepath)


if __name__ == '__main__':
    main()