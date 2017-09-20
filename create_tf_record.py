import tensorflow as tf
import numpy as np
from tensorflow.python.platform import gfile
from tensorflow.python.platform import flags
import cv2
import os
import cPickle
import glob

FLAGS = flags.FLAGS
flags.DEFINE_string('data_path', './', 'path to trajectory folders')
flags.DEFINE_string('out_path', './', 'output file directory')


def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def write_tf_records(images, angles, velocities, ef_poses, filepath):

    print 'Writing', filepath
    writer = tf.python_io.TFRecordWriter(filepath)

    feature = {}

    for traj_iter in range(len(images)):
        print 'On traj', traj_iter

        image_raw = images[traj_iter].astype(np.uint8)
        image_raw = image_raw.tostring()

        feature[str(traj_iter) + '/image'] = _bytes_feature(image_raw)
        feature[str(traj_iter) + '/angle'] = _float_feature(angles[traj_iter].flatten().tolist())
        feature[str(traj_iter) + '/velocity'] = _float_feature(velocities[traj_iter].flatten().tolist())
        feature[str(traj_iter) + '/endeffector_pos'] = _float_feature(ef_poses[traj_iter].flatten().tolist())

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())

    writer.close()


def main():
    data_path = FLAGS.data_path
    out_path = FLAGS.out_path
    groups = [x for x in os.listdir(data_path) if 'traj_group' in x]

    for traj_group in groups:
        group_path = data_path + '/' + traj_group
        group_out = out_path + '/traj_group'+'_record.tfrecords'

        trajs = [x for x in os.listdir(group_path) if 'traj' in x]
        trajs = sorted(trajs, key = lambda x: int(x.split('traj')[1]))
        images, angles, velocities, ef_poses = [], [], [], []

        for traj in trajs:
            traj_path = group_path + '/' + traj

            traj_images = []
            image_files = glob.glob(traj_path + '/images/*.png')

            for i in range(len(image_files)):
                im_path = [x for x in image_files if 'im%d' % (i) in x][0]
                traj_images.append(load_image(im_path))

            pkl_path = glob.glob(traj_path + '/*.pkl')[0]

            sawyer_data = cPickle.load(open(pkl_path, 'rb'))
            joint_angles = sawyer_data['jointangles']
            joint_velocities = sawyer_data['jointvelocities']
            endeffector_pos = sawyer_data['endeffector_pos']

            stacked = np.stack(traj_images, axis = 3)

            images.append(stacked)
            angles.append(joint_angles)
            velocities.append(joint_velocities)
            ef_poses.append(endeffector_pos)

	print len(images), images[0].shape, len(angles), len(angles[0]), len(angles[0][0])

        write_tf_records(images, angles, velocities, ef_poses, group_out)


if __name__ == '__main__':
    main()
