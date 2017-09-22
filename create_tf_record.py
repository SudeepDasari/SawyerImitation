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
    img =  cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    return img


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def write_tf_records(images, angles, velocities, ef_poses, filepath):

    print 'Writing', filepath
    writer = tf.python_io.TFRecordWriter(filepath)

    feature = {}
    s_order = np.random.choice(len(images), len(images), replace = False)
    num_train = int(s_order.shape[0] * 0.9)

    print 'Train:', num_train, 'Test', s_order.shape[0] - num_train

    for traj_iter in s_order[:num_train]:
        print 'Outputting train traj', traj_iter

        image_raw = images[traj_iter].astype(np.uint8)
        image_raw = image_raw.tostring()

        feature['train/image'] = _bytes_feature(image_raw)
        feature['train/angle'] = _float_feature(angles[traj_iter].flatten().tolist())
        feature['train/velocity'] = _float_feature(velocities[traj_iter].flatten().tolist())
        feature['train/endeffector_pos'] = _float_feature(ef_poses[traj_iter].flatten().tolist())

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())

    feature = {}
    for traj_iter in s_order[num_train:]:
        print 'Outputting test traj', traj_iter

        image_raw = images[traj_iter].astype(np.uint8)
        image_raw = image_raw.tostring()

        feature['test/image'] = _bytes_feature(image_raw)
        feature['test/angle'] = _float_feature(angles[traj_iter].flatten().tolist())
        feature['test/velocity'] = _float_feature(velocities[traj_iter].flatten().tolist())
        feature['test/endeffector_pos'] = _float_feature(ef_poses[traj_iter].flatten().tolist())

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
    writer.close()


def main():
    data_path = FLAGS.data_path
    out_path = FLAGS.out_path
    groups = [x for x in os.listdir(data_path) if 'traj_group' in x]

    for traj_group in groups:
        print 'reading in traj_group', traj_group
        group_path = data_path + '/' + traj_group
        group_out = out_path + '/traj_group'+'_record.tfrecords'

        trajs = [x for x in os.listdir(group_path) if 'traj' in x]
        trajs = sorted(trajs, key = lambda x: int(x.split('traj')[1]))
        images, angles, velocities, ef_poses = [], [], [], []

        for traj in trajs:
            print 'reading in traj', traj
            traj_path = group_path + '/' + traj

            traj_images = []
            image_files = glob.glob(traj_path + '/images/*.jpg')

            for img_path in image_files:
                img = load_image(img_path)
                traj_images.append(img)

            pkl_path = glob.glob(traj_path + '/*.pkl')[0]

            sawyer_data = cPickle.load(open(pkl_path, 'rb'))
            joint_angles = sawyer_data['jointangles']
            joint_velocities = sawyer_data['jointvelocities']
            endeffector_pos = sawyer_data['endeffector_pos']

            stacked = np.stack(traj_images, axis = 0)
            images.append(stacked)
            angles.append(joint_angles)
            velocities.append(joint_velocities)
            ef_poses.append(endeffector_pos)

        write_tf_records(images, angles, velocities, ef_poses, group_out)


if __name__ == '__main__':
    main()
