import tensorflow as tf
import numpy as np
from tensorflow.python.platform import gfile
from tensorflow.python.platform import flags
import cv2
import os
import cPickle
import glob
from ee_velocity_compute import EE_Calculator
FLAGS = flags.FLAGS
flags.DEFINE_string('data_path', './', 'path to trajectory folders')
flags.DEFINE_string('out_path', './', 'output file directory')


def load_image(path):
    img = cv2.imread(path)
    img = img[:-150,275:-150,:]
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    cv2.imshow('img', img)
    cv2.waitKey(1)
    return img


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def write_tf_records(images, angles, velocities, ef_poses, filepath):
    filpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), filepath)
    print 'Writing', filepath + 'train.tfrecords'
    writer = tf.python_io.TFRecordWriter(filepath + 'train.tfrecords')

    s_order = np.random.choice(200, 200, replace = False)

    for traj_iter in s_order:
        print 'Outputting train traj', traj_iter

        image_raw = images[traj_iter].astype(np.uint8)
        image_raw = image_raw.tostring()
        feature = {}
        feature['train/image'] = _bytes_feature(image_raw)
        feature['train/angle'] = _float_feature(angles[traj_iter].astype(np.float32).flatten().tolist())
        feature['train/velocity'] = _float_feature(velocities[traj_iter].astype(np.float32).flatten().tolist())
        feature['train/endeffector_pos'] = _float_feature(ef_poses[traj_iter].astype(np.float32).flatten().tolist())

        assert 'train/image' in feature, "Missing image entry"
        assert 'train/angle' in feature, "Missing angle entry"
        assert 'train/velocity' in feature, "Missing velocity entry"
        assert 'train/endeffector_pos' in feature, "Missing end effector entry"

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
    writer.close()

    writer = tf.python_io.TFRecordWriter(filepath + 'test.tfrecords')
    print 'Writing', filepath + 'test.tfrecords'
    s_order = np.random.choice(20, 20, replace=False) + 200

    for traj_iter in s_order:
        print 'Outputting test traj', traj_iter

        image_raw = images[traj_iter].astype(np.uint8)
        image_raw = image_raw.tostring()
        feature = {}

        feature['test/image'] = _bytes_feature(image_raw)
        feature['test/angle'] = _float_feature(angles[traj_iter].astype(np.float32).flatten().tolist())
        feature['test/velocity'] = _float_feature(velocities[traj_iter].astype(np.float32).flatten().tolist())
        feature['test/endeffector_pos'] = _float_feature(ef_poses[traj_iter].astype(np.float32).flatten().tolist())

        assert 'test/image' in feature, "Missing image entry"
        assert 'test/angle' in feature, "Missing angle entry"
        assert 'test/velocity' in feature, "Missing velocity entry"
        assert 'test/endeffector_pos' in feature, "Missing end effector entry"


        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
    writer.close()


def main():
    calc = EE_Calculator()
    data_path = FLAGS.data_path
    out_path = FLAGS.out_path
    groups = [x for x in os.listdir(data_path) if 'traj_group' in x]

    for traj_group in groups:
        print 'reading in traj_group', traj_group
        group_path = data_path + '/' + traj_group
        group_out = out_path + '/traj_group'+'_record'

        trajs = [x for x in os.listdir(group_path) if 'traj' in x]
        trajs = sorted(trajs, key = lambda x: int(x.split('traj')[1]))
        images, angles, velocities, ef_poses = [], [], [], []

        for traj in trajs:
            print 'reading in traj', traj
            traj_path = group_path + '/' + traj

            traj_images = []
            image_files = glob.glob(traj_path + '/images/*.jpg')
            image_files = sorted([(i, int(i.split('main_full_cropped_')[1][2:4])) for i in image_files], key = lambda x: x[1])

            # print image_files
            for img_path in image_files:
                img = load_image(img_path[0])
                traj_images.append(img)

            pkl_path = glob.glob(traj_path + '/*.pkl')[0]

            sawyer_data = cPickle.load(open(pkl_path, 'rb'))
            joint_angles = sawyer_data['jointangles']
            joint_velocities = sawyer_data['jointvelocities']

            endeffector_pos = sawyer_data['endeffector_pos']


            out_velocities = np.zeros((joint_velocities.shape[0], 13), np.float32)
            out_ef = np.zeros((endeffector_pos.shape[0], 7), np.float32)

            for i in range(endeffector_pos.shape[0]):
                ja = joint_angles[i, :]
                jv = joint_velocities[i, :]
                out_velocities[i, :7] = jv

                jac = calc.jacobian(ja)

                ee_velocity = jac.dot(jv.reshape((-1, 1)))
                ee_pose = calc.forward_position_kinematics(ja)

                out_ef[i, :]  = ee_pose
                out_velocities[i, 7:] = ee_velocity.reshape(-1)

            stacked = np.stack(traj_images, axis = 0)

            images.append(stacked)
            angles.append(joint_angles)
            velocities.append(out_velocities)
            ef_poses.append(out_ef)

        write_tf_records(images, angles, velocities, ef_poses, group_out)


if __name__ == '__main__':
    main()
