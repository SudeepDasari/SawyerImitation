import tensorflow as tf
import numpy as np
from imitation_learning import ImitationLearningModel as Model
from tensorflow.python.platform import flags


FLAGS = flags.FLAGS
flags.DEFINE_string('vgg19_path', './', 'path to npy file')
flags.DEFINE_string('data_path', './', 'path to tfrecords file')


def main():
    vgg_path = FLAGS.vgg19_path
    data_path = FLAGS.data_path

    NUM_ITERS = 1000

if __name__ == '__main__':
    main()