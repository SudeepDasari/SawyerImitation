import tensorflow as tf
import numpy as np
import argparse
import rospy
from sensor_msgs.msg import Image as Image_msg
import cv2
from cv_bridge import CvBridge, CvBridgeError
from intera_interface import CHECK_VERSION
import intera_interface
from berkeley_sawyer.srv import *
from tensorflow.python.platform import flags
import pdb

from robot_controller import RobotController
from recorder.robot_recorder import RobotRecorder
from setup_predictor import setup_predictor


class SawyerImitation(object):
    def __init__(self, model_path, vgg19_path):

        print("Initializing node... ")
        rospy.init_node("sawyer_imitation")

        self.ctrl = RobotController()

        self.recorder = RobotRecorder(save_dir='',
                                      seq_len=60,
                                      use_aux=False,
                                      save_video=True,
                                      save_actions=False,
                                      save_images=False)

        self.action_interval = 20 #Hz
        self.action_sequence_length = 20
        self.traj_duration = self.action_sequence_length * self.action_interval
        self.action_rate = rospy.Rate(self.action_interval)
        self.control_rate = rospy.Rate(20)
        self.predictor = setup_predictor(model_path, vgg19_path)

        rospy.on_shutdown(self.ctrl.clean_shutdown)

        self.ctrl.set_neutral()

    def query_action(self):
        image = cv2.resize(self.recorder.ltob.img_cropped, (224, 224), interpolation=cv2.INTER_AREA)
        robot_configs = np.concatenate((self.recorder.get_joint_angles(), self.recorder.get_endeffector_pos()))

        action, predicted_eep = self.predictor(image, robot_configs)
        print 'action vector: ', action
        print 'predicted end effector pose: ', predicted_eep
        return action

    def apply_action(self, action):
        try:
            self.ctrl.set_joint_velocities(action)
        except OSError:
            rospy.logerr('collision detected, stopping trajectory, going to reset robot...')
            rospy.sleep(.5)
        if self.ctrl.limb.has_collided():
            rospy.logerr('collision detected!!!')
            rospy.sleep(.5)

    def run_trajectory(self):
        self.ctrl.set_neutral()

        step = 0
        while step < self.action_sequence_length:
            action = self.query_action()

            # self.apply_action(action)

            step += 1

if __name__ == '__main__':
    FLAGS = flags.FLAGS
    flags.DEFINE_string('model_path', './', 'path to output model/stats')
    flags.DEFINE_string('vgg19_path', './', 'path to npy file')
    d = SawyerImitation(FLAGS.model_path, FLAGS.vgg19_path)
    pdb.set_trace()
    d.run_trajectory()
