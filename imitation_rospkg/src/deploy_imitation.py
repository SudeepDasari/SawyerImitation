#!/usr/bin/env python
import numpy as np
import argparse
import rospy
import pdb
from intera_interface import CHECK_VERSION
import intera_interface
from berkeley_sawyer.srv import *
from tensorflow.python.platform import flags
import matplotlib.pyplot as plt

from robot_controller import RobotController
from recorder.robot_recorder import RobotRecorder
from setup_predictor import setup_predictor
import cv2

class SawyerImitation(object):
    def __init__(self, model_path, vgg19_path):

        self.ctrl = RobotController()


        self.recorder = RobotRecorder(save_dir='',
                                      seq_len=60,
                                      use_aux=False,
                                      save_video=True,
                                      save_actions=False,
                                      save_images=False)

        self.action_interval = 20 #Hz
        self.action_sequence_length = 15
        self.traj_duration = self.action_sequence_length * self.action_interval
        self.action_rate = rospy.Rate(self.action_interval)
        self.control_rate = rospy.Rate(20)
        self.predictor = setup_predictor(model_path, vgg19_path)
        self.save_ctr = 0
        self.ctrl.set_neutral()

    def query_action(self):
        image = self.recorder.ltob.img_cropped
        cv2.imwrite('test' + str(self.save_ctr)+'.png', image)
        self.save_ctr += 1
        # cv2.imshow('image', image)
        # cv2.waitKey(0)
        robot_configs = np.concatenate((self.recorder.get_endeffector_pos(), self.recorder.get_joint_angles()))

        if image is None or robot_configs is None:
            return None

        # print 'prediction', image.shape, robot_configs.shape
        action_vec, predicted_eep = self.predictor(image, robot_configs)
        return action_vec, predicted_eep

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
        print 'start', self.recorder.get_endeffector_pos()
        self.ctrl.set_neutral()

        step = 0
        actions = []
        while step < self.action_sequence_length:
            self.control_rate.sleep()
            action, predicted_eep = self.query_action()
            print 'predicted_eep', predicted_eep

            action_dict = {}
            for i in range(len(self.ctrl.joint_names)):
                action_dict[self.ctrl.joint_names[i]] = action[i]
                # print 'key', self.ctrl.joint_names[i], 'value', action_dict[self.ctrl.joint_names[i]]
            actions.append(action)
            self.apply_action(action_dict)

            step += 1
        print 'end', self.recorder.get_endeffector_pos()

if __name__ == '__main__':
    # FLAGS = flags.FLAGS
    # flags.DEFINE_string('model_path', './', 'path to output model/stats')
    # flags.DEFINE_string('vgg19_path', './', 'path to npy file')
    d = SawyerImitation('modeldata_100_100/modelfinal', 'out/')
    pdb.set_trace()
    d.run_trajectory()
