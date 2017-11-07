#!/usr/bin/env python

import numpy as np

import rospy
import pdb
from intera_interface import CHECK_VERSION
import intera_interface
from berkeley_sawyer.srv import *
from tensorflow.python.platform import flags
import matplotlib.pyplot as plt
import pdb
import moviepy.editor as mpy
from robot_controller import RobotController
from recorder.robot_recorder import RobotRecorder
from setup_predictor import setup_MAML_predictor_human as setup_predictor
import cv2

class SawyerOneShot(object):
    def __init__(self, meta_path, norm_path, recording_path):
        self.ctrl = RobotController()

        self.recorder = RobotRecorder(save_dir='',
                                      seq_len=60,
                                      use_aux=False,
                                      save_video=True,
                                      save_actions=False,
                                      save_images=False)

        self.action_interval = 20 #Hz
        self.action_sequence_length = 40

        self.traj_duration = self.action_sequence_length * self.action_interval
        self.action_rate = rospy.Rate(self.action_interval)
        self.control_rate = rospy.Rate(20)
        self.predictor = setup_predictor(meta_path, norm_path, recording_path)
        self.save_ctr = 0
        self.ctrl.set_neutral()
        self.s = 0

    def query_action(self):
        image = cv2.resize(self.recorder.ltob.img_cv2[:-150, 225:-225, :], (100, 100), interpolation=cv2.INTER_AREA)[:, :, ::-1]

        robot_configs = np.concatenate((self.recorder.get_joint_angles(), self.recorder.get_endeffector_pos()))

        if image is None or robot_configs is None:
            return None

        action = self.predictor(image, robot_configs)
        self.img_stack.append(image)

        self.s += 1
        # if self.s <=5:
        #     action[np.abs(action) < 0.05] *= 15

        print 'action vector: ', action
        # print 'predicted end effector pose: ', predicted_eep
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
        self.start = self.recorder.get_endeffector_pos()
        print 'actual end eep', self.start
        self.ctrl.set_neutral()
        self.img_stack = []

        step = 0
        actions = []
        while step < self.action_sequence_length:
            self.control_rate.sleep()
            action = self.query_action()

            action_dict = dict(zip(self.ctrl.joint_names, action))
            # for i in range(len(self.ctrl.joint_names)):
            #     action_dict[self.ctrl.joint_names[i]] = action[i]
                # print 'key', self.ctrl.joint_names[i], 'value', action_dict[self.ctrl.joint_names[i]]
            actions.append(action)
            self.apply_action(action_dict)

            step += 1
        print 'end', self.recorder.get_endeffector_pos()
        clip = mpy.ImageSequenceClip([i for i in self.img_stack], fps=20)
        clip.write_gif('test_frames.gif')



if __name__ == '__main__':
    # FLAGS = flags.FLAGS
    # flags.DEFINE_string('model_path', './', 'path to output model/stats')
    # flags.DEFINE_string('vgg19_path', './', 'path to npy file')

    recording_path = '~/demos_test0/human/object2/'
    d = SawyerOneShot('place_sawyer_maml_human/place_sawyer_maml_human_act_ee_2_32_10x1_1d_conv_50k.meta',
                      'place_sawyer_maml_human/scale_and_bias_place_sawyer_human.pkl', recording_path)
    while True:
        pdb.set_trace()
        d.run_trajectory()