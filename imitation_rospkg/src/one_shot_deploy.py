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
import os


import inverse_kinematics



class Traj_aborted_except(Exception):
    pass

class SawyerOneShot(object):
    EE_STEPS = 2
    ACTION_SEQUENCE_LENGTH = 3
    SPRING_STIFF = 220
    CONTROL_RATE = 20

    CROP_H_MIN = 0
    CROP_H_MAX = -150
    CROP_W_MIN = 190
    CROP_W_MAX = -235
    def __init__(self, meta_path, norm_path, recording_path):
        self.ctrl = RobotController(self.CONTROL_RATE)

        self.recorder = RobotRecorder(save_dir='',
                                      use_aux=False,
                                      save_actions=False,
                                      save_images=False)

        self.control_rate = rospy.Rate(self.CONTROL_RATE)
        self.predictor = setup_predictor(meta_path, norm_path, recording_path, self.CROP_H_MIN, self.CROP_H_MAX, self.CROP_W_MIN, self.CROP_W_MAX)

        self.move_netural()

    def query_action(self):
        image = cv2.resize(self.recorder.ltob.img_cv2[self.CROP_H_MIN:self.CROP_H_MAX, self.CROP_W_MIN:self.CROP_W_MAX, :], (100, 100), interpolation=cv2.INTER_AREA)[:, :, ::-1]

        robot_configs = self.recorder.get_endeffector_pos()

        if image is None or robot_configs is None:
            return None

        action, ee = self.predictor(image, robot_configs)
        self.img_stack.append(image)

        return action, ee

    def move_to(self, des_pos):
        desired_pose = inverse_kinematics.get_pose_stamped(des_pos[0],
                                                           des_pos[1],
                                                           des_pos[2],
                                                           inverse_kinematics.EXAMPLE_O)
        start_joints = self.ctrl.limb.joint_angles()
        try:
            print 'start ik'
            des_joint_angles = inverse_kinematics.get_joint_angles(desired_pose, seed_cmd=start_joints,
                                                                   use_advanced_options=True)
            print 'after ik'
        except ValueError:
            rospy.logerr('no inverse kinematics solution found, '
                         'going to reset robot...')
            current_joints = self.ctrl.limb.joint_angles()
            des_joint_angles = current_joints
            # raise Traj_aborted_except('raising Traj_aborted_except')



        self.ctrl.imp_ctrl_release_spring(self.SPRING_STIFF)
        self.ctrl.move_with_impedance_sec(des_joint_angles)
        # self.ctrl.set_joint_positions(des_joint_angles)




    def move_netural(self):
        # self.ctrl.set_neutral()
        self.ctrl.imp_ctrl_release_spring(280)
        self.ctrl.set_neutral_with_impedance()

    def run_trajectory(self):
        self.start = self.recorder.get_endeffector_pos()
        print 'actual end eep', self.start
        self.move_netural()
        self.img_stack = []

        step = 0
        actions = []

        for i in range(self.EE_STEPS):
            current_eep = self.recorder.get_endeffector_pos()
            eep_diff_action, pred_final = self.query_action()
            current_eep[:2] = pred_final
            self.move_to(current_eep[:3])

        while step < self.ACTION_SEQUENCE_LENGTH:
            self.control_rate.sleep()
            current_eep = self.recorder.get_endeffector_pos()
            print 'step', step

            eep_diff_action, pred_final = self.query_action()
            # print 'ee_diff_action', eep_diff_action

            # print 'before', current_eep[:3]
            current_eep[:3] += 0.05 * eep_diff_action
            current_eep[2] = max(current_eep[2], 0.2)


            self.move_to(current_eep[:3])

            step += 1

        print 'end', self.recorder.get_endeffector_pos()
        clip = mpy.ImageSequenceClip([i for i in self.img_stack], fps=20)
        clip.write_gif('test_frames.gif')



if __name__ == '__main__':
    # FLAGS = flags.FLAGS
    # flags.DEFINE_string('model_path', './', 'path to output model/stats')
    # flags.DEFINE_string('vgg19_path', './', 'path to npy file')

    recording_path = os.path.expanduser('~/human_demos_nov/traj0/')
    d = SawyerOneShot('to_test/model_human_maml/place_sawyer_maml_3_layers_200_dim_1d_conv_ee_3_32_act_2_32_20x1_filters_64_128_3x3_filters_human_light_68k.meta',
                      'to_test/model_human_maml/scale_and_bias_place_sawyer_kinect_view_human.pkl', recording_path)
    while True:
        pdb.set_trace()
        d.run_trajectory()