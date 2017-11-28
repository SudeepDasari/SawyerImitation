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
from setup_predictor import setup_MAML_predictor as setup_predictor
import cv2
import os
from keyboard.msg._Key import Key
from ee_velocity_compute import EE_Calculator

import inverse_kinematics



class Traj_aborted_except(Exception):
    pass

class SawyerOneShot(object):
    EE_STEPS = 2
    ACTION_SEQUENCE_LENGTH = 10
    SPRING_STIFF = 240
    CONTROL_RATE = 10000
    PREDICT_RATE = 20

    CROP_H_MIN = 0
    CROP_H_MAX = -150
    CROP_W_MIN = 190
    CROP_W_MAX = -235
    TRAJ_LEN = 40

    Z_SAFETY_THRESH = 0.20
    def __init__(self, meta_path, norm_path):

        self.ctrl = RobotController(self.CONTROL_RATE)
        self.recorder = RobotRecorder(save_dir='',
                                      use_aux=False,
                                      save_actions=False,
                                      save_images=False)

        self.control_rate = rospy.Rate(self.PREDICT_RATE)
        self.predictor = setup_predictor(meta_path, norm_path, self.CROP_H_MIN, self.CROP_H_MAX, self.CROP_W_MIN, self.CROP_W_MAX)

        self.move_netural()
        rospy.Subscriber("/keyboard/keyup", Key, self.keyboard_up_listener)

        self._navigator = intera_interface.Navigator()
        self.demo_collect = self._navigator.register_callback(self.record_demo, 'right_button_ok')
        self.calc = EE_Calculator()

        print 'ONE SHOT READY!'
        self.running = False
        self.demo_imgs = None
        rospy.spin()


    def record_demo(self, value):
        if self.running:
            return
        self.running = True
        self.demo_imgs = []

        traj_ee_pos = np.zeros((self.TRAJ_LEN, 7))
        traj_ee_velocity = np.zeros((self.TRAJ_LEN, 6))
        self.control_rate.sleep()

        for i in range(self.TRAJ_LEN):
            self.control_rate.sleep()
            self.demo_imgs.append(self.recorder.ltob.img_cv2)
            traj_ee_pos[i, :] = self.recorder.get_endeffector_pos()

            angles = self.recorder.get_joint_angles()
            velocities = self.recorder.get_joint_angles_velocity()

            traj_ee_velocity[i, :] = self.calc.jacobian(angles.reshape(-1)).dot(velocities.reshape((-1, 1))).reshape(-1)

        final_ee = np.tile(traj_ee_pos[-1, :2], (40, 1))
        self.record_state = traj_ee_pos
        self.record_action = np.concatenate((traj_ee_velocity[:, :3], final_ee), axis=1)

        self.demo_imgs = np.stack([cv2.resize(img[self.CROP_H_MIN:self.CROP_H_MAX, self.CROP_W_MIN:self.CROP_W_MAX, :],
                                     (100, 100), interpolation=cv2.INTER_AREA)[:, :, ::-1] for img in self.demo_imgs], axis = 0)
        for i in self.demo_imgs:
            cv2.imshow('img', i[:, :, ::-1])
            cv2.waitKey(100)
        cv2.destroyAllWindows()
        self.running = False



    def keyboard_up_listener(self, key_msg):
        if key_msg.code == 99 and not self.running:
            if self.demo_imgs is None:
                print "PLEASE COLLECT A DEMO FIRST"
                return

            self.run_trajectory()
        if key_msg.code == 101 and not self.running:
            rospy.signal_shutdown('User shutdown!')
        if key_msg.code == 110 and not self.running:
            self.move_netural()

    def query_action(self):
        image = cv2.resize(self.recorder.ltob.img_cv2[self.CROP_H_MIN:self.CROP_H_MAX, self.CROP_W_MIN:self.CROP_W_MAX, :], (100, 100), interpolation=cv2.INTER_AREA)[:, :, ::-1]

        robot_configs = self.recorder.get_endeffector_pos()

        if image is None or robot_configs is None:
            return None
        action, ee = self.predictor(self.demo_imgs, self.record_state, self.record_action, image, robot_configs)
        self.img_stack.append(image)

        return action, ee

    def move_to(self, des_pos, interp = True):
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



        # self.ctrl.imp_ctrl_release_spring(self.SPRING_STIFF)
        # self.ctrl.move_with_impedance_sec(des_joint_angles)
        # self.ctrl.set_joint_positions(des_joint_angles)
        if interp:
            self.ctrl.set_joint_positions_interp(des_joint_angles)
        else:
            self.ctrl.limb.set_joint_position_speed(0.15)
            self.ctrl.set_joint_positions(des_joint_angles)



    def move_netural(self):
        self.ctrl.set_neutral()
        # self.ctrl.imp_ctrl_release_spring(280)
        # self.ctrl.set_neutral_with_impedance()

    def run_trajectory(self):
        self.running = True
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

            # current_eep[2] += np.sum(np.abs(current_eep[:2])) * 0.05
            self.move_to(current_eep[:3], i > 0)

        while step < self.ACTION_SEQUENCE_LENGTH:
            self.control_rate.sleep()
            current_eep = self.recorder.get_endeffector_pos()
            print 'step', step

            eep_diff_action, pred_final = self.query_action()
            # print 'ee_diff_action', eep_diff_action

            # print 'before', current_eep[:3]
            current_eep[:3] += 0.05 * eep_diff_action
            #add some fuzz to z when far away to fix joint impedence. pretty hacky for now

            current_eep[2] = max(current_eep[2], self.Z_SAFETY_THRESH)


            self.move_to(current_eep[:3])

            step += 1

        print 'end', self.recorder.get_endeffector_pos()
        clip = mpy.ImageSequenceClip([i for i in self.img_stack], fps=20)
        clip.write_gif('test_frames.gif')
        self.running = False



if __name__ == '__main__':
    # FLAGS = flags.FLAGS
    # flags.DEFINE_string('model_path', './', 'path to output model/stats')
    # flags.DEFINE_string('vgg19_path', './', 'path to npy file')
    # model_human = 'to_test/model_human_maml/place_sawyer_maml_3_layers_200_dim_1d_conv_ee_3_32_act_2_32_20x1_filters_64_128_3x3_filters_human_light_68k.meta'
    # bias_human = 'to_test/model_human_maml/scale_and_bias_place_sawyer_kinect_view_human.pkl'
    d = SawyerOneShot('to_test/model_maml_kinect2_2/place_sawyer_maml_3_layers_200_dim_ee_2_layers_100_dim_clip_20_kinect_view_47k.meta','to_test/model_maml_kinect2_2/scale_and_bias_place_sawyer_kinect_view_both.pkl')
