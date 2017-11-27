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
from ee_velocity_compute import EE_Calculator

import inverse_kinematics
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32
from std_msgs.msg import Int64

class Traj_aborted_except(Exception):
    pass

class SawyerOneShot(object):
    EE_STEPS = 1
    ACTION_SEQUENCE_LENGTH = 5
    SPRING_STIFF = 220

    def __init__(self, meta_path, norm_path, recording_path):
        self.ctrl = RobotController()

        self.recorder = RobotRecorder(save_dir='',
                                      use_aux=False,
                                      save_actions=False,
                                      save_images=False)

        self.action_interval = 20 #Hz
        self.action_sequence_length = 20

        self.imp_ctrl_publisher = rospy.Publisher('desired_joint_pos', JointState, queue_size=1)
        self.imp_ctrl_release_spring_pub = rospy.Publisher('release_spring', Float32, queue_size=10)
        self.imp_ctrl_active = rospy.Publisher('imp_ctrl_active', Int64, queue_size=10)

        self.traj_duration = self.action_sequence_length * self.action_interval
        self.action_rate = rospy.Rate(self.action_interval)
        self.control_rate = rospy.Rate(20)
        self.predictor = setup_predictor(meta_path, norm_path, recording_path)
        self.save_ctr = 0

        self.move_netural()

        self.s = 0

    def query_action(self):
        image = cv2.resize(self.recorder.ltob.img_cv2[:-150, 190:-235, :], (100, 100), interpolation=cv2.INTER_AREA)[:, :, ::-1]

        robot_configs = self.recorder.get_endeffector_pos()

        if image is None or robot_configs is None:
            return None

        action, ee = self.predictor(image, robot_configs)
        self.img_stack.append(image)

        print 'ee', ee

        return action, ee
    def imp_ctrl_release_spring(self, maxstiff):
        self.imp_ctrl_release_spring_pub.publish(maxstiff)

    def apply_action(self, action):
        try:
            self.ctrl.set_joint_velocities(action)

        except OSError:
            rospy.logerr('collision detected, stopping trajectory, going to reset robot...')
            rospy.sleep(.5)
        if self.ctrl.limb.has_collided():
            rospy.logerr('collision detected!!!')
            rospy.sleep(.5)
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



        self.imp_ctrl_release_spring(self.SPRING_STIFF)
        self.move_with_impedance_sec(des_joint_angles)
        # self.ctrl.set_joint_positions(des_joint_angles)

    def move_with_impedance_sec(self, cmd, duration=2.):
        jointnames = self.ctrl.limb.joint_names()
        prev_joint = np.array([self.ctrl.limb.joint_angle(j) for j in jointnames])
        new_joint = np.array([cmd[j] for j in jointnames])

        start_time = rospy.get_time()  # in seconds
        finish_time = start_time + duration  # in seconds

        while rospy.get_time() < finish_time:
            int_joints = prev_joint + (rospy.get_time()-start_time)/(finish_time-start_time)*(new_joint-prev_joint)
            # print int_joints
            cmd = dict(zip(self.ctrl.limb.joint_names(), list(int_joints)))
            self.move_with_impedance(cmd)
            self.control_rate.sleep()

    def move_with_impedance(self, des_joint_angles):
        """
        non-blocking
        """
        js = JointState()
        js.name = self.ctrl.limb.joint_names()
        js.position = [des_joint_angles[n] for n in js.name]
        self.imp_ctrl_publisher.publish(js)

    def set_neutral_with_impedance(self):
        neutral_jointangles = [0.412271, -0.434908, -1.198768, 1.795462, 1.160788, 1.107675, 2.068076]
        cmd = dict(zip(self.ctrl.limb.joint_names(), neutral_jointangles))
        # self.imp_ctrl_release_spring(20)
        self.move_with_impedance_sec(cmd)


    def move_netural(self):
        # self.ctrl.set_neutral()
        self.imp_ctrl_release_spring(280)
        self.set_neutral_with_impedance()

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