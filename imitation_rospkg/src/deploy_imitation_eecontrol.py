#!/usr/bin/env python
import tensorflow as tf
import numpy as np
import argparse
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
from setup_predictor import setup_predictor_ee_control as setup_predictor
import cv2
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32
from std_msgs.msg import Int64
import inverse_kinematics

class Traj_aborted_except(Exception):
    pass

class SawyerImitation(object):
    def __init__(self, model_path, vgg19_path):
        self.ctrl = RobotController()

        self.recorder = RobotRecorder(save_dir='',
                                      seq_len=60,
                                      use_aux=False,
                                      save_video=True,
                                      save_actions=False,
                                      save_images=False)

        self.action_interval = 10 #Hz
        self.action_sequence_length = 60

        self.control_rate = rospy.Rate(self.action_interval)
        self.predictor = setup_predictor(model_path, vgg19_path)
        self.save_ctr = 0

        self.s = 0
        self.imp_ctrl_publisher = rospy.Publisher('desired_joint_pos', JointState, queue_size=1)
        self.imp_ctrl_release_spring_pub = rospy.Publisher('release_spring', Float32, queue_size=10)
        self.imp_ctrl_active = rospy.Publisher('imp_ctrl_active', Int64, queue_size=10)
        self.move_neutral()

    def move_neutral(self):
        # self.imp_ctrl_active.publish(0)
        self.ctrl.set_neutral()
        # self.set_neutral_with_impedance()
        # self.imp_ctrl_active.publish(1)
        # rospy.sleep(.2)

    def query_action(self, robot_configs):
        image = cv2.resize(self.recorder.ltob.img_cv2[:-150,150:-275,:], (224, 224), interpolation = cv2.INTER_AREA)



        if image is None or robot_configs is None:
            return None

        action, predicted_eep, image = self.predictor(image, robot_configs)

        self.img_stack.append(image[0, :, :, :])

        self.s += 1
        # if self.s <=5:
        #     action[np.abs(action) < 0.05] *= 15

        # print 'action vector: ', action
        # print 'predicted end effector pose: ', predicted_eep
        return action, predicted_eep

    def move_to(self, des_pos):
        desired_pose = inverse_kinematics.get_pose_stamped(des_pos[0],
                                                           des_pos[1],
                                                           des_pos[2],
                                                           inverse_kinematics.EXAMPLE_O)
        start_joints = self.ctrl.limb.joint_angles()
        try:
            des_joint_angles = inverse_kinematics.get_joint_angles(desired_pose, seed_cmd=start_joints,
                                                                   use_advanced_options=True)
        except ValueError:
            rospy.logerr('no inverse kinematics solution found, '
                         'going to reset robot...')
            current_joints = self.ctrl.limb.joint_angles()
            self.ctrl.limb.set_joint_positions(current_joints)
            raise Traj_aborted_except('raising Traj_aborted_except')

        # self.move_with_impedance(des_joint_angles)
        self.ctrl.set_joint_positions(des_joint_angles)


    def move_with_impedance(self, des_joint_angles):
        """
        non-blocking
        """
        js = JointState()
        js.name = self.ctrl.limb.joint_names()
        js.position = [des_joint_angles[n] for n in js.name]
        self.imp_ctrl_publisher.publish(js)

    def imp_ctrl_release_spring(self, maxstiff):
        self.imp_ctrl_release_spring_pub.publish(maxstiff)

    def move_with_impedance_sec(self, cmd, tsec = 2.):
        """
        blocking
        """
        tstart = rospy.get_time()
        delta_t = 0
        while delta_t < tsec:
            delta_t = rospy.get_time() - tstart
            self.move_with_impedance(cmd)

    def set_neutral_with_impedance(self):
        neutral_jointangles = [0.412271, -0.434908, -1.198768, 1.795462, 1.160788, 1.107675, 2.068076]
        cmd = dict(zip(self.ctrl.limb.joint_names(), neutral_jointangles))
        self.imp_ctrl_release_spring(20)
        self.move_with_impedance_sec(cmd)

    def run_trajectory(self):
        self.start = self.recorder.get_endeffector_pos()
        print 'actual end eep', self.start

        self.move_neutral()

        self.img_stack = []

        step = 0
        actions = []
        while step < self.action_sequence_length:
            print step
            self.control_rate.sleep()
            current_eep = self.recorder.get_endeffector_pos()
            eep_diff_action, predicted_eep = self.query_action(current_eep)


            self.move_to(current_eep[:3] + eep_diff_action[:3])

            step += 1
        print 'end', self.recorder.get_endeffector_pos()
        clip = mpy.ImageSequenceClip([cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in self.img_stack], fps=20)
        clip.write_gif('test_frames.gif')

        self.move_neutral()

if __name__ == '__main__':
    # FLAGS = flags.FLAGS
    # flags.DEFINE_string('model_path', './', 'path to output model/stats')
    # flags.DEFINE_string('vgg19_path', './', 'path to npy file')
    d = SawyerImitation('pred_diff_100_75_norm/modelfinal', 'data/')
    pdb.set_trace()
    d.run_trajectory()
