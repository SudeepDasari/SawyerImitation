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
from setup_predictor import setup_MAML_predictor_human as setup_predictor_human
import cv2
import os
from keyboard.msg._Key import Key
from ee_velocity_compute import EE_Calculator

import inverse_kinematics
from sensor_msgs.msg import Image as Image_msg


class Traj_aborted_except(Exception):
    pass
from wsg_50_common.msg import Cmd, Status

class SawyerOneShot(object):
    EE_STEPS = 2
    ACTION_SEQUENCE_LENGTH = 10
    SPRING_STIFF = 240
    CONTROL_RATE = 10000
    PREDICT_RATE = 20
    UI_RATE = 1

    CROP_H_MIN = 0
    CROP_H_MAX = -150
    CROP_W_MIN = 190
    CROP_W_MAX = -235
    TRAJ_LEN = 40

    HUMAN_MODEL = 'to_test/model_human_maml/place_sawyer_maml_3_layers_200_dim_1d_conv_ee_3_32_act_2_32_20x1_filters_64_128_3x3_filters_human_light_68k.meta'
    HUMAN_BIAS = 'to_test/model_human_maml/scale_and_bias_place_sawyer_kinect_view_human.pkl'

    ROBOT_MODEL = 'to_test/model_maml_kinect2_2/place_sawyer_maml_3_layers_200_dim_ee_2_layers_100_dim_clip_20_kinect_view_47k.meta'
    ROBOT_BIAS = 'to_test/model_maml_kinect2_2/scale_and_bias_place_sawyer_kinect_view_both.pkl'
    Z_SAFETY_THRESH = 0.20
    def __init__(self):

        self.ctrl = RobotController(self.CONTROL_RATE)



        self.recorder = RobotRecorder(save_dir='',
                                      use_aux=False,
                                      save_actions=False,
                                      save_images=False)
        self.image_publisher = rospy.Publisher('/robot/head_display', Image_msg, queue_size=2)
        self.load_splashes()

        self.control_rate = rospy.Rate(self.PREDICT_RATE)
        self.ui_rate = rospy.Rate(self.UI_RATE)

        self.robot_predictor = setup_predictor(self.ROBOT_MODEL, self.ROBOT_BIAS, self.CROP_H_MIN, self.CROP_H_MAX, self.CROP_W_MIN, self.CROP_W_MAX)
        self.human_predictor = setup_predictor_human(self.HUMAN_MODEL, self.HUMAN_BIAS, self.CROP_H_MIN, self.CROP_H_MAX, self.CROP_W_MIN,
                                               self.CROP_W_MAX)
        self.is_human = False

        self.predictor = self.robot_predictor

        rospy.Subscriber("/keyboard/keyup", Key, self.keyboard_up_listener)
        self.weiss_pub = rospy.Publisher('/wsg_50_driver/goal_position', Cmd, queue_size=10)
        # rospy.Subscriber("/wsg_50_driver/status", Status, self.weiss_status_listener)

        self._navigator = intera_interface.Navigator()
        self.demo_collect = self._navigator.register_callback(self.record_demo, 'right_button_ok')
        self.swap_key = self._navigator.register_callback(self.swap_sawyer_cuff, 'right_button_back')
        self.start_key = self._navigator.register_callback(self.start_traj_cuff, 'right_button_square')
        self.neutral_key = self._navigator.register_callback(self.neutral_cuff, 'right_button_show')
        self.x_button = self._navigator.register_callback(self.gripper_cuff, 'right_button_triangle')

        self.calc = EE_Calculator()



        print 'ONE SHOT READY!'
        self.running = False
        self.demo_imgs = None

        self.move_netural()

        rospy.spin()

    def publish_to_head(self, img):
        num_rows, num_cols = img.shape[:2]
        rotation_matrix = np.array([[0., 1., 0.,], [-1., 0., 1024.,]])
        img = cv2.warpAffine(img, rotation_matrix, (num_rows, num_cols))[:, ::-1, :]


        img = np.swapaxes(img, 0, 1)[::-1, ::-1, :]

        #I don't know why the transforms above work either

        img_message = self.recorder.bridge.cv2_to_imgmsg(img)
        self.image_publisher.publish(img_message)

    def gripper_cuff(self, value):
        if self.running or not value:
            return
        print value
        self.set_weiss_griper(10.)

    def neutral_cuff(self, value):
        if self.running or not value:
          return
        self.move_netural()

    def load_splashes(self):
        self.human_splash = cv2.imread('splashes/human_imitation_start.png')
        self.robot_splash = cv2.imread('splashes/robot_imitation_start.png')
        self.demo_splash = cv2.imread('splashes/recording_oneshot.png')
        #make review splash

        self.publish_to_head(self.robot_splash)

    def swap_model(self):
        if self.is_human:
            self.publish_to_head(self.robot_splash)
            self.predictor = self.robot_predictor
            self.is_human = False
        else:
            self.predictor = self.human_predictor
            self.is_human = True
            self.publish_to_head(self.human_splash)
        self.demo_imgs = None

    def swap_sawyer_cuff(self, value):
        if not value or self.running:
            return
        self.swap_model()

    def start_traj_cuff(self, value):
        if not value or self.running:
            return
        if self.demo_imgs is None:
            print "PLEASE COLLECT DEMOS FIRST"
            return
        self.run_trajectory()

    def record_demo(self, value):
        if not value or self.running:
            return
        print "beginning demo!"
        self.running = True
        self.demo_imgs = []

        traj_ee_pos = np.zeros((self.TRAJ_LEN, 7))
        traj_ee_velocity = np.zeros((self.TRAJ_LEN, 6))
        full_duration = float(self.TRAJ_LEN) / self.PREDICT_RATE

        for i in range(4, 0, -1):
            splash = np.copy(self.demo_splash)
            if i > 1:
                cv2.putText(splash, "{}...".format(i - 1), (325, 460),
                            cv2.FONT_HERSHEY_SIMPLEX, 6, (255, 255, 255), 20,
                            cv2.LINE_AA)
                self.publish_to_head(splash)
                self.ui_rate.sleep()
            else:
                cv2.putText(splash, "GO!", (250, 460),
                            cv2.FONT_HERSHEY_SIMPLEX, 6, (255, 255, 255), 20,
                            cv2.LINE_AA)
                self.publish_to_head(splash)
                for i in range(5):
                    self.control_rate.sleep()


        for i in range(self.TRAJ_LEN):
            splash = np.copy(self.demo_splash)
            cv2.putText(splash, "{:.2f}s".format(full_duration - i * 1. / self.PREDICT_RATE), (250, 460), cv2.FONT_HERSHEY_SIMPLEX, 6, (255, 255, 255), 20,
                        cv2.LINE_AA)
            self.publish_to_head(splash)

            self.control_rate.sleep()
            self.demo_imgs.append(self.recorder.ltob.img_cv2)
            traj_ee_pos[i, :] = self.recorder.get_endeffector_pos()

            angles = self.recorder.get_joint_angles()
            velocities = self.recorder.get_joint_angles_velocity()

            traj_ee_velocity[i, :] = self.calc.jacobian(angles.reshape(-1)).dot(velocities.reshape((-1, 1))).reshape(-1)

        splash = np.copy(self.demo_splash)
        cv2.putText(splash, "{:.2f}s".format(0.00), (250, 460),
                    cv2.FONT_HERSHEY_SIMPLEX, 6, (255, 255, 255), 20,
                    cv2.LINE_AA)
        self.publish_to_head(splash)

        final_ee = np.tile(traj_ee_pos[-1, :2], (40, 1))
        self.record_state = traj_ee_pos
        self.record_action = np.concatenate((traj_ee_velocity[:, :3], final_ee), axis=1)

        self.demo_imgs = np.stack([cv2.resize(img[self.CROP_H_MIN:self.CROP_H_MAX, self.CROP_W_MIN:self.CROP_W_MAX, :],
                                     (100, 100), interpolation=cv2.INTER_AREA)[:, :, ::-1] for img in self.demo_imgs], axis = 0)
        for i in self.demo_imgs:
            splash = np.copy(self.demo_splash)
            resized_i = cv2.resize(i,(400, 400), interpolation = cv2.INTER_CUBIC)
            splash[190:590, 316:716, :] = resized_i[:, :, ::-1]
            self.publish_to_head(splash)

            cv2.imshow('img', i[:, :, ::-1])
            cv2.waitKey(200)
        cv2.destroyAllWindows()

        self.running = False
        print "DEMO DONE"

        if self.is_human:
            self.publish_to_head(self.human_splash)
        else:
            self.publish_to_head(self.robot_splash)


    def keyboard_up_listener(self, key_msg):
        if key_msg.code == 99 and not self.running: #c
            if self.demo_imgs is None:
                print "PLEASE COLLECT A DEMO FIRST"
                return

            self.run_trajectory()

        if key_msg.code == 101 and not self.running: #e
            rospy.signal_shutdown('User shutdown!')
        if key_msg.code == 110 and not self.running: #n
            self.move_netural()
        if key_msg.code == 115 and not self.running: #s
            self.swap_model()
        if key_msg.code == 103 and not self.running: #g
            self.set_weiss_griper(10.)

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

        if interp:
            self.ctrl.set_joint_positions_interp(des_joint_angles)
        else:
            self.ctrl.limb.set_joint_position_speed(0.15)
            self.ctrl.set_joint_positions(des_joint_angles)

    def set_weiss_griper(self, width):
        cmd = Cmd()
        cmd.pos = width
        cmd.speed = 100.
        self.weiss_pub.publish(cmd)

    def move_netural(self, gripper_open = True):
        self.ctrl.set_neutral()
        if gripper_open:
            self.set_weiss_griper(100.)

    def run_trajectory(self):
        self.running = True
        self.start = self.recorder.get_endeffector_pos()
        print 'actual end eep', self.start
        self.move_netural(gripper_open=False)
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
        self.set_weiss_griper(100.)
        print 'end', self.recorder.get_endeffector_pos()
        clip = mpy.ImageSequenceClip([i for i in self.img_stack], fps=20)
        clip.write_gif('test_frames.gif')
        self.running = False



if __name__ == '__main__':
    d = SawyerOneShot()
