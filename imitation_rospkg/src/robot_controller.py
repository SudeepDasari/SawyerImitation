#!/usr/bin/env python

import argparse
import rospy

import socket
import intera_interface
import intera_external_devices
from intera_interface import CHECK_VERSION

import numpy as np
import socket
import pdb
from std_msgs.msg import Float32
from std_msgs.msg import Int64
from sensor_msgs.msg import JointState

class RobotController(object):

    def __init__(self, CONTROL_RATE):
        """Initializes a controller for the robot"""

        print("Initializing node... ")
        rospy.init_node("sawyer_custom_controller")
        rospy.on_shutdown(self.clean_shutdown)

        rs = intera_interface.RobotEnable(CHECK_VERSION)
        init_state = rs.state().enabled
        print("Robot enabled...")

        self.limb = intera_interface.Limb("right")
        self.limb.set_command_timeout(1 / 20.)

        self.joint_names = self.limb.joint_names()
        print("Done initializing controller.")

        self.imp_ctrl_publisher = rospy.Publisher('desired_joint_pos', JointState, queue_size=1)
        self.imp_ctrl_release_spring_pub = rospy.Publisher('release_spring', Float32, queue_size=10)
        self.imp_ctrl_active = rospy.Publisher('imp_ctrl_active', Int64, queue_size=10)

        self.control_rate = rospy.Rate(CONTROL_RATE)


    def set_joint_delta(self, joint_name, delta):
        """Move a single joint by a delta"""
        current_position = self.limb.joint_angle(joint_name)
        self.set_joint_position(joint_name, current_position + delta)

    def set_joint_position(self, joint_name, pos):
        """Move a single joint to a target position"""
        joint_command = {joint_name: pos}
        self.limb.set_joint_positions(joint_command)

    def set_joint_positions(self, positions):
        """Move joints to commmand"""
        self.limb.move_to_joint_positions(positions)

    def set_joint_velocities(self, velocities):
        self.limb.set_joint_velocities(velocities)

    def set_neutral(self, speed=.2):
        # using a custom handpicked neutral position
        # starting from j0 to j6:
        neutral_jointangles = [0.412271, -0.434908, -1.198768, 1.795462, 1.160788, 1.107675, 2.068076]
        cmd = dict(zip(self.joint_names, neutral_jointangles))

        self.limb.set_joint_position_speed(speed)

        done = False
        while not done:
            try:
                self.set_joint_positions(cmd)
            except:
                print 'retrying set neutral...'

            done = True
    def imp_ctrl_release_spring(self, maxstiff):
        self.imp_ctrl_release_spring_pub.publish(maxstiff)

    def move_with_impedance_sec(self, cmd, duration=2.):
        jointnames = self.limb.joint_names()
        prev_joint = np.array([self.limb.joint_angle(j) for j in jointnames])
        new_joint = np.array([cmd[j] for j in jointnames])

        start_time = rospy.get_time()  # in seconds
        finish_time = start_time + duration  # in seconds

        while rospy.get_time() < finish_time:
            int_joints = prev_joint + (rospy.get_time()-start_time)/(finish_time-start_time)*(new_joint-prev_joint)
            # print int_joints
            cmd = dict(zip(self.limb.joint_names(), list(int_joints)))
            self.move_with_impedance(cmd)
            self.control_rate.sleep()

    def move_with_impedance(self, des_joint_angles):
        """
        non-blocking
        """
        js = JointState()
        js.name = self.limb.joint_names()
        js.position = [des_joint_angles[n] for n in js.name]
        self.imp_ctrl_publisher.publish(js)

    def set_neutral_with_impedance(self):
        neutral_jointangles = [0.412271, -0.434908, -1.198768, 1.795462, 1.160788, 1.107675, 2.068076]
        cmd = dict(zip(self.limb.joint_names(), neutral_jointangles))
        # self.imp_ctrl_release_spring(20)
        self.move_with_impedance_sec(cmd)

    def clean_shutdown(self):
        print("\nExiting example.")
        # if not init_state:
        #     print("Disabling robot...")
            # rs.disable()