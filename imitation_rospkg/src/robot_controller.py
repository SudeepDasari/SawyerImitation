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


class RobotController(object):

    def __init__(self):
        """Initializes a controller for the robot"""

        print("Initializing node... ")
        rospy.init_node("sawyer_custom_controller")
        rospy.on_shutdown(self.clean_shutdown)

        rs = intera_interface.RobotEnable(CHECK_VERSION)
        init_state = rs.state().enabled
        print("Robot enabled...")

        self.limb = intera_interface.Limb("right")

        self.joint_names = self.limb.joint_names()
        print("Done initializing controller.")

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

    def clean_shutdown(self):
        print("\nExiting example.")
        # if not init_state:
        #     print("Disabling robot...")
            # rs.disable()