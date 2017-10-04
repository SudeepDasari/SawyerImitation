from train_net import Model
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
import flags

from robot_controller import RobotController
from recorder.robot_recorder import RobotRecorder
from setup_predictor import setup_predictor


class SawyerImitation(object):
    def __init__(self, model_path, vgg19_path):

        parser = argparse.ArgumentParser()
        self.args = parser.parse_args()
        parser.add_argument('--save_dir', default='./', type=str, help='')

        print("Initializing node... ")
        rospy.init_node("sawyer_imitation_client")

        self.rs = intera_interface.RobotEnable(CHECK_VERSION)
        self.init_state = self.rs.state().enabled

        self.ctrl = RobotController()

        self.limb = intera_interface.Limb("right")
        self.joint_names = self.limb.joint_names()

        self.recorder = RobotRecorder(save_dir=self.args.save_dir,
                                      seq_len=60,
                                      use_aux=False,
                                      save_video=True,
                                      save_actions=False,
                                      save_images=False)

        self.action_interval = 20 #Hz
        self.action_sequence_length = 60
        self.traj_duration = self.action_sequence_length * self.action_interval
        self.action_rate = rospy.Rate(self.action_interval)
        self.control_rate = rospy.Rate(20)
        self.predictor = setup_predictor(model_path, vgg19_path)

        # self.imp_ctrl_publisher = rospy.Publisher('desired_joint_pos', JointState, queue_size=1)
        # self.imp_ctrl_release_spring_pub = rospy.Publisher('release_spring', Float32, queue_size=10)
        # self.imp_ctrl_active = rospy.Publisher('imp_ctrl_active', Int64, queue_size=10)
        # self.imp_ctrl_active.publish(0)

        rospy.on_shutdown(self.ctrl.clean_shutdown)

        self.ctrl.set_neutral()

    def query_action(self):
        image = self.recorder.bridge.cv2_to_imgmsg(self.recorder.ltob.img_cropped)
        robot_configs = np.concatenate((self.recorder.get_endeffector_pos(), self.recorder.get_joint_angles()))

        try:
            rospy.wait_for_service('get_action', timeout=240)
            action_vec, _ = self.predictor(image, robot_configs)

        except (rospy.ServiceException, rospy.ROSException), e:
            rospy.logerr("Service call failed: %s" % (e,))
            raise ValueError('get action service call failed')
        return action_vec

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
            print 'action vector: ', action

            # self.apply_action(action)

            step += 1

if __name__ == '__main__':
    FLAGS = flags.FLAGS
    flags.DEFINE_string('model_path', './', 'path to output model/stats')
    flags.DEFINE_string('vgg19_path', './', 'path to npy file')
    d = SawyerImitation(flags.model_path, flags.vgg19_path)
    d.run_trajectory()
