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
from sensor_msgs.msg import JointState
from intera_core_msgs.srv import (
    SolvePositionFK,
    SolvePositionFKRequest,
)

from robot_controller import RobotController
from recorder.robot_recorder import RobotRecorder

class SawyerImitation(object):
    def __init__(self):

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
                                      use_aux=self.use_aux,
                                      save_video=True,
                                      save_actions=False,
                                      save_images=False)

        self.name_of_service = "ExternalTools/right/PositionKinematicsNode/FKService"
        self.fksvc = rospy.ServiceProxy(self.name_of_service, SolvePositionFK)

        self.bridge = CvBridge()

        self.action_interval = 1 #Hz
        self.action_sequence_length = 60
        self.traj_duration = self.action_sequence_length * self.action_interval
        self.action_rate = rospy.Rate(self.action_interval)
        self.control_rate = rospy.Rate(20)

        self.get_action_func = rospy.ServiceProxy('get_action', get_action)

        # self.imp_ctrl_publisher = rospy.Publisher('desired_joint_pos', JointState, queue_size=1)
        # self.imp_ctrl_release_spring_pub = rospy.Publisher('release_spring', Float32, queue_size=10)
        # self.imp_ctrl_active = rospy.Publisher('imp_ctrl_active', Int64, queue_size=10)
        # self.imp_ctrl_active.publish(0)

        rospy.on_shutdown(self.ctrl.clean_shutdown)

        self.ctrl.set_neutral()

    # def start_recording(self, value):
    #     if not value:
    #         return
    #
    #     if self.collect_active:
    #         return
    #     self.recorder.init_traj(self.record_iter)
    #     self.record_iter += 1
    #
    #     self.collect_active = True
    #     self.imp_ctrl_active.publish(0)
    #     # self.joint_pos = []
    #
    #     iter = 0
    #     while (self.collect_active):
    #         self.control_rate.sleep()
    #         # self.joint_pos.append(self.limb.joint_angles())
    #         pose = self.get_endeffector_pos()
    #         print 'recording ', iter
    #         self.recorder.save(iter, pose)
    #         iter += 1
    #
    #         if (iter >= self.N_SAMPLES):
    #             self.collect_active = False
    #
    #             # filename = '/home/sudeep/outputs/pushback_traj_.pkl'
    #             # with open(filename, 'wb') as f:
    #             #     cPickle.dump(self.joint_pos, f)
    #
    #             # print 'saved file to ', filename

    def query_action(self):

        if self.use_aux:
            self.recorder.get_aux_img()
            imageaux1 = self.recorder.ltob_aux1.img_msg
        else:
            imageaux1 = np.zeros((64, 64, 3), dtype=np.uint8)
            imageaux1 = self.bridge.cv2_to_imgmsg(imageaux1)

        imagemain = self.bridge.cv2_to_imgmsg(self.recorder.ltob.img_cropped)
        robot_configs = self.get_endeffector_pos() # need to add joint angles here

        try:
            rospy.wait_for_service('get_action', timeout=240)
            get_action_resp = self.get_action_func(imagemain, tuple(robot_configs))

            action_vec = get_action_resp.action

        except (rospy.ServiceException, rospy.ROSException), e:
            rospy.logerr("Service call failed: %s" % (e,))
            raise ValueError('get action service call failed')
        return action_vec

    def get_endeffector_pos(self):
        fkreq = SolvePositionFKRequest()
        joints = JointState()
        joints.name = self.limb.joint_names()
        joints.position = [self.limb.joint_angle(j)
                           for j in joints.name]

        # Add desired pose for forward kinematics
        fkreq.configuration.append(joints)
        fkreq.tip_names.append('right_hand')
        try:
            rospy.wait_for_service(self.name_of_service, 5)
            resp = self.fksvc(fkreq)
        except (rospy.ServiceException, rospy.ROSException), e:
            rospy.logerr("Service call failed: %s" % (e,))
            return False

        pos = np.array([resp.pose_stamp[0].pose.position.x,
                        resp.pose_stamp[0].pose.position.y,
                        resp.pose_stamp[0].pose.position.z])
        return pos
