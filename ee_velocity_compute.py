import numpy as np
import PyKDL

from baxter_kdl.kdl_parser import kdl_tree_from_urdf_model
from urdf_parser_py.urdf import URDF
import cPickle as pkl
import os

class EE_Calculator:
    NUM_JOINTS = 7
    def __init__(self):
        self._baxter = URDF.from_parameter_server(key='robot_description')
        self._kdl_tree = kdl_tree_from_urdf_model(self._baxter)
        self._base_link = self._baxter.get_root()
        self._tip_link = 'right_hand'
        self._arm_chain = self._kdl_tree.getChain(self._base_link,
                                                  self._tip_link)

        self.jac_solver = PyKDL.ChainJntToJacSolver(self._arm_chain)
        self.fk_solver = PyKDL.ChainFkSolverPos_recursive(self._arm_chain)

    def kdl_to_mat(self, data):
        mat =  np.mat(np.zeros((data.rows(), data.columns())))
        for i in range(data.rows()):
            for j in range(data.columns()):
                mat[i,j] = data[i,j]
        return mat

    def jacobian(self,joint_values):
        jacobian = PyKDL.Jacobian(self.NUM_JOINTS)

        joint_array = PyKDL.JntArray(self.NUM_JOINTS)
        for i, val in enumerate(joint_values):
            joint_array[i] = val

        self.jac_solver.JntToJac(joint_array, jacobian)
        return self.kdl_to_mat(jacobian)

    def forward_position_kinematics(self,joint_values):
        end_frame = PyKDL.Frame()

        joint_array = PyKDL.JntArray(self.NUM_JOINTS)
        for i, val in enumerate(joint_values):
            joint_array[i] = val


        self.fk_solver.JntToCart(joint_array,end_frame)
        pos = end_frame.p
        rot = PyKDL.Rotation(end_frame.M)
        rot = rot.GetQuaternion()
        return np.array([pos[0], pos[1], pos[2],
                         rot[0], rot[1], rot[2], rot[3]])

calc = EE_Calculator()
#print calc.jacobian(np.zeros(7)).shape


# demo_dict = pkl.load(open(os.path.expanduser('~/oneshot_demos_10_24/robot/demo1.pkl'), 'rb'))
# states = demo_dict['demoX']
# actions = demo_dict['demoU']
#
# ja_1 = states[0, 0, 1, :7]
# ja_2 = states[0, 0, 2, :7]
#
# jv_1 = actions[0, 0, 1, :]
# jv_2 = actions[0, 0, 2, :]
#
# ee_position1 = calc.forward_position_kinematics(ja_1)
# ee_position2 = calc.forward_position_kinematics(ja_2)
#
# print 'ee_vel diff', (ee_position2 - ee_position1) * 20.
# print 'ee_vel calc 1', calc.jacobian(ja_1).dot(jv_1.reshape((-1, 1)))
# print 'ee_vel calc 2', calc.jacobian(ja_2).dot(jv_2.reshape((-1, 1)))
#
# print 'ee_pos1', ee_position2[:3]
# print 'ee_pos1 rec', states[0, 0, 2, 7:]
#
# import matplotlib.pyplot as plt
#
# plot_vel = np.zeros(40)
#
# for i in range(40):
#     ja_fr = states[0, 0, i, :7]
#     jv_fr = actions[0, 0, i, :]
#     plot_vel[i] = calc.jacobian(ja_fr).dot(jv_fr.reshape((-1, 1)))[4]
# plt.plot(plot_vel)
#
# plt.figure()
#
# plot_vel = np.zeros(39)
#
# for i in range(39):
#     ja_fr = states[0, 0, i, :7]
#     ja_fr_plus = states[0, 0, i + 1, :7]
#
#     ee_fr = calc.forward_position_kinematics(ja_fr)[4]
#     ee_fr_plus = calc.forward_position_kinematics(ja_fr_plus)[4]
#     # print (ee_fr_plus - ee_fr)
#     plot_vel[i] = (ee_fr_plus - ee_fr) * 20.
# plt.plot(plot_vel)
# plt.show()

in_append = os.path.expanduser('~/oneshot_demos_10_24/')
out_append = os.path.expanduser('~/oneshot_demos_eemod_10_24/')

N_TO_CONVERT = 162
for c in ['human', 'robot']:
    for file_num in range(N_TO_CONVERT):
        in_path = in_append + c + '/demo' + str(file_num) + '.pkl'
        out_path = out_append + c + '/demo' + str(file_num) + '.pkl'

        in_dict = pkl.load(open(in_path, 'rb'))
        states = in_dict['demoX']
        actions = in_dict['demoU']

        samp_dim, demo_dim, time_dim = states.shape[:3]

        #output ja, eep
        #output jv, ee velocities
        out_states = np.zeros((samp_dim, demo_dim, time_dim, 14))
        out_actions = np.zeros((samp_dim, demo_dim, time_dim, 13))

        for i in range(samp_dim):
            for j in range(demo_dim):
                for k in range(time_dim):
                    ja_fr = states[i, j, k, :7]
                    jv_fr = actions[i, j, k, :]

                    out_states[i, j, k, :7] = ja_fr
                    out_states[i, j, k, 7:] = calc.forward_position_kinematics(ja_fr)

                    out_actions[i, j, k, :7] = jv_fr
                    ee_velocity = calc.jacobian(ja_fr).dot(jv_fr.reshape((-1, 1)))
                    out_actions[i, j, k, 7:] = ee_velocity.reshape(-1)

        out_dict = {'demoX':out_states, 'demoU':out_actions}
        pkl.dump(out_dict, open(out_path, 'wb'))
        print 'for type', c, 'converted obj dict:', file_num