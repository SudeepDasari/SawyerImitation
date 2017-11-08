import tensorflow as tf
import numpy as np
from imitation_learning2 import ImitationLearningModel
import os
import cv2
import time
import cPickle as pickle
import moviepy.editor as mpy

def setup_predictor_ee_control(model_path, vgg19_path):
    from imitation_learning6 import ImitationLearningModel as ImitationLearningModelEE
    images_pl = tf.placeholder(tf.uint8, name='images', shape=(1, 224, 224, 3))
    configs_pl = tf.placeholder(tf.float32, name='configs', shape=(1, 7))

    with tf.variable_scope('model', reuse=None):
        model = ImitationLearningModelEE(vgg19_path, images=images_pl, robot_configs=configs_pl)
        model.build()
        print model.predicted_actions
        print model.predicted_eeps

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)

    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    saver = tf.train.Saver(vars, max_to_keep=0)

    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    saver.restore(sess, model_path)

    def predictor_func(images=None, robot_configs=None):
        feed_image = images.astype(np.uint8).reshape((1, 224, 224, 3))
        feed_config = robot_configs.astype(np.float32).reshape((1, 7))

        feed_dict = {
            images_pl: feed_image,
            configs_pl: feed_config
        }

        predicted_actions, predicted_eeps = sess.run([model.predicted_actions, model.predicted_eeps], feed_dict)
        # print 'fp_x', fp_x
        # print 'fp_y', fp_y
        # height, width = images.shape[:2]
        # drawn = images.copy()
        # for i in range(32):
        #     cv2.circle(drawn, (int(height * (fp_x[i] + 0.5)) , int(width * (fp_y[i] + 0.5))), 5, (0, 0, 255), -1)

        return predicted_actions[0], predicted_eeps[0]

    return predictor_func

def setup_predictor(model_path, vgg19_path):
    images_pl = tf.placeholder(tf.uint8, name='images', shape=(1, 224, 224, 3))
    configs_pl = tf.placeholder(tf.float32, name='configs', shape=(1, 10))

    with tf.variable_scope('model', reuse=None):
        model = ImitationLearningModel(vgg19_path, images=images_pl, robot_configs=configs_pl)
        model.build()
        print model.predicted_actions
        print model.predicted_eeps


    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)

    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    saver = tf.train.Saver(vars, max_to_keep=0)

    sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    saver.restore(sess, model_path)

    def predictor_func(images=None, robot_configs=None):

        feed_image = images.astype(np.uint8).reshape((1, 224, 224, 3))
        feed_config = robot_configs.astype(np.float32).reshape((1, 10))

        feed_dict = {
            images_pl: feed_image,
            configs_pl: feed_config
        }


        predicted_actions, predicted_eeps = sess.run([model.predicted_actions, model.predicted_eeps], feed_dict)
        # print 'fp_x', fp_x
        # print 'fp_y', fp_y
        # height, width = images.shape[:2]
        # drawn = images.copy()
        # for i in range(32):
        #     cv2.circle(drawn, (int(height * (fp_x[i] + 0.5)) , int(width * (fp_y[i] + 0.5))), 5, (0, 0, 255), -1)

        return predicted_actions[0], predicted_eeps[0]

    return predictor_func



def load_video_ref(imagea, imageb, statea, actiona, stateb, scale, bias, T=40):

   imagea = np.transpose(imagea, [0, 3, 2, 1])
   imagea = np.reshape(imagea, [1, T, -1]).astype(np.float32) / 255.0
   #imageb is just an image
   imageb = np.expand_dims(imageb, axis=0)
   imageb = np.transpose(imageb, [0, 3, 2, 1])
   imageb = np.reshape(imageb, [1, 1, -1]).astype(np.float32) / 255.0
   actiona = actiona.reshape((1, T, -1))
   statea = statea.reshape((1, T, -1))
   stateb = stateb.reshape((1, 1, -1))
   final_eepta = np.tile(np.expand_dims(statea[:, -1, 7:9], axis=1), [1, T, 1])
   #final_eepta = np.tile(statea[:, -1, 7:], axis=1), [1, T, 1])
   actiona = np.concatenate((actiona, final_eepta), axis=2)
   statea = statea.dot(scale) + bias
   stateb = stateb.dot(scale) + bias
   return imagea, imageb, statea, stateb, actiona


def load_video_ref_human(imagea, imageb, stateb, scale, bias, T=40):

   imagea = np.transpose(imagea, [0, 3, 2, 1])
   imagea = np.reshape(imagea, [1, T, -1]).astype(np.float32) / 255.0
   #imageb is just an image
   imageb = np.expand_dims(imageb, axis=0)
   imageb = np.transpose(imageb, [0, 3, 2, 1])
   imageb = np.reshape(imageb, [1, 1, -1]).astype(np.float32) / 255.0

   stateb = stateb.reshape((1, 1, -1))

   stateb = stateb.dot(scale) + bias
   return imagea, imageb, stateb


def setup_MAML_predictor(meta_path, norm_path, recording_path):
    meta_path = os.path.expanduser(meta_path)
    norm_path = os.path.expanduser(norm_path)
    recording_path = os.path.expanduser(recording_path)
    sess = tf.InteractiveSession()
    print 'made sess'
    saver = tf.train.import_meta_graph(meta_path)
    print 'imported meta'
    saver.restore(sess, meta_path[:-5])
    print 'restored'

    #video demo recorded
    obsa = tf.get_default_graph().get_tensor_by_name('obsa:0')
    #robot state vector
    statea = tf.get_default_graph().get_tensor_by_name('statea:0')
    #robot velocity vector
    actiona = tf.get_default_graph().get_tensor_by_name('actiona:0')
    #from robot (single image)
    obsb = tf.get_default_graph().get_tensor_by_name('obsb:0')
    #from robot (1x7)
    stateb = tf.get_default_graph().get_tensor_by_name('stateb:0')
    #
    output = tf.get_default_graph().get_tensor_by_name('output_action:0')
    print 'got tensors'
    rec_dict = pickle.load(open(recording_path + 'states.pkl', 'rb'))
    record_state =  rec_dict['demoX'] # get from pickle
    record_action = rec_dict['demoU'] #get from pickle

    print 'loaded input state', record_state.shape, 'actions', record_action.shape

    with open(norm_path, 'rb') as f:
        res = pickle.load(f)
        scale = res['scale']

        bias = res['bias']
    print 'loaded scale', scale.shape, 'bias', bias.shape
    clip = mpy.VideoFileClip(recording_path + 'record.gif')
    image_a = np.stack([fr for fr in clip.iter_frames()], axis = 0)[:40, :, :, :3]
    print 'loaded img gif', image_a.shape

    def predictor_func(imageb, robot_state):
            img_a, img_b, state_a, state_b, action_a = load_video_ref(image_a, imageb, record_state, record_action, robot_state,
                                                                      scale, bias)
            #1x40x(100*100*3)
            print 'obsa', img_a.shape
            #1 x 1 x (100*100*3)
            print 'obsb', img_b.shape
            #1 x 40 x 10
            print 'statea', state_a.shape
            # 1 x 1 x 10
            print 'stateb', state_b.shape
            #1 x 40 x 9
            print 'actiona', action_a.shape
            # print XD
            pred_actions = sess.run([output], feed_dict={obsa: img_a, obsb: img_b, statea: state_a, stateb: state_b,
                                                        actiona: action_a})[0].reshape(-1)
            return pred_actions

    return predictor_func

def setup_MAML_predictor_human(meta_path, norm_path, recording_path):
    meta_path = os.path.expanduser(meta_path)
    norm_path = os.path.expanduser(norm_path)
    recording_path = os.path.expanduser(recording_path)
    sess = tf.InteractiveSession()
    print 'made sess'
    saver = tf.train.import_meta_graph(meta_path)
    print 'imported meta'
    saver.restore(sess, meta_path[:-5])
    print 'restored'

    #video demo recorded
    obsa = tf.get_default_graph().get_tensor_by_name('obsa:0')
    #from robot (single image)
    obsb = tf.get_default_graph().get_tensor_by_name('obsb:0')
    #from robot (1x10)
    stateb = tf.get_default_graph().get_tensor_by_name('stateb:0')
    #
    output = tf.get_default_graph().get_tensor_by_name('output_action:0')
    print 'got tensors'



    with open(norm_path, 'rb') as f:
        res = pickle.load(f)
        scale = res['scale']
        bias = res['bias']
        print 'from', norm_path, 'loaded scale', scale.shape, 'bias', bias.shape
    clip = mpy.VideoFileClip(recording_path + 'record.gif')
    image_a = np.stack([fr for fr in clip.iter_frames()], axis = 0)[:40, :, :, :3]
    print 'loaded img gif', image_a.shape

    def predictor_func(imageb, robot_state):
        #imagea, imageb, stateb
            img_a, img_b, state_b = load_video_ref_human(image_a, imageb, robot_state, scale, bias)
            #1x40x(100*100*3)
            print 'obsa', img_a.shape
            #1 x 1 x (100*100*3)
            print 'obsb', img_b.shape
            # 1 x 1 x 10
            print 'stateb', state_b.shape
            # print XD
            pred_actions = sess.run([output], feed_dict={obsa: img_a, obsb: img_b, stateb: state_b})[0].reshape(-1)
            return pred_actions

    return predictor_func

if __name__ == '__main__':
    demo_image = cv2.imread('../../test0.jpg')
    demo_action = np.zeros(10)
    demo_action[0] = 0.1
    pred = setup_predictor('../../single_lossrev_model/modelfinal', '../../out/')
    pred_actions, pred_eep = pred(demo_image, demo_action)
    print 'predicted actions', pred_actions
    print 'predicted eep', pred_eep

    t0 = time.time()
    for i in range(1000):

        pred_actions, pred_eep = pred(demo_image, demo_action)
    t1 = time.time()
    print 'time:', (t1 - t0) / 1000
