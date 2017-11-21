import os
import glob
import numpy as np
import cv2
import imageio
import cPickle as pkl
import random
from ee_velocity_compute import EE_Calculator

calc = EE_Calculator()
data_dir = '~/oneshot_nov_13/'
data_out = '~/oneshot_data_nov13/'
collector = ['robot', 'human']
IN_T = 40
OUT_T = 40
TRAJ_PER_OBJ = 8
SAMPLES = 1

for c in collector:
    base_path = os.path.expanduser(data_dir + c+'/')
    out_path = os.path.expanduser(data_out + c+'/')
    objects = glob.glob(base_path + '*')
    objects = [p.split(base_path)[1] for p in objects]

    trajs = {obj:sorted(glob.glob(base_path + obj +'/*'), key=lambda x: int(x.split('traj')[1])) for obj in objects}


    for obj_trajs in trajs:
        print 'Loading', c+'/'+obj_trajs
        obj_id = int(obj_trajs.split('object')[1])

        demoX = np.zeros((SAMPLES, TRAJ_PER_OBJ, OUT_T, 14), np.float32)
        demoU = np.zeros((SAMPLES, TRAJ_PER_OBJ, OUT_T, 13), np.float32)

        cntr = 0
        for t in trajs[obj_trajs]:
            imgs_kinect = glob.glob(t + '/images/*.jpg')
            imgs_kinect.sort(key = lambda x: int(x.split('_im')[1][:2]))
            imgs_kinect = [cv2.resize(cv2.imread(i)[:-150, 225:-225, :], (100, 100), interpolation=cv2.INTER_AREA)[:, :, ::-1] for i in imgs_kinect]

            imgs_birds_eye = glob.glob(t + '/images/cam0/*.jpg')
            imgs_birds_eye.sort(key = lambda x: int(x.split('_im')[1][:2]))
            imgs_birds_eye = [cv2.resize(cv2.imread(i)[150:, 175:-400, :], (100, 100), interpolation=cv2.INTER_AREA)[:, :, ::-1] for i in imgs_birds_eye]

            imgs_bottom = glob.glob(t + '/images/cam1/*.jpg')
            imgs_bottom.sort(key = lambda x: int(x.split('_im')[1][:2]))
            imgs_bottom = [cv2.resize(cv2.imread(i)[25:-75, 400:-75, :], (100, 100), interpolation=cv2.INTER_AREA)[:, :, ::-1] for i in imgs_bottom]

            gif_out_dir = out_path + obj_trajs +'/'
            if not os.path.exists(gif_out_dir):
                os.makedirs(gif_out_dir)

            traj_pkl = glob.glob(t + '/*.pkl')[0]
            traj_pkl = pkl.load(open(traj_pkl, 'rb'))

            traj_angles = traj_pkl['jointangles']
            traj_vels = traj_pkl['jointvelocities']
            traj_eep = np.zeros((IN_T, 7))
            for i in range(IN_T):
                traj_eep[i, :] = calc.forward_position_kinematics(traj_angles[i, :])
            traj_eep_vel = np.zeros((IN_T, 6))
            for i in range(IN_T):
                ja_fr = traj_angles[i, :]
                jv_fr = traj_vels[i, :]
                ee_vel = calc.jacobian(ja_fr).dot(jv_fr.reshape((-1, 1)))
                traj_eep_vel[i, :] = ee_vel.reshape(-1)



            for s in range(SAMPLES):
                for imgs, view in [(imgs_birds_eye, 'view0'), (imgs_kinect, 'view1'), (imgs_bottom, 'view2')]:
                    first_fr = [imgs[0]]
                    last_fr = [imgs[-1]]
                    samps = sorted(random.sample(range(1, IN_T), k = OUT_T - 2))
                    img_out = first_fr + [imgs[i] for i in samps] + last_fr

                    gif_out = gif_out_dir+ 'cond' + str(cntr) + '.samp' + str(s) + '.' + view + '.gif'
                    print 'Outputing', len(img_out), 'images to', gif_out

                    writer = imageio.save(gif_out, duration= (IN_T / 20.) / OUT_T,
                                          quantizer= 0 , palettesize=256)
                    i = 0
                    for frame in img_out:
                        writer.append_data(frame)
                        i += 1
                    print 'Wrote', i , 'frames for', view

                samps = [0] + samps + [IN_T - 1]
                # print 'Outputting', len(samps), 'samples'
                demoX[s, cntr, :, :7] = traj_angles[samps, :]
                demoX[s, cntr, :, 7:] = traj_eep[samps, :]
                demoU[s, cntr, :, :7] = traj_vels[samps, :]
                demoU[s, cntr, :, 7:] = traj_eep_vel[samps, :]

            cntr += 1


        demo_dict = {'demoX':demoX, 'demoU':demoU}
        pkl.dump(demo_dict, open(out_path + '/demo' + str(obj_id) + '.pkl', 'wb'))