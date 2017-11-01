import os
import glob
import numpy as np
import cv2
import moviepy.editor as mpy
import imageio
import cPickle as pkl
import random
data_dir = '~/demonstration_10_24/'
data_out = '~/oneshot_demos_10_24/'
collector = ['human', 'robot']
IN_T = 60
OUT_T = 40
TRAJ_PER_OBJ = 8
SAMPLES = 5

for c in collector:
    base_path = os.path.expanduser(data_dir + c+'/')
    out_path = os.path.expanduser(data_out + c+'/')
    objects = glob.glob(base_path + '*')
    objects = [p.split(base_path)[1] for p in objects]

    trajs = {obj:sorted(glob.glob(base_path + obj +'/*'), key=lambda x: int(x.split('traj')[1])) for obj in objects}


    for obj_trajs in trajs:
        print 'Loading', c+'/'+obj_trajs
        obj_id = int(obj_trajs.split('object')[1])

        demoX = np.zeros((SAMPLES, TRAJ_PER_OBJ, OUT_T, 10), np.float32)
        demoU = np.zeros((SAMPLES, TRAJ_PER_OBJ, OUT_T, 7), np.float32)

        cntr = 0
        for t in trajs[obj_trajs]:
            imgs = glob.glob(t + '/images/*.jpg')
            imgs.sort(key = lambda x: int(x.split('_im')[1][:2]))

            imgs = [cv2.resize(cv2.imread(i)[:-150, 225:-225, :], (100, 100), interpolation=cv2.INTER_AREA)[:, :, ::-1] for i in imgs]
            gif_out_dir = out_path + obj_trajs +'/'
            if not os.path.exists(gif_out_dir):
                os.makedirs(gif_out_dir)

            traj_pkl = glob.glob(t + '/*.pkl')[0]
            traj_pkl = pkl.load(open(traj_pkl, 'rb'))
            traj_angles = traj_pkl['jointangles']
            traj_vels = traj_pkl['jointvelocities']
            traj_efs = traj_pkl['endeffector_pos']

            first_fr = [imgs[0]]
            last_fr = [imgs[-1]]

            for s in range(SAMPLES):
                samps = sorted(random.sample(range(1, IN_T), k = OUT_T - 2))
                img_out = first_fr + [imgs[i] for i in samps] + last_fr

                gif_out = gif_out_dir+ 'cond' + str(cntr) + '.samp' + str(s) + '.gif'
                print 'Outputing', len(img_out), 'images to', gif_out

                writer = imageio.save(gif_out, duration= (IN_T / 20.) / OUT_T,
                                      quantizer= 0 , palettesize=256)
                i = 0
                for frame in img_out:
                    writer.append_data(frame)
                    i += 1
                print 'Wrote', i , 'frames'

                samps = [0] + samps + [IN_T - 1]
                # print 'Outputting', len(samps), 'samples'
                demoX[s, cntr, :, :7] = traj_angles[samps, :]
                demoX[s, cntr, :, 7:] = traj_efs[samps, :]
                demoU[s, cntr, :, :] = traj_vels[samps, :]

            cntr += 1


        demo_dict = {'demoX':demoX, 'demoU':demoU}
        pkl.dump(demo_dict, open(out_path + '/demo' + str(obj_id) + '.pkl', 'wb'))