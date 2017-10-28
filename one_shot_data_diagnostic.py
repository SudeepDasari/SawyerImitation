import os
import glob
import numpy as np
import cv2

data_dir = '~/demonstration_10_24/'
diag_out = '~/diag_10_24/'
collector = ['human', 'robot']
T = 60
TRAJ_PER_OBJ = 8

for c in collector:
    base_path = os.path.expanduser(data_dir + c+'/')
    out_path = os.path.expanduser(diag_out + c+'/')
    objects = glob.glob(base_path + '*')
    objects = [p.split(base_path)[1] for p in objects]

    trajs = {obj:glob.glob(base_path + obj +'/*') for obj in objects}
    for obj_trajs in trajs:
        print 'Loading', c+'/'+obj_trajs
        last_frames = []
        for t in trajs[obj_trajs]:
            imgs = glob.glob(t + '/images/*.jpg')
            last_frame = [i for i in imgs if 'im'+str(T - 1) in i][0]
            last_frame = cv2.imread(last_frame)[:-150, 225:-225, :]
            last_frames.append(cv2.resize(last_frame, (100, 100), interpolation=cv2.INTER_AREA))
        assert (len(last_frames) == TRAJ_PER_OBJ), 'OBJ_TRAJ ' + obj_trajs + ' HAS ONLY ' + str(len(obj_trajs)) + ' TRAJS'
        last_frames = np.concatenate(last_frames, axis = 1)
        cv2.imwrite(out_path+obj_trajs+'_last.jpg', last_frames)
