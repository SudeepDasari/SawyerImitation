import os
import glob
import numpy as np
import cv2
import cPickle as pkl
data_dir = '~/oneshot_nov_13/'
diag_out = '~/diag_nov_13/'
collector = ['human', 'robot']
T = 40
TRAJ_PER_OBJ = 8


base_path_human = os.path.expanduser(data_dir + 'human/')
objects_human = glob.glob(base_path_human + '*')
objects_human = [p.split(base_path_human)[1] for p in objects_human]

base_path_robot = os.path.expanduser(data_dir + 'robot/')
objects_robot = glob.glob(base_path_robot + '*')
objects_robot = [p.split(base_path_robot)[1] for p in objects_robot]

for o in objects_human:
    assert (o in objects_robot), 'OBJECT ' + o + ' IN HUMAN NOT IN ROBOT'

for o in objects_robot:
    assert (o in objects_human), 'OBJECT ' + o + ' IN ROBOT NOT IN HUMAN'

out_path = os.path.expanduser(diag_out + 'cross_check'+'/')
for o in objects_human:
    print 'cross check', o
    traj_human = glob.glob(base_path_human + o + '/*')[0]
    imgs_human = glob.glob(traj_human + '/images/*.jpg')
    last_fr_human = [i for i in imgs_human if 'im'+str(T - 1) in i][0]
    last_fr_human = cv2.resize(cv2.imread(last_fr_human)[:-150, 225:-225, :], (100, 100), interpolation=cv2.INTER_AREA)

    traj_robot = glob.glob(base_path_robot + o + '/*')[0]
    imgs_robot = glob.glob( traj_robot + '/images/*.jpg')
    last_fr_robot = [i for i in imgs_robot if 'im' + str(T - 1) in i][0]
    last_fr_robot = cv2.resize(cv2.imread(last_fr_robot)[:-150, 225:-225, :], (100, 100), interpolation=cv2.INTER_AREA)

    last_frames = np.concatenate((last_fr_human, last_fr_robot), axis=1)
    cv2.imwrite(out_path + o + '_lasts.jpg', last_frames)


for c in collector:
    base_path = os.path.expanduser(data_dir + c+'/')
    out_path = os.path.expanduser(diag_out + c+'/')
    objects = glob.glob(base_path + '*')
    objects = [p.split(base_path)[1] for p in objects]

    trajs = {obj:sorted(glob.glob(base_path + obj +'/*'), key=lambda x: int(x.split('traj')[1])) for obj in objects}
    for obj_trajs in trajs:
        print 'Loading', c+'/'+obj_trajs
        last_frames = []
        for t in trajs[obj_trajs]:
            imgs = glob.glob(t + '/images/*.jpg')
            last_frame = [i for i in imgs if 'im'+str(T - 1) in i][0]

            if not len(imgs) == T:
                print 'obj', obj_trajs, 'on traj', trajs

            t_name = 'traj' + t.split('traj')[1]
            pkl_dict = pkl.load(open(t + '/joint_' + t_name + '.pkl', 'rb'))
            if not pkl_dict['jointvelocities'].shape == (T, 7):
                print 'ERROR on jv', t_name, 'for object', c, obj_trajs
            if not pkl_dict['jointangles'].shape == (T, 7):
                print 'ERROR ON JA', t_name, 'for object', c, obj_trajs
            if not pkl_dict['endeffector_pos'].shape == (T, 3):
                print 'ERROR ON EF', t_name, 'for object', c, obj_trajs

            last_frame = cv2.imread(last_frame)[:-150, 225:-225, :]
            last_frames.append(cv2.resize(last_frame, (100, 100), interpolation=cv2.INTER_AREA))

        assert (len(last_frames) == TRAJ_PER_OBJ), 'OBJ_TRAJ ' + c+'/'+ obj_trajs + ' HAS ONLY ' + str(len(obj_trajs)) + ' TRAJS'
        last_frames = np.concatenate(last_frames, axis = 1)
        cv2.imwrite(out_path+obj_trajs+'_last.jpg', last_frames)




