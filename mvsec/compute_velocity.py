import numpy as np
import scipy as sp
from scipy.linalg import logm, expm
from scipy.spatial.transform import Rotation
import os
import h5py
import yaml
import argparse
import MVSEC_bags


def compute_velocity(poss, rots, ts, filter_size=10):
    nframe = len(ts)
    vs = np.zeros([nframe, 3])
    angs = np.zeros([nframe, 3])
    H0 = None
    for idx1 in range(nframe):
        H1 = np.eye(4)
        H1[:3, :3] = Rotation.from_quat(rots[idx1]).as_matrix()
        H1[:3, 3] = poss[idx1]
        if(H0 is not None):
            H01 = np.matmul(np.linalg.inv(H0), H1)
            dt = ts[idx1] - ts[idx1-1]
            v = H01[:3, 3] / dt
            w = logm(H01[:3, :3]) / dt
            ang = np.array([w[2, 1], w[0, 2], w[1, 0]])
            
            vs[idx1, :] = v
            angs[idx1, :] = ang
            
        H0 = H1
        
    sm_vs = vs
    sm_angs = angs
    for idx2 in range(nframe):
        if(idx2-filter_size < 0):
            sm_vs[idx2, :] = np.mean(vs[0:idx2+filter_size+1, :], axis=0)
            sm_angs[idx2, :] = np.mean(angs[0:idx2+filter_size+1, :], axis=0)
        elif(idx2+filter_size >= nframe):
            sm_vs[idx2, :] = np.mean(vs[idx2-filter_size:nframe, :], axis=0)
            sm_angs[idx2, :] = np.mean(angs[idx2-filter_size:nframe, :], axis=0)
        else:
            sm_vs[idx2, :] = np.mean(vs[idx2-filter_size:idx2+filter_size+1, :], axis=0)
            sm_angs[idx2, :] = np.mean(angs[idx2-filter_size:idx2+filter_size+1, :], axis=0)
            
    return sm_vs, sm_angs


parser = argparse.ArgumentParser()
parser.add_argument('--mvsec_root', required=True,
                    help='Path to the mvsec dataset')
parser.add_argument('--output', required=True,
                    help='Output path')
par_arg = parser.parse_args()

root = par_arg.mvsec_root #'D:\\AI\\visual\\mv' par_arg.mvsec_root
save_dir = par_arg.output #'F:\\mvsec' par_arg.output
seqs = ['outdoor_day1', 'outdoor_day2']

for seq in seqs:
    # load files
    odom_file_read = os.path.join(root, seq + '_gt.bag')
    odom_file_save = os.path.join(save_dir, seq + '_odom.h5')
    

    odom = MVSEC_bags.read_bag_odom(odom_file_read)
    
    poss = []
    rots = []
    timestamps2 = []
    for i in range(len(odom)):
        P, Q, T = MVSEC_bags.p_q_t_from_msg(odom[i].message)
        poss.append(P)
        rots.append(Q)
        timestamps2.append(T)
    vs, angs = compute_velocity(poss, rots, timestamps2)
    
    with h5py.File(odom_file_save, mode='w') as file_to_save:
        file_to_save['poss'] = np.array(poss)
        file_to_save['rots'] = np.array(rots)
        file_to_save['ts'] = np.array(timestamps2)
        file_to_save['vs'] = vs
        file_to_save['angs'] = angs
    