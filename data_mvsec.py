import numpy as np
import scipy as sp
from scipy.linalg import logm, expm
from scipy.spatial.transform import Rotation
import os
import h5py
import yaml
import argparse
#import MVSEC_bags


def read_proj_mar(root, seq):
    if(seq == 'outdoor_day1' or seq == 'outdoor_day2'):
        mar_file = os.path.join(root, 'outdoor_day_calib', 'camchain-imucam-outdoor_day.yaml')
    else:
        print('Not Implemented sequences.')
        raise ValueError('Not Implemented sequences.')
        
    with open(mar_file, 'r', encoding='utf-8') as f:
        file = yaml.safe_load(f)
                
        left_mar = file['cam0']['projection_matrix']
        left_mar = np.array(left_mar)
        left_cam = file['cam0']['intrinsics']
        left_cam = np.array([[left_cam[0], 0., left_cam[2]],
                             [0., left_cam[1], left_cam[3]],
                             [0., 0., 1.]])
        left_rect = file['cam0']['rectification_matrix']
        left_rect = np.array(left_rect)
        left_k = file['cam0']['distortion_coeffs']
        left_k = np.array(left_k)
                
        right_mar = file['cam1']['projection_matrix']
        right_mar = np.array(right_mar)

    return left_mar, right_mar, left_cam, left_rect, left_k

def read_rect_maps(root, seq):
    if(seq == 'outdoor_day1' or seq == 'outdoor_day2'):
        x_map_file = os.path.join(root, 'outdoor_day_calib', 'outdoor_day_left_x_map.txt')
        y_map_file = os.path.join(root, 'outdoor_day_calib', 'outdoor_day_left_y_map.txt')
        x_map = np.loadtxt(x_map_file)
        y_map = np.loadtxt(y_map_file)
    else:
        print('Not Implemented sequences.')
        raise ValueError('Not Implemented sequences.')
        
    return x_map, y_map

def dep_to_pc(dep, proj_mar):
    fx = proj_mar[0][0]
    fy = proj_mar[1][1]
    cx = proj_mar[0][2]
    cy = proj_mar[1][2]
    mask = np.isfinite(dep)
    
    x_grid = np.arange(dep.shape[1]).reshape(1, -1)
    x_grid = x_grid.repeat(dep.shape[0], axis=0)
    y_grid = np.arange(dep.shape[0]).reshape(-1, 1)
    y_grid = y_grid.repeat(dep.shape[1], axis=1)
        
    x_grid = (x_grid - cx) * (1./fx)
    y_grid = (y_grid - cy) * (1./fy)
    
    pc_img = np.stack([x_grid, y_grid, dep], axis=2)
    pc = pc_img[mask]
    pc[..., 0] = pc[..., 0] * pc[..., 2]
    pc[..., 1] = pc[..., 1] * pc[..., 2]
    
    return pc

def dep_to_pc_invrect(dep, proj_mar, rect_mar):
    fx = proj_mar[0][0]
    fy = proj_mar[1][1]
    cx = proj_mar[0][2]
    cy = proj_mar[1][2]
    mask = np.isfinite(dep)
    
    x_grid = np.arange(dep.shape[1]).reshape(1, -1)
    x_grid = x_grid.repeat(dep.shape[0], axis=0)
    y_grid = np.arange(dep.shape[0]).reshape(-1, 1)
    y_grid = y_grid.repeat(dep.shape[1], axis=1)
    
    x_grid = (x_grid - cx) * (1./fx)
    y_grid = (y_grid - cy) * (1./fy)
        
    rect_pc_img = np.stack([x_grid, y_grid], axis=2)
    rect_pc = rect_pc_img[mask]
    pc_invrect = inv_rect(rect_pc, rect_mar) #[N, 2]
    pc_invrect = np.concatenate([pc_invrect, dep[mask].reshape(-1, 1)], axis=1)
    pc_invrect[..., 0] = pc_invrect[..., 0] * pc_invrect[..., 2]
    pc_invrect[..., 1] = pc_invrect[..., 1] * pc_invrect[..., 2]
    
    return pc_invrect

def project_to_2d_numpy(xyz, trans_mar, disp=False):

    N, _ = xyz.shape
    xyz_t = np.concatenate([xyz, np.ones([N, 1])], axis=1) 
    xyz_t = xyz_t[:, :, None]#[N, 4, 1]
    back_projected_xyz = np.matmul(trans_mar[None, :, :], xyz_t) #[N, 4, 1]
    w = (back_projected_xyz[:, 3, :])[:, None, :]
    if(disp):
        back_projected_xy = (back_projected_xyz[:, :3, :]) / w #[N, 3, 1] (x, y, d)
    else:
        back_projected_xy = (back_projected_xyz[:, :2, :]) / w #[N, 2, 1] (x, y)
    back_projected_xy = back_projected_xy.squeeze(2)

    return back_projected_xy    

def inv_rect(xy, rect_mar):
    # xy: [N, 2]
    rect_mar = np.linalg.inv(rect_mar)
    
    xy = np.concatenate([xy, np.ones([xy.shape[0], 1])], axis=1)
    xy_invrect = np.matmul(rect_mar[None, :, :], xy[:, :, None])[..., 0] #[N, 3]
    xy_invrect = xy_invrect[:, :2] / xy_invrect[:, 2:]
    
    return xy_invrect

def rect(xy, rect_mar):
    # xy: [N, 2]
    
    xy = np.concatenate([xy, np.ones([xy.shape[0], 1])], axis=1)
    xy_rect = np.matmul(rect_mar[None, :, :], xy[:, :, None])[..., 0] #[N, 3]
    xy_rect = xy_rect[:, :2] / xy_rect[:, 2:]
     
    return xy_rect

def rect2dist(xy, trans_mar, cam_mar, rect_mar, k):
    # xy: [N, 2]
    rect_mar = np.linalg.inv(rect_mar)
    f = trans_mar[2][3]
    cx = -trans_mar[0][3]
    cy = -trans_mar[1][3]
    dfx = cam_mar[0][0]
    dfy = cam_mar[1][1]
    dcx = cam_mar[0][2]
    dcy = cam_mar[1][2]
    
    xyf = np.stack([(xy[:, 0]-cx)/f, (xy[:, 1]-cy)/f, np.ones(xy.shape[0])], axis=1)
    xyf_invrect = np.matmul(rect_mar[None, :, :], xyf[:, :, None])[..., 0]
    xy_invrect = xyf_invrect[:, :2] / xyf_invrect[:, 2:]
    
    r = np.linalg.norm(xy_invrect, axis=1)
    theta = np.arctan(r)
    thetad = theta * (1 + k[0]*(theta**2) +
                      k[1]*(theta**4) +
                      k[2]*(theta**6) +
                      k[3]*(theta**8))
    x_dist = dfx * ((thetad * xy_invrect[:, 0]) / r) + dcx
    y_dist = dfy * ((thetad * xy_invrect[:, 1]) / r) + dcy
    xy_dist = np.stack([x_dist, y_dist], axis=1)
    
    return xy_dist

def dist2rect(xy, trans_mar, cam_mar, rect_mar, k):
    # xy: [N, 2]
    f = trans_mar[2][3]
    cx = -trans_mar[0][3]
    cy = -trans_mar[1][3]
    dfx = cam_mar[0][0]
    dfy = cam_mar[1][1]
    dcx = cam_mar[0][2]
    dcy = cam_mar[1][2]
    
    #xy_undist = cv2.fisheye.undistortPoints(xy[None, :, :], cam_mar, k)
    xy = np.stack([(xy[:, 0] - dcx)/dfx, (xy[:, 1] - dcy)/dfy], axis=1)
    thetad = np.linalg.norm(xy, axis=1)
    theta = thetad
    for i in range(10):
        thetan = 1 + k[0]*(theta**2) + \
        k[1]*(theta**4) + \
        k[2]*(theta**6) + \
        k[3]*(theta**8)
        
        theta = thetad / thetan
        
    x_undist = xy[:, 0] * np.tan(theta)/thetad
    y_undist = xy[:, 1] * np.tan(theta)/thetad
    xy_undist = np.stack([x_undist, y_undist], axis=1)[None, :, :]
    
    xyf_undist = np.concatenate([xy_undist, np.ones([1, xy_undist.shape[1], 1])], axis=2)
    xyf_rect = np.matmul(rect_mar[None, :, :], xyf_undist.transpose(1, 2, 0))[..., 0]
    xy_rect = f * xyf_rect[:, :2] / xyf_rect[:, 2:]
    xy_rect[:, 0] += cx
    xy_rect[:, 1] += cy
    
    return xy_rect

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
  
def compute_op_flow(dep, xy_moved_rect, proj_mar):
    fx = proj_mar[0][0]
    fy = proj_mar[1][1]
    cx = proj_mar[0][2]
    cy = proj_mar[1][2]
    mask = np.isfinite(dep)
    
    x_grid = np.arange(dep.shape[1]).astype('float32').reshape(1, -1)
    x_grid = x_grid.repeat(dep.shape[0], axis=0)
    y_grid = np.arange(dep.shape[0]).astype('float32').reshape(-1, 1)
    y_grid = y_grid.repeat(dep.shape[1], axis=1) 
    
    x_moved_rect = xy_moved_rect[:, 0] * fx + cx
    y_moved_rect = xy_moved_rect[:, 1] * fy + cy
    
    flow_x = np.zeros([dep.shape[0], dep.shape[1]])
    flow_y = np.zeros([dep.shape[0], dep.shape[1]])
    flow_x[mask] = x_moved_rect - x_grid[mask]
    flow_y[mask] = y_moved_rect - y_grid[mask]
    
    flow = np.stack([flow_x, flow_y], axis=2)
    
    return flow, mask
     
def proj_to_trans(left_mar, right_mar):
    f = left_mar[0][0]
    xc = left_mar[0][2]
    yc = left_mar[1][2]
    bf = -right_mar[0][3]
    b = bf / f
    trans_mar = np.array([[1., 0., 0., -xc],
                          [0., 1., 0., -yc],
                          [0., 0., 0., f],
                          [0., 0., 1./b, 0.]])
    
    return trans_mar

def get_rec_events(root, seq, events, timestamps, x_map=None, y_map=None, idx_start=0):    
    # align along the t-axis
    t = events[:, 2] #float64
    t0 = np.floor(t.min())
    t_norm = t - t0
    t_us = (t_norm * 1e6)
    ts_start = ((timestamps[0] - t0) * 1e6)
    ts_end = ((timestamps[1] - t0) * 1e6)
    
    while(t_us[idx_start] < ts_start):
        idx_start += 1
    idx_end = idx_start
    while(t_us[idx_end] < ts_end):
        idx_end += 1
    useful_events = events[idx_start:idx_end, :]
    useful_x = useful_events[:, 0].astype('int64')
    useful_y = useful_events[:, 1].astype('int64')
    
    # rectify events
    rect_useful_events = {}
    if((x_map is None) and (y_map is None)):
        rect_useful_events['x'] = useful_x 
        rect_useful_events['y'] = useful_y 
    else:
        rect_useful_events['x'] = x_map[useful_y, useful_x]
        rect_useful_events['y'] = y_map[useful_y, useful_x]
    rect_useful_events['p'] = useful_events[:, 3]
    rect_useful_events['t'] = t_us[idx_start:idx_end]
    
    rect_useful_events['t'] -= np.min(rect_useful_events['t'])
    
    return rect_useful_events, idx_start, idx_end


# global settings
parser = argparse.ArgumentParser()
parser.add_argument('--mvsec_root', required=True,
                    help='Path to the mvsec dataset')
parser.add_argument('--output', required=True,
                    help='Output path')
par_arg = parser.parse_args()

root = par_arg.mvsec_root
save_dir = os.path.join(par_arg.output, 'Processed_Scene_Flow_New_Rect3')
seqs = ['outdoor_day1', 'outdoor_day2']
nsep = 1

for seq in seqs:
    # load files
    dep_file = os.path.join(root, seq + '_gt.hdf5')
    event_file = os.path.join(root, seq + '_data.hdf5') 
    #odom_file = os.path.join(root, seq + '_gt.bag')
    odom_file = os.path.join(root, seq + '_odom.h5')
    
    deps_and_timestamps = h5py.File(dep_file, mode='r')['davis']['left']
    deps = deps_and_timestamps['depth_image_rect'][:]
    timestamps1 = deps_and_timestamps['depth_image_rect_ts'][:]
    events = h5py.File(event_file, mode='r')['davis']['left']['events']
    left_mar, right_mar, left_cam, left_rect, left_k = read_proj_mar(root, seq)
    trans_mar = proj_to_trans(left_mar, right_mar)
    x_map, y_map = read_rect_maps(root, seq)
    #odom = MVSEC_bags.read_bag_odom(odom_file)
    odom = h5py.File(odom_file, mode='r')
    

    #poss = []
    #rots = []
    #timestamps2 = []
    #for i in range(len(odom)):
    #    P, Q, T = MVSEC_bags.p_q_t_from_msg(odom[i].message)
    #    poss.append(P)
    #    rots.append(Q)
    #    timestamps2.append(T)
    #vs, angs = compute_velocity(poss, rots, timestamps2)
    poss = odom['poss'][:]
    rots = odom['rots'][:]
    vs = odom['vs'][:]
    angs = odom['angs'][:]
    timestamps2 = odom['ts'][:]

    nframe = deps.shape[0]
    event_start_idx = 0
    if(not os.path.exists(os.path.join(save_dir, seq))):
        os.makedirs(os.path.join(save_dir, seq))
    # save calibration info
    np.save(os.path.join(save_dir, seq, 'trans_mar.npy'), trans_mar)
    np.save(os.path.join(save_dir, seq, 'cam_mar.npy'), left_cam)
    np.save(os.path.join(save_dir, seq, 'rect_mar.npy'), left_rect)
    np.save(os.path.join(save_dir, seq, 'distortion.npy'), left_k)
    
    for idx in range(nframe):
        if(idx < nsep):
            continue
        if(idx == nframe-nsep):
            break

        cur_dep = deps[idx]
        cur_timestamp = timestamps1[idx]
        next_dep = deps[idx+nsep]
        next_timestamp = timestamps1[idx+nsep]
        pre_timestamp = timestamps1[idx-nsep]
    
        # align timestamps
        for idx2 in range(len(timestamps2)):
            if(np.abs(cur_timestamp - timestamps2[idx2])*1000 < 1):
                cur_v = vs[idx2]
                cur_ang = angs[idx2]
                cur_pos = poss[idx2]
                cur_rot = rots[idx2]
                break
            else:
                cur_v = None
                cur_ang = None
                cur_pos = None
                cur_rot = None
        if(cur_v is None):
            print('skip one sample due to missed odometry. seq= %s idx= %d' % (seq, idx))
            continue
    
        # depth maps to point clouds
        cur_pc = dep_to_pc(cur_dep, left_mar)
        cur_pc_invrect = dep_to_pc_invrect(cur_dep, left_mar, left_rect)
        next_pc = dep_to_pc(next_dep, left_mar)
    
        # scene flow computation
        dt = next_timestamp - cur_timestamp
        ang_mar = np.array([[0., -cur_ang[2], cur_ang[1]],
                            [cur_ang[2], 0., -cur_ang[0]],
                            [-cur_ang[1], cur_ang[0], 0.]])
        H01 = np.eye(4)
        H01[:3, 3] = cur_v * dt
        H01[:3, :3] = expm(ang_mar * dt)
        H10 = np.linalg.inv(H01)
    
        ex_cur_pc_invrect = np.concatenate([cur_pc_invrect, np.ones([cur_pc_invrect.shape[0], 1])], axis=1)
        ex_cur_pc_invrect_moved = np.matmul(H10, ex_cur_pc_invrect.transpose(1, 0)) #[4, N]
        cur_pc_invrect_moved = ex_cur_pc_invrect_moved[:3, :].transpose(1, 0) #[N, 3]
        cur_xy_moved = rect(cur_pc_invrect_moved[:, :2] / cur_pc_invrect_moved[:, 2:], left_rect) #[N, 2]
        cur_pc_moved = np.concatenate([cur_xy_moved*cur_pc_invrect_moved[:, 2:], cur_pc_invrect_moved[:, 2:]], axis=1)
        scene_flow = cur_pc_moved - cur_pc
        
        scene_flow_norm = np.linalg.norm(scene_flow, axis=1, ord=2)
        mean_scene_flow_norm = np.mean(scene_flow_norm, axis=0)
        if(mean_scene_flow_norm < 0.10 and seq != 'outdoor_day1'):
            print('Mean scene flow norm = %.4f, too small. Skip one sample. seq = %s, idx = %d' % 
                  (mean_scene_flow_norm, seq, idx))
            continue
    
        # optical flow computation
        op_flow, op_mask = compute_op_flow(cur_dep, cur_xy_moved, left_mar)
    
        # get events
        pre_events, event_start_idx, _ = get_rec_events(root, seq, events, [pre_timestamp, cur_timestamp], x_map, y_map, event_start_idx)
        cur_events, _, _ = get_rec_events(root, seq, events, [cur_timestamp, next_timestamp], x_map, y_map, event_start_idx)
    
        # save results
        print('Save one sample. Mean scene flow norm = %.4f, seq= %s idx= %d' % (mean_scene_flow_norm, seq, idx))
        cur_path = os.path.join(save_dir, seq, str(idx))
        if(not os.path.exists(cur_path)):
            os.makedirs(cur_path)
        np.save(os.path.join(cur_path, 'pc1.npy'), cur_pc)
        np.save(os.path.join(cur_path, 'pc2.npy'), next_pc)
        np.save(os.path.join(cur_path, 'flow.npy'), scene_flow)
        np.save(os.path.join(cur_path, 'op_flow.npy'), op_flow)
        np.save(os.path.join(cur_path, 'op_mask.npy'), op_mask)
        

        pre_ev_data = h5py.File(os.path.join(cur_path, 'event0.h5'), mode='w')
        for pre_k, pre_v in pre_events.items():
            pre_ev_data[pre_k] = pre_v
        pre_ev_data.close()
        cur_ev_data = h5py.File(os.path.join(cur_path, 'event.h5'), mode='w')
        for cur_k, cur_v in cur_events.items():
            cur_ev_data[cur_k] = cur_v
        cur_ev_data.close()


    odom.close()
