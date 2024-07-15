import numpy as np
import os
import yaml
from collections import defaultdict
import argparse

import cv2
import imageio as imageio
import h5py

class data_zip():
    def __init__(self, path, scene, timestamp, data=None):
        self.path = path
        self.data = data
        self.timestamp = timestamp
        self.scene = scene
        
        
def read_op_flow(root='./train_optical_flow', valid_scenes=[], us=False):
    content = []
    for cur_root, cur_dirs, cur_files in os.walk(root):
        if('forward_timestamps.txt' in cur_files or 'backward_timestamps.txt' in cur_files):
            timestamps_path = os.path.join(cur_root, 'forward_timestamps.txt')
            scene_name = timestamps_path.split('/')[-3]
            if(scene_name not in valid_scenes):
                continue
                
            if(us):
                timestamps = np.loadtxt(timestamps_path, delimiter=',', comments='#', dtype='int64')
            else:
                timestamps = np.loadtxt(timestamps_path, delimiter=',', comments='#', dtype='int64') // 1000 #ms

            flow_data_names = os.listdir(os.path.join(cur_root, 'forward'))
            flow_data_names.sort()
            assert len(flow_data_names) == timestamps.shape[0]
            
            for i in range(len(flow_data_names)):
                path = os.path.join(cur_root,'forward', flow_data_names[i])
                stamp = timestamps[i, :]
                
                flow_16bit = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                flow_16bit = np.flip(flow_16bit, axis=-1)
                mask = flow_16bit[..., 2] == 1 #[h, w]
                flow_16bit = flow_16bit.astype('float32')
                
                flow_x = (flow_16bit[..., 0] - 2**15)/128.0
                flow_y = (flow_16bit[..., 1] - 2**15)/128.0
                flow_map = np.array([flow_x, flow_y]).transpose(1, 2, 0) #[h, w, 2] (x, y)
                
                flow_mask = [flow_map, mask]
                
                content.append(data_zip(path, scene_name, stamp, flow_mask))
                
    return content


def read_disp(root='./train_disparity', valid_scenes=[], us=False):
    content = []
    for cur_root, cur_dirs, cur_files in os.walk(root):
        if('timestamps.txt' in cur_files):
            timestamps_path = os.path.join(cur_root, 'timestamps.txt')
            scene_name = timestamps_path.split('/')[-3]
            if(scene_name not in valid_scenes):
                continue
            if(us):
                timestamps = np.loadtxt(timestamps_path, delimiter=',', dtype='int64')
            else:
                timestamps = np.loadtxt(timestamps_path, delimiter=',', dtype='int64') // 1000 #ms
            disp_data_names = os.listdir(os.path.join(cur_root, 'event'))
            disp_data_names.sort()
            assert len(disp_data_names) == timestamps.shape[0]
            
            for i in range(len(disp_data_names)):
                path = os.path.join(cur_root, 'event', disp_data_names[i])
                stamp = timestamps[i]
                
                disp_16bit = cv2.imread(path, cv2.IMREAD_ANYDEPTH) #[h, w]
                disp = disp_16bit.astype('float32') / 256.0
                
                content.append(data_zip(path, scene_name, stamp, disp))
            
    return content

def read_event(root='./train_events', timestamps=np.zeros([100, 2]), valid_scenes=[]):
    content = []
    for cur_root, cur_dirs, cur_files in os.walk(root):
        if('left' in cur_dirs):
            scene_name = cur_root.split('/')[-2]
            if(scene_name not in valid_scenes):
                continue
            event_data = h5py.File(os.path.join(cur_root, 'left', 'events.h5'), mode='r')
            rectify_data = h5py.File(os.path.join(cur_root, 'left', 'rectify_map.h5'), mode='r')
            re_map = rectify_data['rectify_map'][:]
            t0 = event_data['t_offset'][()]
            timestamps_us_ev = (timestamps - t0).astype('int64') #us
            timestamps_ms_ev = (timestamps_us_ev // 1000).astype('int64') #ms

            for i in range(timestamps.shape[0]):
                start_idx_us = event_data['ms_to_idx'][timestamps_ms_ev[i, 0]]
                end_idx_us = event_data['ms_to_idx'][timestamps_ms_ev[i, 1]]
                while(event_data['events']['t'][start_idx_us] < timestamps_us_ev[i, 0]):
                    start_idx_us += 1
                    start_idx_us = start_idx_us.astype('uint64')
                while(event_data['events']['t'][end_idx_us] < timestamps_us_ev[i, 1]):
                    end_idx_us += 1
                    end_idx_us = end_idx_us.astype('uint64')
                
                useful_events = {}
                for key, value in event_data['events'].items():
                    useful_events[key] = value[start_idx_us:end_idx_us]
                
                ori_x = useful_events['x']
                ori_y = useful_events['y']
                re_xy = re_map[ori_y, ori_x]
                useful_events['x'] = re_xy[:, 0]
                useful_events['y'] = re_xy[:, 1]
                useful_events['p'] = useful_events['p'].astype('int8')
                useful_events['p'][useful_events['p'] == 0] = -1
                useful_events['t'] -= np.min(useful_events['t'])
            
                content.append(data_zip(os.path.join(cur_root, 'left', 'events.h5'), scene_name, timestamps[i], useful_events))
                
            event_data.close()
            rectify_data.close()
            
    return content

def read_cam_mar(root='./train_calibration', valid_scenes=[]):
    content = []
    for cur_root, cur_dirs, cur_files in os.walk(root):
        if(len(cur_dirs) == 0):
            path = os.path.join(cur_root, 'cam_to_cam.yaml')
            scene_name = path.split('/')[-3]
            if(len(valid_scenes) > 0):
                if(scene_name not in valid_scenes):
                    continue
            with open(path, 'r', encoding='utf-8') as f:
                file = yaml.safe_load(f)
                cam_mar = file['disparity_to_depth']['cams_03']
                cam_mar = np.array(cam_mar)
                print(cam_mar.shape)
                
            content.append(data_zip(path, scene_name, 0, cam_mar))
            
    return content
        
def align_disp_flow_event(root, valid_scenes=[]):
    assert len(valid_scenes) == 1
    disp_root = os.path.join(root, 'train_disparity')
    flow_root = os.path.join(root, 'train_optical_flow')
    event_root = os.path.join(root, 'train_events')
    
    disp_data = read_disp(root=disp_root, valid_scenes=valid_scenes, us=True)
    print('disp OK...')
    flow_data = read_op_flow(root=flow_root, valid_scenes=valid_scenes, us=True)
    print('op_flow OK...')
    disp_data.sort(key=lambda x: x.timestamp)
    flow_data.sort(key=lambda x: x.timestamp[0])
    
    event_timestamps = np.zeros([len(disp_data)-1, 2])
    for k in range(len(disp_data)-1):
        event_timestamps[k] = np.array([disp_data[k].timestamp, disp_data[k+1].timestamp])
    event_data = read_event(root=event_root, timestamps=event_timestamps, valid_scenes=valid_scenes)
    print('events OK...')
    event_data.sort(key=lambda x: x.timestamp[0])
    
    aligned_data = defaultdict(lambda:list())
    begin = 0
    for flow in flow_data:
        cur_data={}
        cur_data['timestamp'] = flow.timestamp
        cur_data['flow'] = flow.data
        cur_data['flow_path'] = flow.path
        
        for i in range(begin, len(disp_data)):
            disp = disp_data[i]
            if(flow.scene == disp.scene):
                if(disp.timestamp // 1000 == cur_data['timestamp'][0] // 1000):
                    assert (cur_data['timestamp'] // 1000 == event_data[i].timestamp // 1000).all()
                    begin = i
                    cur_data['disp1'] = disp.data
                    cur_data['disp1_path'] = disp.path
                    cur_data['event'] = event_data[i].data
                    cur_data['event_path'] = event_data[i].path
                    if(i > 0):
                        cur_data['exten_timestamp'] = np.insert(cur_data['timestamp'], 0,  disp_data[i-1].timestamp, axis=0)
                        cur_data['pre_event'] = event_data[i-1].data
                if(disp.timestamp // 1000 == cur_data['timestamp'][1] // 1000):
                    cur_data['disp2'] = disp.data
                    cur_data['disp2_path'] = disp.path
                    break
                    
        
        aligned_data[flow.scene].append(cur_data)
                
    return aligned_data




from utils import compute_scene_flow, disp_to_pc

parser = argparse.ArgumentParser()
parser.add_argument('--dsec_root', required=True,
                    help='Path to the dsec dataset')
parser.add_argument('--output', required=True,
                    help='Output path')
par_arg = parser.parse_args()

scenes = os.listdir(os.path.join(par_arg.dsec_root, 'train_optical_flow'))
illegal = []
for ill_name in illegal:
    scenes.remove(ill_name)

for name in scenes:
    cur_root = par_arg.dsec_root
    cur_flow_disp_event = align_disp_flow_event(cur_root, [name])
    cur_trans_mar = read_cam_mar(os.path.join(cur_root, 'train_calibration'), valid_scenes=[name])

    cur_root = os.path.join(par_arg.output, 'Processed_Point_Clouds_Events_Scene_Flow', 'forwardbi', name)
    if(not os.path.exists(cur_root)):
        os.makedirs(cur_root)       
    np.save(os.path.join(cur_root, 'trans_mar.npy'), cur_trans_mar[0].data)
    
    for k, v in cur_flow_disp_event.items(): #k: scene name, v: data list
        idx = 1
        for fd_data in v:
            op_flow = fd_data['flow'][0]
            op_flow_mask = fd_data['flow'][1]
            disp1 = fd_data['disp1']
            disp1_mask = disp1 > 0
            disp2 = fd_data['disp2']
            disp2_mask = disp2 > 0
            event = fd_data['event']
            event0 = fd_data['pre_event']
            timestamp = fd_data['exten_timestamp']
            
            print('transform to 3d...')
            points1, sc_flow = compute_scene_flow(disp1, disp2, disp1_mask, disp2_mask,
                                              op_flow, op_flow_mask, cur_trans_mar[0].data)
            points2 = disp_to_pc(disp2, disp2_mask, cur_trans_mar[0].data)
        
            cur_path = os.path.join(cur_root, str(idx))
            if(not os.path.exists(cur_path)):
                os.makedirs(cur_path)
            np.save(os.path.join(cur_path, 'pc1.npy'), points1)
            np.save(os.path.join(cur_path, 'pc2.npy'), points2)
            np.save(os.path.join(cur_path, 'flow.npy'), sc_flow)
            np.save(os.path.join(cur_path, 'op_mask.npy'), op_flow_mask)
            np.save(os.path.join(cur_path, 'op_flow.npy'), op_flow)
            np.save(os.path.join(cur_path, 'timestamp.npy'), timestamp)
            
            event_data = h5py.File(os.path.join(cur_path, 'event.h5'), mode='w')
            for ke, ve in event.items():
                event_data[ke] = ve
            event_data.close()
            event0_data = h5py.File(os.path.join(cur_path, 'event0.h5'), mode='w')
            for ke0, ve0 in event0.items():
                event0_data[ke0] = ve0
            event0_data.close()
       
            idx += 1
        