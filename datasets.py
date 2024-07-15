import os
import h5py
import numpy as np
import torch
import torch.utils.data as data
import utils

def pad_event_opflow(grid0, grid1, op_flow, op_mask, paddings=[4, 6]):
    c, h, w = grid0.shape
    
    grid0_pad_horizontal = np.zeros([c, h, paddings[1]]).astype('float32')
    grid0_pad_vertical = np.zeros([c, paddings[0], w+paddings[1]]).astype('float32')
    grid1_pad_horizontal = np.zeros([c, h, paddings[1]]).astype('float32')
    grid1_pad_vertical = np.zeros([c, paddings[0], w+paddings[1]]).astype('float32')
    flow_pad_horizontal = np.zeros([h, paddings[1], 2]).astype('float32')
    flow_pad_vertical = np.zeros([paddings[0], w+paddings[1], 2]).astype('float32')
    mask_pad_horizontal = np.zeros([h, paddings[1]]).astype('bool')
    mask_pad_vertical = np.zeros([paddings[0], w+paddings[1]]).astype('bool')
    
    grid0 = np.concatenate([grid0, grid0_pad_horizontal], axis=2)
    grid0 = np.concatenate([grid0, grid0_pad_vertical], axis=1)
    grid1 = np.concatenate([grid1, grid1_pad_horizontal], axis=2)
    grid1 = np.concatenate([grid1, grid1_pad_vertical], axis=1)
    op_flow = np.concatenate([op_flow, flow_pad_horizontal], axis=1)
    op_flow = np.concatenate([op_flow, flow_pad_vertical], axis=0)
    op_mask = np.concatenate([op_mask, mask_pad_horizontal], axis=1)
    op_mask = np.concatenate([op_mask, mask_pad_vertical], axis=0)
    
    return grid0, grid1, op_flow, op_mask

def PCHorizontalFlip(xyz1, xyz2, flow, trans_mar, size):
    
    inverse_trans_mar = np.linalg.inv(trans_mar)
    N = xyz1.shape[0]
    one_vector = np.ones([N, 1, 1])
    
    xyz1t = xyz1 + flow
    
    #project to 2d
    xyz1 = np.concatenate([xyz1[:, :, None], one_vector], axis=1)
    xyd1 = np.matmul(inverse_trans_mar[None, :, :], xyz1) #[N, 4, 1]
    w1 = xyd1[:, 3:, :]
    xyd1 = xyd1[:, :3, :] / w1
    
    xyz1t = np.concatenate([xyz1t[:, :, None], one_vector], axis=1)
    xyd1t = np.matmul(inverse_trans_mar[None, :, :], xyz1t) #[N, 4, 1]
    w1t = xyd1t[:, 3:, :]
    xyd1t = xyd1t[:, :3, :] / w1t
    
    xyz2 = np.concatenate([xyz2[:, :, None], one_vector], axis=1)
    xyd2 = np.matmul(inverse_trans_mar[None, :, :], xyz2) #[N, 4, 1]
    w2 = xyd2[:, 3:, :]
    xyd2 = xyd2[:, :3, :] / w2
    
    #horizontal flip
    new_x1 = -1 * xyd1[:, :1, :] + size[1] - 1
    new_xyd1 = np.concatenate([new_x1, xyd1[:, 1:, :]], axis=1)
    
    new_x1t = -1 * xyd1t[:, :1, :] + size[1] - 1
    new_xyd1t = np.concatenate([new_x1t, xyd1t[:, 1:, :]], axis=1)
    
    new_x2 = -1 * xyd2[:, :1, :] + size[1] - 1
    new_xyd2 = np.concatenate([new_x2, xyd2[:, 1:, :]], axis=1)
    
    #new_flow = np.concatenate([-1*flow[:, :1], flow[:, 1:]], axis=1)
    
    #project to 3d
    new_xyd1 = np.concatenate([new_xyd1, one_vector], axis=1) #[N, 4, 1]
    new_xyz1 = np.matmul(trans_mar[None, :, :], new_xyd1)
    w1 = new_xyz1[:, 3:, :]
    new_xyz1 = new_xyz1[:, :3, :] / w1
    new_xyz1 = new_xyz1[:, :, 0]
    
    new_xyd1t = np.concatenate([new_xyd1t, one_vector], axis=1) #[N, 4, 1]
    new_xyz1t = np.matmul(trans_mar[None, :, :], new_xyd1t)
    w1t = new_xyz1t[:, 3:, :]
    new_xyz1t = new_xyz1t[:, :3, :] / w1t
    new_xyz1t = new_xyz1t[:, :, 0]
    
    new_xyd2 = np.concatenate([new_xyd2, one_vector], axis=1) #[N, 4, 1]
    new_xyz2 = np.matmul(trans_mar[None, :, :], new_xyd2)
    w2 = new_xyz2[:, 3:, :]
    new_xyz2 = new_xyz2[:, :3, :] / w2
    new_xyz2 = new_xyz2[:, :, 0]
    
    new_flow = new_xyz1t - new_xyz1
    
    return new_xyz1, new_xyz2, new_flow

def RandomHorizontalFlip_dense(event0, event1, pc1, pc2, flow, op_flow, op_mask, trans_mar, s=[480, 640], p=0.5):
    flag = np.random.random()
    flag = (flag >= p)
    
    if(not flag):
        return event0, event1, pc1, pc2, flow, op_flow, op_mask
    else:
        new_event0 = np.flip(event0, axis=2)
        new_event0 = new_event0.copy().astype('float32')
        new_event1 = np.flip(event1, axis=2)
        new_event1 = new_event1.copy().astype('float32')
        new_op_flow = np.flip(op_flow, axis=1)
        new_op_flow[:, :, 0] = new_op_flow[:, :, 0] * (-1)
        new_op_flow = new_op_flow.copy().astype('float32')
        new_op_mask = np.flip(op_mask, axis=1)
        new_op_mask = new_op_mask.copy().astype('bool')
        new_pc1, new_pc2, new_flow = PCHorizontalFlip(pc1, pc2, flow, trans_mar, size=s)
        new_pc1 = new_pc1.astype('float32')
        new_pc2 = new_pc2.astype('float32')
        new_flow = new_flow.astype('float32')
        
        return new_event0, new_event1, new_pc1, new_pc2, new_flow, new_op_flow, new_op_mask

class DSEC_dense(data.Dataset):
    def __init__(self, npoint, nbin, root='', train=True):
        self.root = os.path.join(root,'Processed_Point_Clouds_Events_Scene_Flow')
        self.npoint = npoint
        self.nbin = nbin
        self.train = train
        
        self.samples = self.get_samples()

        
    def __getitem__(self, index):
        pc1 = np.load(os.path.join(self.samples[index], 'pc1.npy')).astype('float32')
        pc2 = np.load(os.path.join(self.samples[index], 'pc2.npy')).astype('float32')
        flow = np.load(os.path.join(self.samples[index], 'flow.npy')).astype('float32')
        trans_mar = np.load(os.path.join(os.path.abspath(os.path.join(self.samples[index], '..')), 'trans_mar.npy')).astype('float32')
        event_data0 = h5py.File(os.path.join(self.samples[index], 'event0.h5'), mode='r')
        event_data1 = h5py.File(os.path.join(self.samples[index], 'event.h5'), mode='r')
        events0 = {}
        for k0, v0 in event_data0.items():
            events0[k0] = v0[:]
        events1 = {}
        for k1, v1 in event_data1.items():
            events1[k1] = v1[:]
        event_data0.close()
        event_data1.close()

        op_flow = np.load(os.path.join(self.samples[index], 'op_flow.npy')).astype('float32')
        op_mask = np.load(os.path.join(self.samples[index], 'op_mask.npy')).astype('bool')
        

        remote_mask1 = pc1[:, 2] < 35.0
        remote_mask2 = pc2[:, 2] < 35.0
        mask1 = remote_mask1
        mask2 = remote_mask2
        
        pc1 = pc1[mask1]
        flow = flow[mask1]
        pc2 = pc2[mask2]
        
        if(self.npoint > 0):
            try:
                idx1 = np.random.choice(pc1.shape[0], self.npoint, replace=False)
            except ValueError:
                idx1 = np.random.choice(pc1.shape[0], self.npoint, replace=True)
            try:
                idx2 = np.random.choice(pc2.shape[0], self.npoint, replace=False)
            except ValueError:
                idx2 = np.random.choice(pc2.shape[0], self.npoint, replace=True)
            
            pc1 = pc1[idx1, :]
            flow = flow[idx1, :]
            pc2 = pc2[idx2, :]
        
            
        event_grid0 = utils.stream_to_voxel(events0, [480, 640], self.nbin, normalize=True) #[C, H, W]
        event_grid1 = utils.stream_to_voxel(events1, [480, 640], self.nbin, normalize=True)
         
        if(self.train):
            event_grid0, event_grid1, pc1, pc2, flow, op_flow, op_mask = RandomHorizontalFlip_dense(event_grid0, event_grid1, pc1, pc2, flow, op_flow, op_mask, trans_mar, s=[480, 640])

        feat1 = pc1
        feat2 = pc2
        
        return pc1, pc2, feat1, feat2, flow, event_grid0, event_grid1, op_flow, op_mask, trans_mar,
    
    def __len__(self):
        return len(self.samples)
    
    def get_samples(self):
        train_sequences = os.listdir(os.path.join(self.root,'forwardbi'))
        eval_sequences = ['thun_00_a', 'zurich_city_01_a', 'zurich_city_02_a', 'zurich_city_07_a']
        samples = []
        for s in eval_sequences:
            train_sequences.remove(s)
        if(self.train):
            for ts in train_sequences:
                for cur_root, cur_dirs, cur_files in os.walk(os.path.join(self.root, 'forwardbi', ts)):
                    if(len(cur_dirs) == 0):
                        samples.append(cur_root)
            print('length of training set:', len(samples))
            assert len(samples) == 7868
        else:
            for es in eval_sequences:
                for cur_root, cur_dirs, cur_files in os.walk(os.path.join(self.root, 'forwardbi', es)):
                    if(len(cur_dirs) == 0):
                        samples.append(cur_root)
            print('length of evaluation set:', len(samples))
            assert len(samples) == 302
            
        return samples

class MVSEC_dense(data.Dataset):
    def __init__(self, npoint, nbin, root='', train=True, pad=True):
        self.root = os.path.join(root,'Processed_Scene_Flow_New_Rect3')
        self.npoint = npoint
        self.nbin = nbin
        self.train = train
        self.pad = pad
        
        self.samples = self.get_samples()
        
        
    def __getitem__(self, index):
        pc1 = np.load(os.path.join(self.samples[index], 'pc1.npy')).astype('float32')
        pc2 = np.load(os.path.join(self.samples[index], 'pc2.npy')).astype('float32')
        flow = np.load(os.path.join(self.samples[index], 'flow.npy')).astype('float32')
        trans_mar = np.load(os.path.join(os.path.abspath(os.path.join(self.samples[index], '..')), 'trans_mar.npy')).astype('float32')
        event_data0 = h5py.File(os.path.join(self.samples[index], 'event0.h5'), mode='r')
        event_data1 = h5py.File(os.path.join(self.samples[index], 'event.h5'), mode='r')
        events0 = {}
        for k0, v0 in event_data0.items():
            events0[k0] = v0[:]
        events1 = {}
        for k1, v1 in event_data1.items():
            events1[k1] = v1[:]
        event_data0.close()
        event_data1.close()

        op_flow = np.load(os.path.join(self.samples[index], 'op_flow.npy')).astype('float32')
        op_mask = np.load(os.path.join(self.samples[index], 'op_mask.npy')).astype('bool')
        op_mask[192:, :] = False  # remove the car hood
       
        remote_mask1 = pc1[:, 2] < 35.0
        remote_mask2 = pc2[:, 2] < 35.0
        mask1 = remote_mask1
        mask2 = remote_mask2
        
        pc1 = pc1[mask1]
        flow = flow[mask1]
        pc2 = pc2[mask2]
        
        if(self.npoint > 0):
            try:
                idx1 = np.random.choice(pc1.shape[0], self.npoint, replace=False)
            except ValueError:
                idx1 = np.random.choice(pc1.shape[0], self.npoint, replace=True)
            try:
                idx2 = np.random.choice(pc2.shape[0], self.npoint, replace=False)
            except ValueError:
                idx2 = np.random.choice(pc2.shape[0], self.npoint, replace=True)
            
            pc1 = pc1[idx1, :]
            flow = flow[idx1, :]
            pc2 = pc2[idx2, :]
        
        event_grid0 = utils.stream_to_voxel(events0, [260, 346], self.nbin, normalize=True) #[C, H, W]
        event_grid1 = utils.stream_to_voxel(events1, [260, 346], self.nbin, normalize=True)

        if(self.train):
            event_grid0, event_grid1, pc1, pc2, flow, op_flow, op_mask = RandomHorizontalFlip_dense(event_grid0, event_grid1, pc1, pc2, flow, op_flow, op_mask, trans_mar, s=[260, 346])

        else:
            edge_h = int((260-256) / 2)
            edge_w = int((346-256) / 2)
            op_mask[0:edge_h, :] = False
            op_mask[:, 0:edge_w] = False
            op_mask[-edge_h:, :] = False
            op_mask[:, -edge_w:] = False
        if(self.pad):
            event_grid0, event_grid1, op_flow, op_mask = pad_event_opflow(event_grid0, event_grid1, op_flow, op_mask, paddings=[4, 6]) #[264, 352]
        feat1 = pc1
        feat2 = pc2
        
        return pc1, pc2, feat1, feat2, flow, event_grid0, event_grid1, op_flow, op_mask, trans_mar

    def __len__(self):
        return len(self.samples)
    
    def get_samples(self):
        train_sequences = ['outdoor_day2']
        eval_sequences = ['outdoor_day1'] 
        eval_idx = list(range(4356, 4706))
        samples = []
        if(self.train):
            for ts in train_sequences:
                for cur_root, cur_dirs, cur_files in os.walk(os.path.join(self.root, ts)):
                    if(len(cur_dirs) == 0):
                        samples.append(cur_root)
            assert len(samples) == 9967
        else:
            for es in eval_sequences:
                for idx in eval_idx:
                    cur_root = os.path.join(self.root, es, str(idx))
                    samples.append(cur_root)
            assert len(samples) == 350
            
        return samples
