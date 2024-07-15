import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm
import collections
import os
import argparse

from models import *
import models
import datasets as ds
import utils
import cmd_args


def test_process(args):
    model = PE_Flow3d_dense_heavy2_align(args.npoint, args.nbin, args.size)
    
    model.cuda()
     
    model.load_state_dict(torch.load(args.pretrain))
    print('Load checkpoint: %s' % args.pretrain)

    if(args.dataset == 'DSEC'):
        eval_set = ds.DSEC_dense(args.npoint, args.nbin, args.data_root, False)
        eval_loader = torch.utils.data.DataLoader(eval_set, batch_size=1, shuffle=False,
                                                   num_workers=8, pin_memory=True,
                                                   worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 **32)))
    elif(args.dataset == 'MVSEC'):
        eval_set = ds.MVSEC_dense(args.npoint, args.nbin, args.data_root, False, True)
        eval_loader = torch.utils.data.DataLoader(eval_set, batch_size=1, shuffle=False,
                                                   num_workers=8, pin_memory=True,
                                                   worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 **32)))

    else:
        raise ValueError('Not implemented dataset %s' % args.dataset)
        
    
    eval_3d, eval_2d, eval_time = eval_model(model, eval_loader, args)

    print(eval_3d)
    print(eval_2d)
    print('total time= %f ms' % (eval_time))

    return eval_3d, eval_2d

    
def eval_model(model, loader, args):
    model = model.eval()
        
    total_seen = 0
    total_time = 0
    total_2d = {'epe': np.zeros([args.eval_iter], dtype='float32'),
               '1px': np.zeros([args.eval_iter], dtype='float32'),
               '3px': np.zeros([args.eval_iter], dtype='float32')}
    total_3d = {'epe': np.zeros([args.eval_iter], dtype='float32'),
               'accs': np.zeros([args.eval_iter], dtype='float32'),
               'outliers': np.zeros([args.eval_iter], dtype='float32')}
    
    
    tik = torch.cuda.Event(enable_timing=True)
    tok = torch.cuda.Event(enable_timing=True)
    for i, data in tqdm(enumerate(loader, 0), total=len(loader), smoothing=0.8):
        pc1, pc2, feat1, feat2, flow, events1, events2, op_flow, op_mask, trans_mar = data
        pc1 = pc1.cuda()
        pc2 = pc2.cuda()
        feat1 = feat1.cuda()
        feat2 = feat2.cuda()
        flow = flow.cuda()
        events1 = events1.cuda()
        events2 = events2.cuda()
        op_flow = op_flow.cuda()
        op_mask = op_mask.cuda()
        trans_mar = trans_mar.cuda()
        
        cur_bs = flow.size()[0]
        with torch.no_grad():
            tik.record()
            pred_3d_flows, pred_2d_flows = model(pc1, pc2, feat1, feat2, events1, events2, trans_mar, args.eval_iter)
            tok.record()
            torch.cuda.synchronize()
            total_time += tik.elapsed_time(tok)
            
            epe3d, accs, out3d = utils.compute_3d_metric(pred_3d_flows, flow.permute(0, 2, 1).contiguous())
            epe2d, onepx, threepx = utils.compute_2d_metric(pred_2d_flows, op_mask, op_flow.permute(0, 3, 1, 2).contiguous())
            
            total_seen += cur_bs
            total_2d['epe'] += epe2d
            total_2d['1px'] += onepx
            total_2d['3px'] += threepx
            total_3d['epe'] += epe3d
            total_3d['accs'] += accs
            total_3d['outliers'] += out3d
        
    print('total samples: %d' % total_seen)        
    for k1,v1 in total_3d.items():
        total_3d[k1] = v1 / total_seen
    for k2,v2 in total_2d.items():
        total_2d[k2] = v2 / total_seen
        
    return total_3d, total_2d, total_time
            
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True,
                        help='Path to the config file (yaml)')
    par_arg = parser.parse_args()
    
    torch.set_printoptions(profile='full')
    root = os.path.dirname(os.path.abspath(__file__))
    #args = cmd_args.parse_args_from_yaml(os.path.join(root, 'test_cfg.yaml'))
    args = cmd_args.parse_args_from_yaml(par_arg.config)
    
    eval_3d, eval_2d = test_process(args)
    