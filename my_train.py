# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 19:55:32 2023

@author: lihl
"""

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import os
import collections
import sys
import logging
import random
import argparse

from models import *
import models
import datasets as ds
import utils
import cmd_args


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()
        
    def preload(self):
        try:
            self.next_data = next(self.loader)
        except StopIteration:
            self.next_data = None
            return
        with torch.cuda.stream(self.stream):
            self.next_data = (item.cuda(non_blocking=True) for item in self.next_data)
            
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        self.preload()
        return data

def training_process(args, logger, chk_dir):
    model = PE_Flow3d_dense_heavy2_align(args.npoint, args.nbin, args.size)
    
    if(args.multi_gpu):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        model.cuda()
        model = nn.DataParallel(model)
    else:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        model.cuda()
        
    if(args.pretrain is None):
        print('Training from scratch')
        logger.info('Training from scratch')
        init_epoch = 0
    else:
        model.module.load_state_dict(torch.load(args.pretrain))
        print('Load checkpoint: %s' % args.pretrain)
        logger.info('Load checkpoint: %s' % args.pretrain)
        init_epoch = args.init_epoch
        
    if(args.optimizer == 'SGD'):
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    elif(args.optimizer == 'Adam'):
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999),
                                      eps=1e-8, weight_decay=args.weight_decay)
    elif(args.optimizer == 'AdamW'):
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, eps=1e-8)
    else:
        logger.info('Not implemented optimizer...')
        raise ValueError('Not implemented optimizer')
    
    if(args.dataset == 'DSEC'):
        train_set = ds.DSEC_dense(args.npoint, args.nbin, args.data_root, True)
        eval_set = ds.DSEC_dense(args.npoint, args.nbin, args.data_root, False)
        
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=8, pin_memory=True,
                                                   worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 **32)))
        eval_loader = torch.utils.data.DataLoader(eval_set, batch_size=1, shuffle=False,
                                                   num_workers=8, pin_memory=True,
                                                   worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 **32)))
        logger.info('Train Dataset:'+args.dataset)
        logger.info('Eval Dataset:'+args.dataset)
        
    elif(args.dataset == 'MVSEC'):
        train_set = ds.MVSEC_dense(args.npoint, args.nbin, args.data_root, True, True)
        eval_set = ds.MVSEC_dense(args.npoint, args.nbin, args.data_root, False, True)
        
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=8, pin_memory=True,
                                                   worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 **32)))
        eval_loader = torch.utils.data.DataLoader(eval_set, batch_size=1, shuffle=False,
                                                   num_workers=8, pin_memory=True,
                                                   worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 **32)))
        logger.info('Train Dataset:'+args.dataset)
        logger.info('Eval Dataset:'+args.dataset)
        
    else:
        logger.info('Not implemented dataset...')
        raise ValueError('Not implemented dataset')

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=args.milestones,
            gamma=0.5,
            last_epoch=-1
        )
    
    best_epe = np.inf
    tik = torch.cuda.Event(enable_timing=True)
    tok = torch.cuda.Event(enable_timing=True)
    for e in range(init_epoch, args.epochs):
        cur_lr = max(optimizer.param_groups[0]['lr'], 1e-5)
        print('learning rate=', cur_lr)
        logger.info('learning rate='+str(cur_lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = cur_lr
            
        total_loss = 0
        total_seen = 0
        
        prefetcher = data_prefetcher(train_loader)
        i = 0
        data = prefetcher.next()
        while(data is not None):
            tik.record()
            print('it=', i, '/', len(train_loader))
            model = model.train()
            
            pc1, pc2, feat1, feat2, flow, events1, events2, op_flow, op_mask, trans_mar = data
            cur_bs = flow.size()[0]
            
            optimizer.zero_grad()    
            pred_3d_flows, pred_2d_flows = model(pc1, pc2, feat1, feat2, events1, events2, trans_mar, args.train_iter)
            alpha = [0.8 ** (args.train_iter - citer)  for citer in range(args.train_iter)]
            loss_3d = models.sequence_loss(pred_3d_flows, flow.permute(0, 2, 1).contiguous(), alpha)
            loss_2d = models.sequence_loss2d(pred_2d_flows, op_flow.permute(0, 3, 1, 2).contiguous(), op_mask, alpha)
            loss = loss_3d + args.op_loss_weight * loss_2d
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.detach().cpu().data * cur_bs
            total_seen += cur_bs
            
            i += 1
            data = prefetcher.next()
            tok.record()
            torch.cuda.synchronize()
            print('time=', tik.elapsed_time(tok)/1000.0, 's/it')
            print('loss=', loss.detach())
            
        scheduler.step()
        
        print('total samples:', total_seen)
        train_loss = total_loss / total_seen
        print('EPOCH %d mean training loss: %f' % (e, train_loss))
        logger.info('EPOCH %d mean training loss: %f' % (e, train_loss))
        
        eval_loss, eval_epe, eval_epe_2d = eval_model(model, eval_loader, args, e)
        for j in range(eval_epe.shape[0]):
            print('EPOCH %d, iter %d, eval epe3d=%f, eval epe2d=%f' % (e, j+1, eval_epe[j], eval_epe_2d[j]))
            logger.info('EPOCH %d, iter %d, eval epe3d=%f, eval epe2d=%f' % (e, j+1, eval_epe[j], eval_epe_2d[j]))
        last_epe = eval_epe_2d[-1]
        if(last_epe < best_epe):
            best_epe = last_epe
            if(args.multi_gpu):
                torch.save(model.module.state_dict(), '%s/%s_%.3d_%.4f.pth' % (chk_dir, args.model_name, e, last_epe))
            else:
                torch.save(model.state_dict(), '%s/%s_%.3d_%.4f.pth' % (chk_dir, args.model_name, e, last_epe))
            print('Save model...')
            logger.info('Save model...')
            print('Best epe2d: %f' % best_epe)
            logger.info('Best epe2d: %f' % best_epe)
            
    return best_epe
    
def eval_model(model, loader, args, e):
    model = model.eval()
    
    total_loss = 0
    total_seen = 0
    total_epe = np.zeros([args.eval_iter], dtype='float32')
    total_epe_2d = np.zeros([args.eval_iter], dtype='float32')
    
    prefetcher = data_prefetcher(loader)
    i = 0
    data = prefetcher.next()
    with torch.no_grad():
        while(data is not None):
            print('it=', i, '/', len(loader))
            pc1, pc2, feat1, feat2, flow, events1, events2, op_flow, op_mask, trans_mar = data
            cur_bs = flow.size()[0]
            
            pred_3d_flows, pred_2d_flows = model(pc1, pc2, feat1, feat2, events1, events2, trans_mar, args.eval_iter)
            alpha = [0.8 ** (args.eval_iter - citer)  for citer in range(args.eval_iter)]
            loss_3d = models.sequence_loss(pred_3d_flows, flow.permute(0, 2, 1).contiguous(), alpha)
            loss_2d = models.sequence_loss2d(pred_2d_flows, op_flow.permute(0, 3, 1, 2).contiguous(), op_mask, alpha)
            loss = loss_3d + args.op_loss_weight * loss_2d
            
            epe3d = utils.compute_epe(pred_3d_flows, flow.permute(0, 2, 1).contiguous())
            epe2d = utils.compute_2depe(pred_2d_flows, op_mask, op_flow.permute(0, 3, 1, 2).contiguous())
            
            total_loss += loss * cur_bs
            total_seen += cur_bs
            total_epe += epe3d
            total_epe_2d += epe2d
            
            i += 1
            data = prefetcher.next()
            
    print('total samples:', total_seen)
    mean_loss = total_loss / total_seen
    mean_epe = total_epe / total_seen
    mean_epe_2d = total_epe_2d / total_seen
    
    return mean_loss, mean_epe, mean_epe_2d
      
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True,
                        help='Path to the config file (yaml)')
    par_arg = parser.parse_args()
    
    root = os.path.dirname(os.path.abspath(__file__))
    #args = cmd_args.parse_args_from_yaml(os.path.join(root, 'train_cfg_mvsec.yaml'))
    args = cmd_args.parse_args_from_yaml(par_arg.config)
    
    exp_name = args.exp_name
    exp_dir = os.path.join(root, 'experiments', exp_name)
    log_dir = os.path.join(exp_dir, 'logs')
    chk_dir = os.path.join(exp_dir, 'checkpoints')
    
    if(not os.path.exists(exp_dir)):
        os.makedirs(exp_dir)
    if(not os.path.exists(chk_dir)):
        os.makedirs(chk_dir)
    if(not os.path.exists(log_dir)):
        os.makedirs(log_dir)
        
    files_to_save = ['utils.py', 'models.py', 'layers.py', 'my_train.py', 'datasets.py', 
                     'configs/train_mvsec_cfg.yaml', 'configs/train_dsec_cfg.yaml']
    for fname in files_to_save:
        os.system('cp %s %s' % (os.path.join(root, fname), log_dir))
    
    logger = logging.getLogger(args.model_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(log_dir + '/train_%s.txt' % args.model_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    print('BEGIN TRAINING...')
    logger.info('BEGIN TRAINING...')
    logger.info(args)
    
    best_epe = training_process(args, logger, chk_dir)
    print('FINISH TRAINING...')
    logger.info('FINISH TRAINING')