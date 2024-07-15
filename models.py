import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pointnet2 import pointnet2_utils

import layers
import utils

class PC_encoder(nn.Module):
    def __init__(self, npoint=[], radius=[], nsample=[], nfeat_d=[], nfeat_u=[], upsample=True):
        super(PC_encoder, self).__init__()
        self.upsample = upsample
        
        last_feat = 3
        self.down_path = nn.ModuleList()
        for i in range(len(nfeat_d)):
            self.down_path.append(layers.SetConv(radius[i], npoint[i], nsample[i], last_feat, [nfeat_d[i], nfeat_d[i]]))
            last_feat = nfeat_d[i]
        
        if(upsample):
            self.up_path = nn.ModuleList()
            for j in range(len(nfeat_u)-1):
                self.up_path.append(layers.SetUpConv(radius[len(npoint)+j], nsample[len(npoint)+j], last_feat, nfeat_d[-2-j], [nfeat_u[j]], [nfeat_u[j]]))
                last_feat = nfeat_u[j]
            if(len(nfeat_u) == len(nfeat_d)):
                self.up_path.append(layers.SetUpConv(radius[-1], nsample[-1], last_feat, 3, [nfeat_u[-1],], [nfeat_u[-1]]))
            else:
                self.up_path.append(layers.SetUpConv(radius[-1], nsample[-1], last_feat, nfeat_d[-1-len(nfeat_u)], [nfeat_u[-1],], [nfeat_u[-1]]))
            last_feat = nfeat_u[-1]
        
    def forward(self, xyz, feat):
        featd = [feat]
        xyzd = [xyz]
        idxd = []
        for layerd in self.down_path:
            cur_feat, cur_xyz, cur_idx = layerd(xyzd[-1], featd[-1])
            featd.append(cur_feat)
            xyzd.append(cur_xyz)
            if(len(idxd) == 0):
                idxd.append(cur_idx)
            else:
                idxd.append(torch.gather(idxd[-1], 1, cur_idx.long()))

        if(not self.upsample):
            return featd[-1], xyzd[-1], idxd[-1].contiguous()
    
        featu = [featd[-1]]
        i = -1
        for layeru in self.up_path:
            featu.append(layeru(xyzd[i], xyzd[i-1], featu[-1], featd[i-1]))
            i -= 1
        if(-i > len(idxd)):
            return featu[-1], xyz, None
        else:
            return featu[-1], xyzd[i], idxd[i]
        
class EV_encoder2(nn.Module):
    def __init__(self, in_feat, nfeat_d=[], nfeat_u=[], upsample=False):
        super(EV_encoder2, self).__init__()
        self.upsample = upsample
        self.act = nn.LeakyReLU(0.1, inplace=True)
        
        self.input_conv1 = nn.Sequential(layers.Conv2d(in_feat, 32, kernel_size=5, padding=2),
                                           layers.Conv2d(32, nfeat_d[0], kernel_size=3, padding=1))
        self.input_conv2 = layers.Conv2d(in_feat, nfeat_d[0], kernel_size=3, padding=1)

        self.down_path = nn.ModuleList()
        last_feat = nfeat_d[0]
        for i in range(len(nfeat_d)):
            self.down_path.append(layers.DownSampleBlock(last_feat, nfeat_d[i], 2))
            last_feat = nfeat_d[i]
            
        if(upsample):
            self.up_path = nn.ModuleList()    
            for j in range(len(nfeat_u)-1):
                self.up_path.append(layers.UpSampleBlock(last_feat, nfeat_d[-2-j], nfeat_u[j]))
                last_feat = nfeat_u[j]
            if(len(nfeat_u) == len(nfeat_d)):
                self.up_path.append(layers.UpSampleBlock(last_feat, nfeat_d[0], nfeat_u[-1]))
            else:
                self.up_path.append(layers.UpSampleBlock(last_feat, nfeat_d[-1-len(nfeat_u)], nfeat_u[-1]))
            last_feat = nfeat_u[-1]

        self.out_conv = nn.Conv2d(last_feat, last_feat, kernel_size=1)
    def forward(self, feat):
        init_feat1 = self.input_conv1(feat)
        init_feat2 = self.input_conv2(feat)
        init_feat = self.act(init_feat1 + init_feat2)
        feat_d = [init_feat]
        for layer_d in self.down_path:
            feat_d.append(layer_d(feat_d[-1]))

        if(not self.upsample):
            new_feat = feat_d[-1]
            new_feat = self.out_conv(new_feat)
            return new_feat
               
        feat_u = [feat_d[-1]]
        i = -2
        for layer_u in self. up_path:
            feat_u.append(layer_u(feat_u[-1], feat_d[i]))
            i -= 1
        new_feat = feat_u[-1]
        new_feat = self.out_conv(new_feat)
        
        return new_feat
    
class PE_Flow3d_dense_heavy2_align(nn.Module):
    def __init__(self, npoint=8192, nbin=15, size=[480, 640]):
        super(PE_Flow3d_dense_heavy2_align, self).__init__()
        self.npoint = npoint
        self.nbin = nbin
        self.size = size
        
        self.point_feat_enc = PC_encoder([npoint//4, npoint//16, npoint//64], [0.5, 1.0, 2.0, 2.4, 1.2], [16, 16, 16, 8, 8], [32, 64, 128], [128, 128], True) #1/4
        self.event_feat_enc = EV_encoder2(nbin, [64, 96, 128], [], False) #1/8

        self.point_corr = layers.FlowEmbedding(128, 128, 32, [128, 256])
        self.event_corr = layers.event_corr_block(128, [4, 4], 3, conv=False)
        self.pc_enc_fuse = layers.AttenFuse(128, 128, detach=True)
        self.ev_enc_fuse = layers.AttenFuse2d(128, 128, detach=True)
        self.pc_hid_fuse = layers.AttenFuse(128, 128, detach=True)
        self.ev_hid_fuse = layers.AttenFuse2d(128, 128, detach=True)

        self.init_pc = layers.SetConv(1.0, npoint//4, 16, 128, [128, 128], last_act=False)
        self.init_ev = nn.Sequential(layers.Conv2d(128, 128, kernel_size=3, padding=1),
                                        nn.Conv2d(128, 128, kernel_size=3, padding=1))
        self.iter_block_pc = layers.Update3d(1.0, 4, npoint//4, 256, 128)
        self.iter_block_ev = layers.Update2d(243, 128)

        self.pc_up = layers.SetFlowUpsampling(8, 128)
        self.ev_up = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), 
                                         nn.LeakyReLU(0.1, inplace=True),
                                         nn.Conv2d(128, 8*8*9, kernel_size=1))
        
        self.motion_head_pc = nn.Sequential(nn.Conv1d(128, 64, kernel_size=1), 
                                         nn.LeakyReLU(0.1, inplace=True),
                                         nn.Conv1d(64, 3, kernel_size=1))
        self.motion_head_ev = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), 
                                         nn.LeakyReLU(0.1, inplace=True),
                                         nn.Conv2d(64, 2, kernel_size=1))
        self.conf_head_pc = nn.Sequential(nn.Conv1d(128+128, 64, kernel_size=1), 
                                         nn.LeakyReLU(0.1, inplace=True),
                                         nn.Conv1d(64, 3, kernel_size=1))
        self.conf_head_ev = nn.Sequential(nn.Conv2d(128+128, 64, kernel_size=1), 
                                         nn.LeakyReLU(0.1, inplace=True),
                                         nn.Conv2d(64, 2, kernel_size=1))
        
    def forward(self, xyz1, xyz2, pc_feat1, pc_feat2, ev_feat1, ev_feat2, trans_mar, niter=8):
        xyz1 = xyz1.permute(0, 2, 1).contiguous()
        xyz2 = xyz2.permute(0, 2, 1).contiguous()
        pc_feat1 = pc_feat1.permute(0, 2, 1).contiguous()
        pc_feat2 = pc_feat2.permute(0, 2, 1).contiguous()
        inverse_trans_mar = torch.linalg.inv(trans_mar) 
        B, _, N1 = xyz1.size()
        _, _, N2 = xyz2.size()
        factor_x = float((self.size[1]//8 - 1) / (self.size[1] - 1))
        factor_y = float((self.size[0]//8 - 1) / (self.size[0] - 1))
        down_factors = torch.FloatTensor([factor_x, factor_y]).view(1, 2, 1)
        down_factors = down_factors.to(xyz1.device)

        enc_pc_feat1, fps_xyz1, fps_idx1 = self.point_feat_enc(xyz1, pc_feat1)
        enc_pc_feat2, fps_xyz2, fps_idx2 = self.point_feat_enc(xyz2, pc_feat2)
        enc_ev_feat1 = self.event_feat_enc(ev_feat1)
        enc_ev_feat2 = self.event_feat_enc(ev_feat2)
        
        xy1 = utils.project_to_2d(xyz1, inverse_trans_mar)
        fps_xy1 = pointnet2_utils.gather_operation(xy1.contiguous(), fps_idx1)
        xyd2 = utils.project_to_2d(xyz2, inverse_trans_mar, disp=True)
        xy2 = xyd2[:, :2, :].contiguous()
        d2 = xyd2[:, 2:, :].contiguous()
        fps_xy2 = pointnet2_utils.gather_operation(xy2, fps_idx2)     
        down_xy1_grid = utils.img_grid([B, self.size[0]//8, self.size[1]//8]).to(ev_feat1.device)
        
        # enoder fuse
        pc_feat1_2d = utils.BallQueryInterpolate(1.0, 8, [self.size[0]//8, self.size[1]//8], down_factors*fps_xy1, enc_pc_feat1)
        pc_feat2_2d = utils.BallQueryInterpolate(1.0, 8, [self.size[0]//8, self.size[1]//8], down_factors*fps_xy2, enc_pc_feat2)
        ev_feat1_3d = utils.BilinearInterpolate(fps_xy1*down_factors, enc_ev_feat1)
        ev_feat2_3d = utils.BilinearInterpolate(fps_xy2*down_factors, enc_ev_feat2)
        new_pc_feat1 = self.pc_enc_fuse(enc_pc_feat1, ev_feat1_3d)
        new_pc_feat2 = self.pc_enc_fuse(enc_pc_feat2, ev_feat2_3d)
        new_ev_feat1 = self.ev_enc_fuse(enc_ev_feat1, pc_feat1_2d)
        new_ev_feat2 = self.ev_enc_fuse(enc_ev_feat2, pc_feat2_2d)
        
        self.event_corr.init_corr(new_ev_feat1, new_ev_feat2)
        init_fused_flow_3d = torch.zeros_like(fps_xyz1).to(fps_xyz1.device)
        init_fused_flow_2d = torch.zeros_like(down_xy1_grid).to(down_xy1_grid.device)
        fused_flow_3d = [init_fused_flow_3d]
        fused_flow_2d = [init_fused_flow_2d]
        full_fused_flow_3d = []
        full_fused_flow_2d = []
        hid_feat_pc = hid_feat_pc, _, _ = self.init_pc(fps_xyz1, new_pc_feat1)
        hid_feat_pc = torch.tanh(hid_feat_pc)
        hid_feat_ev = self.init_ev(new_ev_feat2)
        hid_feat_ev = torch.tanh(hid_feat_ev)
        for i in range(niter):
            # warp
            fps_warped_xyz1 = fps_xyz1 + fused_flow_3d[-1]
            fps_warped_xyz1 = fps_warped_xyz1.detach()
            pre_flow_3d = fps_warped_xyz1 - fps_xyz1
            down_warped_xy1_grid = down_xy1_grid + fused_flow_2d[-1]
            down_warped_xy1_grid = down_warped_xy1_grid.detach()
            pre_flow_2d = down_warped_xy1_grid - down_xy1_grid
            
            # corr
            corr_feat_pc = self.point_corr(fps_warped_xyz1, fps_xyz2, new_pc_feat1, new_pc_feat2) #[B, C ,N//4]
            corr_feat_ev = self.event_corr(down_warped_xy1_grid) #[B, C, H//8, W//8]

            # GRUs
            hid_feat_pc = self.iter_block_pc(fps_warped_xyz1, corr_feat_pc, pre_flow_3d, hid_feat_pc)
            hid_feat_ev = self.iter_block_ev(corr_feat_ev, pre_flow_2d, hid_feat_ev)
            
            # decoder fuse
            hid_feat_pc_2d = utils.BallQueryInterpolate(1.0, 8, [self.size[0]//8, self.size[1]//8], down_factors*fps_xy1, hid_feat_pc)
            hid_feat_ev_3d = utils.BilinearInterpolate(down_factors*fps_xy1, hid_feat_ev)
            fused_hid_pc = self.pc_hid_fuse(hid_feat_pc, hid_feat_ev_3d)
            fused_hid_ev = self.ev_hid_fuse(hid_feat_ev, hid_feat_pc_2d)
            
            # flows            
            res_flow_pc = self.motion_head_pc(fused_hid_pc)
            new_flow_pc = pre_flow_3d + res_flow_pc
            
            res_flow_ev = self.motion_head_ev(fused_hid_ev)
            new_flow_ev = pre_flow_2d + res_flow_ev
            

            # project flows
            warped_fps_xy1_ev = fps_xy1 + 8.0 * utils.BilinearInterpolate(down_factors*fps_xy1, new_flow_ev)
            ex_warped_fps_xy1_ev = torch.cat([warped_fps_xy1_ev, torch.ones([B, 1, N1//4], device=warped_fps_xy1_ev.device)], dim=1)
            ex_xy2 = torch.cat([xy2, torch.ones([B, 1, N2], device=xy2.device)], dim=1)
            _, nn_idx = pointnet2_utils.knn(1, ex_warped_fps_xy1_ev.permute(0, 2, 1).contiguous(), ex_xy2.permute(0, 2, 1).contiguous())
            nn_d2 = pointnet2_utils.grouping_operation(d2, nn_idx.int()) #[B, 1, N, 1]
            nn_disp = nn_d2.squeeze(3) #[B, 1, N]
            warped_fps_xyz1_ev = utils.project_to_3d(torch.cat([warped_fps_xy1_ev, nn_disp], dim=1), trans_mar)
            new_flow_ev_3d = warped_fps_xyz1_ev - fps_xyz1
            res_flow_ev_3d = new_flow_ev_3d - pre_flow_3d
            
            warped_fps_xyz1_pc = fps_xyz1 + new_flow_pc
            new_flow_pc_2d = utils.project_to_2d(warped_fps_xyz1_pc, inverse_trans_mar) - fps_xy1 #[B, 2, N]
            new_flow_pc_2d = utils.BallQueryInterpolate(1.0, 8, [self.size[0]//8, self.size[1]//8], down_factors*fps_xy1, 0.125*new_flow_pc_2d)
            mask_flow_pc_2d = (torch.sum(new_flow_pc_2d != 0, dim=1, keepdim=True) != 0).detach()
            res_flow_pc_2d = (new_flow_pc_2d - pre_flow_2d) * mask_flow_pc_2d

            # fused flows
            fused_hid_ev_3d = utils.BilinearInterpolate(down_factors*fps_xy1, fused_hid_ev)
            conf_feat_pc = torch.cat([fused_hid_pc, fused_hid_ev_3d.detach()], dim=1)
            confidence_pc = self.conf_head_pc(conf_feat_pc)
            confidence_pc = torch.sigmoid(confidence_pc)
            res_fused_flow_3d = confidence_pc * res_flow_ev_3d.detach() + (1-confidence_pc) * res_flow_pc
            fused_flow_3d.append(pre_flow_3d + res_fused_flow_3d)
            
            fused_hid_pc_2d = utils.BallQueryInterpolate(1.0, 8, [self.size[0]//8, self.size[1]//8], down_factors*fps_xy1, fused_hid_pc)
            conf_feat_ev = torch.cat([fused_hid_ev, fused_hid_pc_2d.detach()], dim=1)
            confidence_ev = self.conf_head_ev(conf_feat_ev)
            confidence_ev = torch.sigmoid(confidence_ev)
            confidence_ev = confidence_ev * mask_flow_pc_2d
            res_fused_flow_2d = confidence_ev * res_flow_pc_2d.detach() + (1-confidence_ev) * res_flow_ev
            fused_flow_2d.append(pre_flow_2d + res_fused_flow_2d)
            
            # upsample flows
            up_fused_flow_3d = self.pc_up(fps_xyz1, xyz1, fused_hid_pc, fused_flow_3d[-1])
            full_fused_flow_3d.append(up_fused_flow_3d)
            up_mask_2d = self.ev_up(fused_hid_ev) * 0.25 #scale to balance gradient
            up_fused_flow_2d = utils.upsample_flow(fused_flow_2d[-1], up_mask_2d)
            #up_fused_flow_2d = 8.0 * F.interpolate(new_fused_flow_2d, size=self.size, mode='bilinear', align_corners=True)
            full_fused_flow_2d.append(up_fused_flow_2d)
        
        return full_fused_flow_3d, full_fused_flow_2d

    
## loss ##

def sequence_loss(pred_flows, flow, alpha=[], fps_idx=None):
    if(fps_idx is not None):
        flow = pointnet2_utils.gather_operation(flow.contiguous(), fps_idx.contiguous()) #[B, 3, N]

    total_loss = torch.zeros(1).to(flow.device)
    for i in range(len(pred_flows)):
        pred = pred_flows[i]

        cur_loss = (flow - pred).abs()
        cur_loss = cur_loss.mean()

        total_loss += alpha[i] * cur_loss
        #if(i == len(pred_flows)-1):
        #    error = (torch.norm(flow - pred, dim=1)).mean(dim=1)
        #    error = error.mean()
        #    print('pc:', error)

    return total_loss

def sequence_loss2d(pred_flows, flow, mask, alpha=[]):
    B = flow.size()[0]
    valid_num = mask.sum()

    total_loss = torch.zeros(1).to(flow.device)
    for i in range(len(pred_flows)):
        pred = pred_flows[i]

        cur_loss = (flow - pred).abs() #[B, 2, H, W]
        cur_loss = cur_loss.permute(0, 2, 3, 1) #[B, H, W, 2]
        cur_loss = cur_loss[mask]
        cur_loss = cur_loss.mean()
        
        total_loss += alpha[i] * cur_loss
        #if(i == len(pred_flows)-1):
        #    error = torch.norm(flow - pred, dim=1)
        #    error = error[mask]
        #    error = error.sum() / valid_num
        #    print('ev:', error)

    return total_loss