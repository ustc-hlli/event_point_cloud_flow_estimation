import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pointnet2 import pointnet2_utils
import utils

USE_GN = True

class Conv1d(nn.Module):
    def __init__(self, in_feat, out_feat, kernel_size=1, stride=1, padding=0, groups=1, bias=True, use_gn=USE_GN, use_act=True):
        super(Conv1d, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        
        if(use_act):
            self.composed_module = nn.Sequential(
            			nn.Conv1d(in_feat, out_feat, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias),
                        nn.GroupNorm(out_feat//16, out_feat) if(use_gn) else nn.BatchNorm1d(out_feat),
            			nn.LeakyReLU(0.1, inplace=True)
                        )
        else:
            self.composed_module = nn.Conv1d(in_feat, out_feat, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias)

    def forward(self, feat):
        feat = self.composed_module(feat)
        return feat
    
class Conv2d(nn.Module):
    def __init__(self, in_feat, out_feat, kernel_size=1, stride=1, padding=0, groups=1, bias=True, use_gn=USE_GN, use_act=True):
        super(Conv2d, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        
        if(use_act):
            self.composed_module = nn.Sequential(
            			nn.Conv2d(in_feat, out_feat, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias),
                        nn.GroupNorm(out_feat//16, out_feat) if(use_gn) else nn.BatchNorm2d(out_feat),
            			nn.LeakyReLU(0.1, inplace=True)
                        )
        else:
            self.composed_module = nn.Conv2d(in_feat, out_feat, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias)

    def forward(self, feat):
        feat = self.composed_module(feat)
        return feat

class Conv3d(nn.Module):
    def __init__(self, in_feat, out_feat, kernel_size=1, stride=1, padding=0, groups=1, use_gn=USE_GN, use_act=True):
        super(Conv3d, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        
        if(use_act):
            self.composed_module = nn.Sequential(
            			nn.Conv3d(in_feat, out_feat, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups),
                        nn.GroupNorm(out_feat//16, out_feat) if(use_gn) else nn.BatchNorm3d(out_feat),
            			nn.LeakyReLU(0.1, inplace=True)
                        )
        else:
            self.composed_module = nn.Conv3d(in_feat, out_feat, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)

    def forward(self, feat):
        feat = self.composed_module(feat)
        return feat
    
class SetConv(nn.Module):
    def __init__(self, radius, npoint, nsample, in_feat, mlp=[], last_act=True):
        super(SetConv, self).__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.radius = radius
        self.knn = radius == 0
        
        self.mlp = nn.ModuleList()
        last_feat = in_feat + 3
        for i in range(len(mlp)):
            if(i < len(mlp)-1 or last_act):
                self.mlp.append(Conv2d(last_feat, mlp[i]))
            else:
                self.mlp.append(Conv2d(last_feat, mlp[i], use_act=False))
            last_feat = mlp[i]
            
    def forward(self, xyz, feat):
        
        B, C, N = feat.size()
        
        xyz_t = xyz.permute(0 ,2, 1).contiguous()
        feat_t = feat.permute(0 ,2, 1).contiguous()
        
        if(N > self.npoint):
            fps_idx = pointnet2_utils.furthest_point_sample(xyz_t, self.npoint) #[B, N']
            new_xyz = pointnet2_utils.gather_operation(xyz, fps_idx) #[B, 3, N']
        else:
            fps_idx = None
            new_xyz = xyz
            
        if(self.knn):
            _, group_idx = pointnet2_utils.knn(self.nsample, new_xyz.permute(0 ,2, 1).contiguous(), xyz_t) #[B, N', S]
        else:
            group_idx = pointnet2_utils.ball_query(self.radius, self.nsample, xyz_t, new_xyz.permute(0 ,2, 1).contiguous())
           
        group_feat = pointnet2_utils.grouping_operation(feat, group_idx) #[B, C, N', S]
        group_xyz = pointnet2_utils.grouping_operation(xyz, group_idx) - new_xyz.unsqueeze(3)
        group_feat = torch.cat([group_feat, group_xyz], dim=1)
        
        for layer in self.mlp:
            group_feat = layer(group_feat)
            
        new_feat = torch.max(group_feat, dim=3)[0]
        
        return new_feat, new_xyz, fps_idx
    
class SetUpConv(nn.Module):
    def __init__(self, radius, nsample, in_feat1, in_feat2, mlp1=[], mlp2=[], last_act=True):
        super(SetUpConv, self).__init__()
        self.radius = radius
        self.nsample = nsample
        self.knn = radius == 0
        self.mlp1 = nn.ModuleList()
        self.mlp2 = nn.ModuleList()

        last_feat = in_feat1 + 3
        for i in range(len(mlp1)):
            self.mlp1.append(Conv2d(last_feat, mlp1[i]))
            last_feat = mlp1[i]
        last_feat =last_feat + in_feat2
        for j in range(len(mlp2)):
            if(j < len(mlp2)-1 or last_act):
                self.mlp2.append(Conv1d(last_feat, mlp2[j]))
            else:
                self.mlp2.append(Conv1d(last_feat, mlp2[j], use_act=False))
            last_feat = mlp2[j]

    def forward(self, xyz1, xyz2, feat1, feat2):
        '''
        xyz2 is the queries and xyz1 is the targets
        '''
        xyz1_t = xyz1.permute(0, 2, 1).contiguous()
        xyz2_t = xyz2.permute(0, 2, 1).contiguous()

        if(self.knn):
            _, idx = pointnet2_utils.knn(self.nsample, xyz2_t, xyz1_t)
        else:
            idx = pointnet2_utils.ball_query(self.radius, self.nsample, xyz1_t, xyz2_t)

        group_feat = pointnet2_utils.grouping_operation(feat1, idx) #[B, C1, N2, S]
        group_xyz = pointnet2_utils.grouping_operation(xyz1, idx) - xyz2.unsqueeze(3) #[B, 3, N2, S]
        new_feat = torch.cat([group_feat, group_xyz], dim=1)

        for layer in self.mlp1:
            new_feat = layer(new_feat)
        new_feat = torch.max(new_feat, dim=3)[0] #[B, C', N2]

        if(feat2 is not None):
            new_feat = torch.cat([feat2, new_feat], dim=1)
        for layer in self.mlp2:
            new_feat = layer(new_feat)
        
        return new_feat
    
class SetPropagation(nn.Module):
    def __init__(self, nsample, in_feat1, in_feat2, mlp=[], last_act=True):
        super(SetPropagation, self).__init__()
        self.nsample = nsample

        self.mlp = nn.ModuleList()
        last_feat = in_feat1 + in_feat2
        for i in range(len(mlp)):
            if(i < len(mlp)-1 or last_act):
                self.mlp.append(Conv1d(last_feat, mlp[i]))
            else:
                self.mlp.append(Conv1d(last_feat, mlp[i], use_act=False))
            last_feat = mlp[i]


    def forward(self, xyz1, xyz2, feat1, feat2):
        '''
        xyz2 is the queries and xyz1 is the targets
        '''
        
        xyz1_t = xyz1.permute(0, 2, 1).contiguous()
        xyz2_t = xyz2.permute(0, 2, 1).contiguous()

        dists, idx = pointnet2_utils.knn(self.nsample, xyz2_t, xyz1_t) #[B, N2, S]
        group_feat1 = pointnet2_utils.grouping_operation(feat1, idx) #[B, C1, N2, S]
        weights = 1.0/(dists + 1e-8)
        #weights = torch.ones_like(dists, dtype=torch.float32, device=xyz1.device)
        weights = weights / (weights.sum(dim=2, keepdim=True))

        interp_feat1 = (group_feat1 * weights.unsqueeze(1)).sum(dim=3) #[B, C1, N2]
        if(feat2 is not None):
            new_feat = torch.cat([feat2, interp_feat1], dim=1)
        else:
            new_feat = interp_feat1

        for layer in self.mlp:
            new_feat = layer(new_feat)

        return new_feat

class SetFlowUpsampling(nn.Module):
    def __init__(self, nsample, in_feat):
        super(SetFlowUpsampling, self).__init__()
        self.nsample = nsample
        
        self.weight_net = nn.Sequential(nn.Conv2d(in_feat + 3, in_feat, kernel_size =1),
                                        nn.LeakyReLU(0.1, inplace=True),
                                        nn.Conv2d(in_feat, 3, kernel_size=1))

        
    def forward(self, xyz1, xyz2, feat1, flow):
        '''
        xyz2 is the queries and xyz1 is the targets
        '''
        B, _, N1 = xyz1.size()
        _, _, N2 = xyz2.size()
        if(flow.size()[1] == 2): #2d flow
            ex_xyz1 = torch.cat([xyz1, torch.ones([B, 1, N1]).to(xyz1.device)], dim=1)
            ex_xyz2 = torch.cat([xyz2, torch.ones([B, 1, N2]).to(xyz2.device)], dim=1)
        else:
            ex_xyz1 = xyz1
            ex_xyz2 = xyz2

        ex_xyz1_t = ex_xyz1.permute(0, 2, 1).contiguous()
        ex_xyz2_t = ex_xyz2.permute(0, 2, 1).contiguous() 
        
        _, idx = pointnet2_utils.knn(self.nsample, ex_xyz2_t, ex_xyz1_t)
        group_flow = pointnet2_utils.grouping_operation(flow, idx) #[B, 3, N2, S]
        group_xyz = pointnet2_utils.grouping_operation(xyz1, idx) - xyz2.unsqueeze(3)
        group_feat = pointnet2_utils.grouping_operation(feat1, idx)
        
        weights = self.weight_net(torch.cat([group_feat, group_xyz], dim=1)) #[B, 3, N2, S]
        weights = torch.softmax(weights, dim=3)

        up_flow = torch.sum(group_flow * weights, dim=3) #[B, 3, N2]
        
        return up_flow
    
class FlowEmbedding(nn.Module):
    def __init__(self, in_feat1, in_feat2, k, mlp=[]):
        super(FlowEmbedding, self).__init__()
        self.in_feat1 = in_feat1
        self.in_feat2 = in_feat2
        self.k = k

        self.mlp = nn.ModuleList()
        last_feat = in_feat1 + in_feat2 + 3
        for out_feat in mlp:
            self.mlp.append(Conv2d(last_feat, out_feat))
            last_feat = out_feat
    
    def forward(self, xyz1, xyz2, feat1, feat2):
        B, C, N = feat1.size()

        dists = torch.norm(xyz1.unsqueeze(3) - xyz2.unsqueeze(2), dim=1)
        k_dist, k_idx = torch.topk(dists, self.k, dim=2, largest=False)

        group_xyz = pointnet2_utils.grouping_operation(xyz2, k_idx.int()) - xyz1.unsqueeze(3)
        group_feat = pointnet2_utils.grouping_operation(feat2, k_idx.int())

        new_feat = torch.cat([feat1.unsqueeze(3).repeat(1, 1, 1, self.k), group_feat, group_xyz], dim=1)
        for layer in self.mlp:
            new_feat = layer(new_feat)

        new_feat = torch.max(new_feat, dim=3)[0]

        return new_feat
    
class AttenFuse(nn.Module):
    def __init__(self, in_feat_1, in_feat_2, detach=True):
        super(AttenFuse, self).__init__()
        self.detach = detach
        self.pre_conv = Conv1d(in_feat_2, in_feat_1)
        self.att_conv = nn.Sequential(Conv1d(2*in_feat_1, in_feat_1),
                                      nn.Conv1d(in_feat_1, 2*in_feat_1, kernel_size=1))
        
    def forward(self, feat1, feat2):
        B, C, N = feat1.size()
        if(self.detach):
            feat2 = feat2.detach()
        feat2 = self.pre_conv(feat2)
        mask = (feat2 != 0)
        mask = torch.sum(mask, dim=1, keepdim=True) #[B, 1, N]
        mask = (mask != 0).detach()

        att = self.att_conv(torch.cat([feat1, feat2], dim=1)) #[B, 2C, N]
        att = torch.softmax(att.view(B, 2, C, N), dim=1)
        masked_att = torch.stack([att[:, 0, :, :], att[:, 1, :, :]*mask], dim=1) #[B, 2, C, N]
        masked_att = masked_att / (masked_att.sum(dim=1, keepdim=True) + 1e-6) 

        fused_feat = masked_att[:, 0, :, :] * feat1 + masked_att[:, 1, :, :] * feat2 

        return fused_feat
    
class AttenFuse2d(nn.Module):
    def __init__(self, in_feat_1, in_feat_2, detach=True):
        super(AttenFuse2d, self).__init__()
        self.detach = detach
        self.pre_conv = Conv2d(in_feat_2, in_feat_1, kernel_size=3, padding=1)
        self.att_conv = nn.Sequential(Conv2d(2*in_feat_1, in_feat_1, kernel_size=3, padding=1),
                                      nn.Conv2d(in_feat_1, 2*in_feat_1, kernel_size=1))
        
    def forward(self, feat1, feat2):
        B, C, H, W = feat1.size()
        if(self.detach):
            feat2 = feat2.detach()
        feat2 = self.pre_conv(feat2)
        mask = (feat2 != 0)
        mask = torch.sum(mask, dim=1, keepdim=True) #[B, 1, H, W]
        mask = (mask != 0).detach()

        att = self.att_conv(torch.cat([feat1, feat2], dim=1)) #[B, 2C, H, W]
        att = torch.softmax(att.view(B, 2, C, H, W), dim=1)
        masked_att = torch.stack([att[:, 0, :, :, :], att[:, 1, :, :, :]*mask], dim=1) #[B, 2, C, H, W]
        masked_att = masked_att / (masked_att.sum(dim=1, keepdim=True) + 1e-6) 

        fused_feat = masked_att[:, 0, :, :, :] * feat1 + masked_att[:, 1, :, :, :] * feat2 

        return fused_feat
    
class MotionEncoder2d(nn.Module):
    def __init__(self, corr_feat, out_feat):
        super(MotionEncoder2d, self).__init__()
        self.convc1 = nn.Conv2d(corr_feat, 192, 1)
        self.convc2 = nn.Conv2d(192, 128, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64+128, out_feat-2, 3, padding=1)

    def forward(self, corr, flow):
        cor = F.leaky_relu(self.convc1(corr), negative_slope=0.1, inplace=True)
        cor = F.leaky_relu(self.convc2(cor), negative_slope=0.1, inplace=True)
        flo = F.leaky_relu(self.convf1(flow), negative_slope=0.1, inplace=True)
        flo = F.leaky_relu(self.convf2(flo), negative_slope=0.1, inplace=True)

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.leaky_relu(self.conv(cor_flo), negative_slope=0.1, inplace=True)
        return torch.cat([out, flow], dim=1)
    
class MotionEncoder3d(nn.Module):
    def __init__(self, npoint, corr_feat, out_feat):
        super(MotionEncoder3d, self).__init__()
        self.convc = SetConv(0, npoint, 4, corr_feat, mlp=[128], last_act=False)
        self.convf = SetConv(0, npoint, 16, 3, mlp=[64], last_act=False)
        self.conv = nn.Conv1d(64+128, out_feat-3, kernel_size=1)

    def forward(self, xyz, corr, flow):
        cor = F.leaky_relu(self.convc(xyz, corr)[0], negative_slope=0.1, inplace=True)
        flo = F.leaky_relu(self.convf(xyz, flow)[0], negative_slope=0.1, inplace=True)

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.leaky_relu(self.conv(cor_flo), negative_slope=0.1, inplace=True)
        return torch.cat([out, flow], dim=1)

class SetGRU(nn.Module):
    def __init__(self, radius, nsample, in_point, in_feat, hid_feat):
        super(SetGRU, self).__init__()
        self.radius = radius
        self.nsamle = nsample
        self.in_point = in_point
        
        self.convz = SetConv(radius, in_point, nsample, hid_feat + in_feat, [hid_feat], last_act=False)
        self.convr = SetConv(radius, in_point, nsample, hid_feat + in_feat, [hid_feat], last_act=False)
        self.convq = SetConv(radius, in_point, nsample, hid_feat + in_feat, [hid_feat], last_act=False)
            
    def forward(self, xyz, feat, hid_state):
        hid_feat = torch.cat([feat, hid_state], dim=1)
        
        z = torch.sigmoid(self.convz(xyz, hid_feat)[0])
        r = torch.sigmoid(self.convr(xyz, hid_feat)[0])
        q = torch.tanh(self.convq(xyz,  torch.cat([feat, r * hid_state], dim=1))[0])
        
        next_hid_state = (1 - z) * hid_state + z * q
        
        return next_hid_state
    
class SepConvGRU(nn.Module):
    def __init__(self, in_feat, hid_feat):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hid_feat + in_feat, hid_feat, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hid_feat + in_feat, hid_feat, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hid_feat + in_feat, hid_feat, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hid_feat + in_feat, hid_feat, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hid_feat + in_feat, hid_feat, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hid_feat + in_feat, hid_feat, (5,1), padding=(2,0))


    def forward(self, x, h):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))        
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))       
        h = (1-z) * h + z * q

        return h    
class Update2d(nn.Module):
    def __init__(self, corr_feat, hid_feat):
        super(Update2d, self).__init__()
        self.motion_encoder = MotionEncoder2d(corr_feat, hid_feat)
        self.gru = SepConvGRU(hid_feat, hid_feat)
        
    def forward(self, corr, flow, hid_state):
        in_feat = self.motion_encoder(corr, flow)
        next_hid_state = self.gru(in_feat, hid_state)
        
        return next_hid_state
    
class Update3d(nn.Module):
    def __init__(self, radius, nsample, npoint, corr_feat, hid_feat):
        super(Update3d, self).__init__()
        self.motion_encoder = MotionEncoder3d(npoint, corr_feat, hid_feat)
        self.gru = SetGRU(1.0, nsample, npoint, hid_feat, hid_feat)
        
    def forward(self, xyz, corr, flow, hid_state):
        in_feat = self.motion_encoder(xyz, corr, flow)
        next_hid_state = self.gru(xyz, in_feat, hid_state)
        
        return next_hid_state
    
class DownSampleBlock(nn.Module):
    def __init__(self, in_feat, out_feat, nlevel):
        super(DownSampleBlock, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.nlevel = nlevel
        self.conv = nn.ModuleList()
        
        self.conv.append(Conv2d(in_feat, out_feat, kernel_size=3, padding=1, stride=2))
        for i in range(1, nlevel):
            self.conv.append(Conv2d(out_feat, out_feat, kernel_size=3, padding=1))            
        self.conv2 = Conv2d(in_feat, out_feat, kernel_size=3, padding=1, stride=2)    
        self.act = nn.LeakyReLU(0.1, inplace=True)
        
    def forward(self, feat):
        new_feat = feat
        for layer in self.conv:
            new_feat = layer(new_feat)
        skip_feat = self.conv2(feat)
        new_feat = self.act(new_feat + skip_feat)
        
        return new_feat
    
class UpSampleBlock(nn.Module):
    def __init__(self, in_feat1, in_feat2, out_feat, use_gn=USE_GN):
        super(UpSampleBlock, self).__init__()
        self.in_feat1 = in_feat1
        self.in_feat2 = in_feat2
        self.out_feat = out_feat
        
        self.tconv = nn.Sequential(nn.ConvTranspose2d(in_feat1, in_feat1, kernel_size=2, stride=2),
                                   nn.GroupNorm(in_feat1//16, in_feat1) if(use_gn) else nn.BatchNorm2d(in_feat1),
                                   nn.LeakyReLU(0.1, inplace=True))
        self.conv = nn.Sequential(Conv2d(in_feat1 + in_feat2, out_feat, kernel_size=3, padding=1),
                                  Conv2d(out_feat, out_feat, kernel_size=3, padding=1))
                                  
        
    def forward(self, feat1, feat2):
        up_feat1 = self.tconv(feat1)
        new_feat = torch.cat([up_feat1, feat2], dim=1)
        new_feat = self.conv(new_feat)
        
        return new_feat
    
class event_corr_block(nn.Module):
    def __init__(self, out_feat, radius=[4, 4], nlevel=3, conv=True):
        super(event_corr_block, self).__init__()
        self.radius = radius
        self.nlevel = nlevel
        self.conv = conv
        
        if(conv):
            self.out_conv = Conv2d(self.nlevel*(2*self.radius[0]+1)*(2*self.radius[1]+1), out_feat, kernel_size=1)

    def forward(self, target_xy):
        '''
        xy: [B, 2, H, W]
        '''
        B, _, H1, W1 = target_xy.size()
        target_xy = target_xy.permute(0, 2, 3, 1).contiguous().view(-1, 1, 1, 2) #[BH1W1, 1, 1, 2]

        deltay = torch.arange(-self.radius[0], self.radius[0] + 1).view(1, -1, 1).repeat(1, 1, 2 * self.radius[1] + 1).float()
        deltax = torch.arange(-self.radius[1], self.radius[1] + 1).view(1, 1, -1).repeat(1, 2 * self.radius[0] + 1, 1).float()
        delta = torch.cat([deltax, deltay], dim=0).to(target_xy.device) #[2, 2Rh+1, 2Rw+1] (x, y)
        delta = delta.permute(1, 2, 0).unsqueeze(0).contiguous() #[1, 2Rh+1, 2Rw+1, 2]

        out_corr = []
        for i in range(len(self.corr_pyramid)):
            cur_corr_field = self.corr_pyramid[i] #[B, H1, W1, 1, H2, W2]
            H2, W2 = cur_corr_field.size()[-2:]
            cur_corr_field = cur_corr_field.view(-1, 1, H2, W2) #[BH1W1, 1, H2, W2]

            cur_xy = target_xy / (2 ** i)
            cur_xy = cur_xy + delta #[BH1W1, 2Rh+1, 2Rw+1, 2]
            norm_cur_x = 2 * cur_xy[:, :, :, 0] / (W2 - 1) -1
            norm_cur_y = 2 * cur_xy[:, :, :, 1] / (H2 - 1) -1
            norm_cur_xy = torch.stack([norm_cur_x, norm_cur_y], dim=3)
            inter_corr = F.grid_sample(cur_corr_field, norm_cur_xy, align_corners=True) #[BH1W1, 1, 2Rh+1, 2Rw+1]
            inter_corr = inter_corr.view(B, H1, W1, -1).permute(0, 3, 1, 2) #[B, (2Rh+1)(2Rw+1), H1, W1]
            out_corr.append(inter_corr)
        out_corr = torch.cat(out_corr, dim=1).contiguous()
        if(self.conv):
            out_corr = self.out_conv(out_corr) #[B, Co, H1, W1]
            
        return out_corr
    
    def init_corr(self, fmap1, fmap2):
        self.feat_size = [fmap1.size()[-2], fmap1.size()[-1]] #(h, w)
        self.corr_pyramid = []
        
        corr_field = self.correlation(fmap1, fmap2)
        for i in range(self.nlevel):
            if(i > 0):
                corr_field = F.avg_pool2d(corr_field, 2, stride=2)
            _, _, H2, W2 = corr_field.size()
            self.corr_pyramid.append(corr_field.view(-1, self.feat_size[0], self.feat_size[1], 1, H2, W2))
            
        return None
        

    @staticmethod
    def correlation(feat1, feat2):
        '''
        feat: [B, C, H, W]
        '''
        B, C, H, W = feat1.size()
        feat1 = feat1.view(B, C, -1)
        feat2 = feat2.view(B, C, -1)

        corr = torch.matmul(feat1.permute(0, 2, 1), feat2) #[B, HW, HW]
        corr = corr / np.sqrt(C)
        corr = corr.view(-1, 1, H, W) #[BHW, 1, H, W]

        return corr