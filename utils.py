import numpy as np
import torch
import torch.nn.functional as F
from scipy import interpolate
import os
import cv2
from cv2 import reprojectImageTo3D, perspectiveTransform
from pointnet2 import pointnet2_utils


def square_dist(src, dst):
     #src : [n, 2]
     #dst : [m, 2]
     n=src.shape[0]
     m=dst.shape[0]
     
     dist = -2 * np.matmul(src, dst.transpose(1, 0)).astype('float16') #[n,m]
     dist += np.sum(src ** 2, -1).view(n, 1)
     dist += np.sum(dst ** 2, -1).view(1, m)
     
     return dist
 
def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img

def img_bilinear_interpolate(target_xy, ori, img, mask_target, mask_img):
    img_y = np.arange(img.shape[0]).reshape(img.shape[0], 1)
    img_y = np.repeat(img_y, img.shape[1], 1)
    img_x = np.arange(img.shape[1]).reshape(1, img.shape[1])
    img_x = np.repeat(img_x, img.shape[0],0)
    img_xy = np.stack([img_x, img_y], axis=2) #[h, w, 2] (x,y)
    
    ori_y = np.arange(ori.shape[0]).reshape(ori.shape[0], 1)
    ori_y = np.repeat(ori_y, ori.shape[1], 1)
    ori_x = np.arange(ori.shape[1]).reshape(1, ori.shape[1])
    ori_x = np.repeat(ori_x, ori.shape[0], 0)
    ori_xy = np.stack([ori_x, ori_y], axis=2) #[h, w, 2] (x,y)
    
    valid_img_xy = img_xy[mask_img] #[v_i, 2] (x,y)
    valid_ori_xy = ori_xy[mask_target] #[v_t, 2] (x,y)
    valid_target_xy = target_xy[mask_target] #[v_t, 2] (x,y)
    valid_ori = ori[mask_target] #[v_t]
    valid_img = img[mask_img] #[v_i]
    vt = valid_target_xy.shape[0]
    vi = valid_img_xy.shape[0]
    
    interpolated_img = np.zeros([vt, 1]) #[v_t, 1]
    for id_point in range(valid_target_xy.shape[0]):
        cur_point = (valid_target_xy[id_point, :])[None, :] #[1,2]
        cur_point_ori = (valid_ori_xy[id_point, :])[None, :] #[1,2]
        mask_neighbor = np.logical_and(np.abs(valid_img_xy[:, 0] - cur_point[:, 0])<1, np.abs(valid_img_xy[:, 1] - cur_point[:, 1])<1)
        neighbors_xy = valid_img_xy[mask_neighbor] #[nei, 2]
        neighbor_img = valid_img[mask_neighbor] #[nei]
        if(neighbor_img.shape[0] == 0):
            interpolated_img[id_point] = 0.0 #invalid points
            continue
        dists = np.linalg.norm(cur_point - neighbors_xy, ord=1, axis=1) #[nei]
        dists = dists + 1e-10
        weights = (1.0/dists)/((1.0/dists).sum(axis=0)) #[nei]
        cur_interpolated_img = (weights * neighbor_img).sum(axis=0)
        interpolated_img[id_point] = cur_interpolated_img
    
    return interpolated_img

def compute_scene_flow(disp1, disp2, disp_mask1, disp_mask2, op_flow, flow_mask, trans_mar=np.array([[1.0, 0.0, 0.0, -335.0999870300293],
                                                                                                     [0.0, 1.0, 0.0, -221.23667526245117],
                                                                                                     [0.0, 0.0, 0.0, 569.7632987676102],
                                                                                                     [0.0, 0.0, 1.6691407297504248, 0.0]])):
    assert disp1.shape == disp_mask1.shape
    assert disp1.shape[0] == op_flow.shape[0]
    assert disp1.shape[1] == op_flow.shape[1]
    assert op_flow.shape[0] == flow_mask.shape[0]
    assert op_flow.shape[1] == flow_mask.shape[1]
    assert disp2.shape == disp_mask2.shape
    
    mask1 = disp_mask1 * flow_mask
    mask2 = disp_mask2
    
    x1 = np.arange(op_flow.shape[1]).reshape(1, op_flow.shape[1])
    x1 = np.repeat(x1, op_flow.shape[0], 0)
    new_x1 = x1 + op_flow[..., 0]
    
    y1 = np.arange(op_flow.shape[0]).reshape(op_flow.shape[0], 1)
    y1 = np.repeat(y1, op_flow.shape[1], 1)
    new_y1 = y1 + op_flow[..., 1]
    
    xy1 = np.stack([x1, y1], axis=2) #[h, w, 2] (x, y)
    new_xy1 = np.stack([new_x1, new_y1], axis=2) #[h, w, 2] (x, y)
    
    inverse_disp2 = np.where(mask2, 1.0/disp2, 0.0)
    inverse_dips1 = np.where(mask1, 1.0/disp1, 0.0)
    inter_inverse_disp1 = img_bilinear_interpolate(new_xy1, inverse_dips1, inverse_disp2, mask1, mask2) #[v_t, 1], [v_t]
    #inter_inverse_disp1 = img_nearest_interpolate(new_xy1, inverse_dips1, inverse_disp2, mask1, mask2) #[v_t, 1], [v_t]
    exten_disp1 = (disp1[mask1])[:, None]
    inter_disp1 = np.where(inter_inverse_disp1 > 0, 1.0/inter_inverse_disp1, 0.0)
    inter_xy1_disp1 = np.concatenate([new_xy1[mask1], inter_disp1], axis=1) #[v_t, 3] (x, y, d)
    xy1_disp1 = np.concatenate([xy1[mask1], exten_disp1], axis=1) #[v_t, 3] (x, y, d)
    
    inter_points1 = perspectiveTransform(inter_xy1_disp1.reshape(-1, 1, 3), trans_mar).reshape(-1, 3) #[v_t, 3], returns zero vector if the disparity is 0.
    inter_points1 = inter_points1.astype('float32')
    
    points1 = perspectiveTransform(xy1_disp1.reshape(-1, 1, 3), trans_mar).reshape(-1, 3) #[v_t, 3]
    points1 = points1.astype('float32')
    
    scene_flow = inter_points1 - points1 #[v_t, 3]
    
    sc_mask1 = np.logical_and(np.abs(scene_flow[:, 2]) <= 3.0, inter_points1[:, 2] > 0)
    valid_scene_flow = scene_flow[sc_mask1]
    valid_points1 = points1[sc_mask1]
    
    sc_mask2 = np.logical_not(np.isnan(valid_scene_flow.sum(axis=1)))
    valid_scene_flow = valid_scene_flow[sc_mask2]
    valid_points1 = valid_points1[sc_mask2]
    
    return valid_points1, valid_scene_flow

def disp_to_pc(disp, mask, trans_mar=np.array([[1.0, 0.0, 0.0, -335.0999870300293],
                                               [0.0, 1.0, 0.0, -221.23667526245117],
                                               [0.0, 0.0, 0.0, 569.7632987676102],
                                               [0.0, 0.0, 1.6691407297504248, 0.0]])):
    
    points = reprojectImageTo3D(disp, trans_mar)
    points = points[mask] #[n,3]
    
    return points

def stream_to_voxel(stream, size=[], nbin=15, normalize=True):
    '''
    from DSEC.
    https://github.com/uzh-rpg/DSEC/tree/main/scripts/dataset
    
    '''
     
    t_norm = (nbin - 1) * (stream['t']-stream['t'].min()) / (stream['t'].max() - stream['t'].min())
    
    with torch.no_grad():
        voxel_grid = torch.zeros([nbin, size[0], size[1]]).float()
        x = torch.FloatTensor(stream['x'])
        y = torch.FloatTensor(stream['y'])
        t = torch.FloatTensor(t_norm)
        p = torch.FloatTensor(stream['p'])
    
        x0 = x.int()
        y0 = y.int()
        t0 = t.int()

        for xlim in [x0,x0+1]:
            for ylim in [y0,y0+1]:
                for tlim in [t0,t0+1]:
                    mask = (xlim < size[1]) & (xlim >= 0) & (ylim < size[0]) & (ylim >= 0) & (tlim >= 0) & (tlim < nbin)
                    interp_values = p * (1 - torch.abs(xlim-x)) * (1 - torch.abs(ylim-y)) * (1 - torch.abs(tlim - t))
                
                    index = (size[0] * size[1] * tlim).long() + \
                                (size[1] * ylim).long() + \
                                xlim.long()
                    voxel_grid.put_(index[mask], interp_values[mask], accumulate=True)
        
        if(normalize):
            nonzero_mask = torch.nonzero(voxel_grid, as_tuple=True)
            if(nonzero_mask[0].size()[0] > 0):
                mean = voxel_grid[nonzero_mask].mean()
                std = voxel_grid[nonzero_mask].std()
        
                voxel_grid[nonzero_mask] = voxel_grid[nonzero_mask] - mean 
                if(std > 0):
                    voxel_grid[nonzero_mask] = voxel_grid[nonzero_mask] / std
           
    return voxel_grid.cpu().numpy()

def compute_3d_metric(pred_flows, flow):
    epe = np.zeros([len(pred_flows)])
    accs = np.zeros([len(pred_flows)])
    out = np.zeros([len(pred_flows)])
    B = flow.size()[0]
    assert B == 1

    mag = torch.norm(flow, dim=1) #[B, N]
    for i in range(len(pred_flows)):
        pred = pred_flows[i]
        cur_err = torch.norm(pred - flow, dim=1) #[B, N]
        
        cur_epe = cur_err.mean()
        
        cur_accs = ((cur_err < 0.05) | ((cur_err / (mag+1e-8)) < 0.05)).float().mean()
        
        cur_out = ((cur_err > 0.3) | ((cur_err / (mag+1e-8)) > 0.1)).float().mean()

        epe[i] = (cur_epe.cpu().data.numpy())
        accs[i] = (cur_accs.cpu().data.numpy())
        out[i] = (cur_out.cpu().data.numpy())

    return epe, accs, out

def compute_2d_metric(pred_flows, mask, flow):
    epe = np.zeros([len(pred_flows)])
    onepx = np.zeros([len(pred_flows)])
    threepx = np.zeros([len(pred_flows)])
    
    B = flow.size()[0]
    assert B == 1

    mag = torch.norm(flow, dim=1)[mask]
    valid_num = mask.sum() 
    for i in range(len(pred_flows)):
        pred = pred_flows[i]
        cur_err = torch.norm(pred - flow, dim=1) #[B, H, W]
        cur_err = cur_err[mask]
        
        cur_epe = cur_err.sum()
        cur_epe = cur_epe / valid_num
        
        cur_onepx = (cur_err > 1.0).float().sum()
        cur_onepx = cur_onepx / valid_num

        cur_threepx = (cur_err > 3.0).float().sum()
        cur_threepx = cur_threepx / valid_num

        epe[i] = (cur_epe.cpu().data.numpy())
        onepx[i] = (cur_onepx.cpu().data.numpy())
        threepx[i] = (cur_threepx.cpu().data.numpy())

    return epe, onepx, threepx

def compute_epe(pred_flows, flow):
    epe = np.zeros([len(pred_flows)])
    B = flow.size()[0]
    assert B == 1
    
    for i in range(len(pred_flows)):
        pred = pred_flows[i]
        cur_epe = torch.norm(pred - flow, dim=1).mean(dim=1)
        cur_epe = cur_epe.mean()

        epe[i] = (cur_epe.cpu().data.numpy())

    return epe

def compute_2depe(pred_flows, mask, flow):
    epe = np.zeros([len(pred_flows)])
    B = flow.size()[0]
    assert B == 1
    
    valid_num = mask.sum() 
    for i in range(len(pred_flows)):
        pred = pred_flows[i]
        cur_epe = torch.norm(pred - flow, dim=1) #[B, H, W]
        cur_epe = (cur_epe[mask]).sum()
        cur_epe = cur_epe / valid_num

        epe[i] = (cur_epe.cpu().data.numpy())

    return epe

def project_to_2d(xyz, inverse_trans_mar, disp=False):

    B, _, N = xyz.size()
    xyz_t = torch.cat([xyz, torch.ones([B, 1, N], device=xyz.device)], dim=1).permute(0, 2, 1).contiguous()
    xyz_t = xyz_t.unsqueeze(3) #[B, N, 4, 1]
    back_projected_xyz = torch.matmul(inverse_trans_mar.unsqueeze(1), xyz_t) #[B, N, 4, 1]
    w = (back_projected_xyz[:,:,3,:]).view(B, N, 1, 1)
    if(disp):
        back_projected_xy = (back_projected_xyz[:, :, :3, :]) / w #[B, N, 3, 1] (x, y, d)
    else:
        back_projected_xy = (back_projected_xyz[:, :, :2, :]) / w #[B, N, 2, 1] (x, y)
    back_projected_xy = back_projected_xy.squeeze(3).permute(0, 2, 1) #[B, 2/3, N]

    return back_projected_xy

def project_to_3d(xyd, trans_mar):

    B, _, N = xyd.size()
    xyd_t = torch.cat([xyd, torch.ones([B, 1, N], device=xyd.device)], dim=1).permute(0, 2, 1).contiguous()
    xyd_t = xyd_t.unsqueeze(3) #[B, N, 4, 1]
    projected_xyd = torch.matmul(trans_mar.unsqueeze(1), xyd_t)
    w = (projected_xyd[:,:,3,:]).view(B, N, 1, 1)
    project_xyz = (projected_xyd[:, :, :3, :]) / w #[B, N, 3, 1] (x, y, z)
    project_xyz = project_xyz.squeeze(3).permute(0, 2, 1) #[B, 3, N]

    return project_xyz

def img_grid(size):
    B = size[0]
    H = size[1]
    W = size[2]
    
    y_grid = torch.arange(0, H).view(1, -1, 1).repeat(1, 1, W)
    x_grid = torch.arange(0, W).view(1, 1, -1).repeat(1, H, 1)
    grid = torch.cat([x_grid, y_grid], dim=0).unsqueeze(0) #[1, 2, H, W], (x, y)
    grid = grid.repeat(B, 1, 1, 1).float().contiguous()
    
    return grid

def BilinearInterpolate(xy, feat):
    '''
    xy : [B, 2, N]
    feat: [B, C, H, W]
    '''
    B, _, H, W = feat.size()
    norm_x = 2 * xy[:, 0, :] / (W-1) - 1
    norm_y = 2 * xy[:, 1, :] / (H-1) - 1
    norm_xy = torch.stack([norm_x, norm_y], dim=2) #[B, N, 2]
    norm_xy = norm_xy.unsqueeze(2) #[B, N, 1, 2]
    
    interpolated = F.grid_sample(feat, norm_xy, align_corners=True) #[B, C, N, 1]
    interpolated = interpolated.squeeze(3)
    
    return interpolated

def BallQueryInterpolate(radius, nsample, size, xy, feat):
    B, _, N = xy.size()
    
    ex_xy = torch.cat([xy, torch.ones([B, 1, N]).to(xy.device)], dim=1).permute(0, 2, 1).contiguous() #[B, N, 3]
    
    imgx = torch.arange(size[1]).view(1, 1, 1, -1).repeat(B, 1, size[0], 1).float() #[B, 1, H, W]
    imgy = torch.arange(size[0]).view(1, 1, -1, 1).repeat(B, 1, 1, size[1]).float()
    ex_imgxy = torch.cat([imgx, imgy, torch.ones_like(imgx)], dim=1).to(xy.device) #[B, 3, H, W]
    ex_imgxy = ex_imgxy.permute(0, 2, 3, 1).contiguous().view(B, -1, 3) #[B, HW, 3]
    
    dist, idx = pointnet2_utils.knn(nsample, ex_imgxy, ex_xy) #[B, HW, K]
    grouped = pointnet2_utils.grouping_operation(feat.contiguous(), idx) #[B, C, HW, K]
    mask = (dist <= radius)
    weights = (1.0 / (dist + 1e-6)) * mask
    weights = weights / (weights.sum(dim=2, keepdim=True) + 1e-6) #[B, HW, K]
    
    interpolated = (grouped * weights.unsqueeze(1)).sum(dim=3) #[B, C, HW]
    interpolated = interpolated.view(B, -1, size[0], size[1]) #[B, C, H, W]
    
    return interpolated

def upsample_flow(flow, mask):
    """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
    B, _, H, W = flow.shape
    mask = mask.view(B, 1, 9, 8, 8, H, W)
    mask = torch.softmax(mask, dim=2)

    up_flow = F.unfold(8 * flow, [3,3], padding=1)
    up_flow = up_flow.view(B, 2, 9, 1, 1, H, W)

    up_flow = torch.sum(mask * up_flow, dim=2) #[B, 2, 8, 8, H, W]
    up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
    return up_flow.reshape(B, 2, 8*H, 8*W)

def rect2dist_torch(xy, trans_mar, cam_mar, rect_mar, k):
    # xy: [B, 2, N]
    B, _, N = xy.size()
    rect_mar = torch.linalg.inv(rect_mar)
    f = trans_mar[:, 2, 3] #[B]
    cx = -trans_mar[:, 0, 3]
    cy = -trans_mar[:, 1, 3]
    dfx = cam_mar[:, 0, 0]
    dfy = cam_mar[:, 1, 1]
    dcx = cam_mar[:, 0, 2]
    dcy = cam_mar[:, 1, 2]
       
    x = (xy[:, 0, :]- cx.view(-1, 1)) / f.view(-1, 1) #[B, N]
    y = (xy[:, 1, :]- cy.view(-1, 1)) / f.view(-1, 1)
    xyf = torch.stack([x, y, torch.ones_like(x).to(x.device)], dim=1) #[B, 3, N]
    xyf_invrect = torch.matmul(rect_mar, xyf) #[B, 3, N]
    xy_invrect = xyf_invrect[:, :2, :] / xyf_invrect[:, 2:, :] #[B, 2, N]
    
    r = torch.norm(xy_invrect, dim=1) #[B, N]
    theta = torch.atan(r)
    thetad = theta * (1 + k[:, 0].view(-1, 1)*(theta**2) +
                      k[:, 1].view(-1, 1)*(theta**4) +
                      k[:, 2].view(-1, 1)*(theta**6) +
                      k[:, 3].view(-1, 1)*(theta**8)) #[B, N]
    x_dist = dfx.view(-1, 1) * ((thetad * xy_invrect[:, 0, :]) / r) + dcx.view(-1, 1) #[B, N]
    y_dist = dfy.view(-1, 1) * ((thetad * xy_invrect[:, 1, :]) / r) + dcy.view(-1, 1)
    xy_dist = torch.stack([x_dist, y_dist], dim=1)
    
    return xy_dist

def dist2rect_torch(xy, trans_mar, cam_mar, rect_mar, k):
    # xy: [B, 2, N]
    B, _, N = xy.size()
    f = trans_mar[:, 2, 3] #[B]
    cx = -trans_mar[:, 0, 3]
    cy = -trans_mar[:, 1, 3]
    dfx = cam_mar[:, 0, 0]
    dfy = cam_mar[:, 1, 1]
    dcx = cam_mar[:, 0, 2]
    dcy = cam_mar[:, 1, 2]
    
    x = (xy[:, 0, :] - dcx.view(-1, 1)) / dfx.view(-1, 1)
    y = (xy[:, 1, :] - dcy.view(-1, 1)) / dfy.view(-1, 1)
    xy = torch.stack([x, y], dim=1) #[B, 2, N]
    thetad = torch.norm(xy, dim=1) #[B, N]
    theta = thetad
    for i in range(10):
        thetan = 1 + k[:, 0].view(-1, 1)*(theta**2) + \
        k[:, 1].view(-1, 1)*(theta**4) + \
        k[:, 2].view(-1, 1)*(theta**6) + \
        k[:, 3].view(-1, 1)*(theta**8)
        
        theta = thetad / thetan #[B, N]
        
    x_undist = xy[:, 0, :] * torch.tan(theta)/thetad
    y_undist = xy[:, 1, :] * torch.tan(theta)/thetad
    xy_undist = torch.stack([x_undist, y_undist], dim=1) #[B, 2, N]
    
    xyf_undist = torch.cat([xy_undist, torch.ones(B, 1, N).to(xy_undist.device)], dim=1) #[B, 3, N]
    xyf_rect = torch.matmul(rect_mar, xyf_undist) #[B, 3, N]
    xy_rect = f.view(-1, 1, 1) * xyf_rect[:, :2, :] / xyf_rect[:, 2:, :] #[B, 2, N]
    xy_rect[:, 0, :] += cx.view(-1, 1)
    xy_rect[:, 1, :] += cy.view(-1, 1)
    
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
