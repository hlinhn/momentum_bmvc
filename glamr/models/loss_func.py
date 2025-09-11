from lib.utils.torch_transform import (
    quat_angle_diff, 
    get_heading, 
    rot6d_to_quat, 
    vec_to_heading
)
import torch
import torch_dct as dct
from torch import nn
from lib.utils.konia_transform import (
    quaternion_to_rotation_matrix,
    rotation_matrix_to_angle_axis
)
from loguru import logger


def rotate_part_rotation(part_rot_mats, grot_mat):
    B, T, P = part_rot_mats.shape[:3]
    grot_mat = grot_mat.transpose(0, 1)
    grot_mat = grot_mat.unsqueeze(2)
    grot_mat = grot_mat.repeat(1, 1, P, 1, 1)
    part_rot = torch.bmm(grot_mat.reshape(-1, 3, 3),
                         part_rot_mats.reshape(-1, 3, 3)).reshape(B, T, P, 3, 3)
    return part_rot    


def rotate_part_centroid(centroids, grot_mat, root_joint, gtrans=None):
    B, T, P = centroids.shape[:3]
    grot_mat = grot_mat.transpose(0, 1)
    grot_mat = grot_mat.unsqueeze(2)
    grot_mat = grot_mat.repeat(1, 1, P, 1, 1)
    center = root_joint.repeat(1, 1, P, 1)
    centroids_rot = torch.bmm(
        grot_mat.reshape(-1, 3, 3),
        centroids.reshape(-1, 3, 1) - center.reshape(-1, 3, 1))\
            .reshape(B, T, P, 3) + center
    if gtrans is not None:
        gtrans = gtrans.transpose(0, 1)
        centroids_rot = centroids_rot + gtrans.unsqueeze(2)
    return centroids_rot


def compute_momentum_root(data, pred_root, pred_trans):
    part_mass = data['part_mass'] # B, P
    B, P = part_mass.shape[:2]
    part_moi = data['part_moi'] # B, P, 3, 3
    unrot_centroid = data['centroid'] # B, T, P, 3
    center = data['center'] # B, T, 1, 3
    unrot_rotation = data['part_rotation'] # B, T, P, 3, 3
    root_mat = quaternion_to_rotation_matrix(pred_root)
    part_rotation = rotate_part_rotation(unrot_rotation, root_mat)
    part_centroid = rotate_part_centroid(unrot_centroid, root_mat, center, None) # B, T, P, 3

    vel = part_centroid[:, 1:] - part_centroid[:, :-1]
    lm_transfer = part_mass.unsqueeze(-1).unsqueeze(1) * vel
    lm = torch.sum(lm_transfer, axis=2) # B, T, 3
    pred_trans_tp = pred_trans.transpose(0, 1)
    lm = lm + (pred_trans_tp[:, 1:] - pred_trans_tp[:, :-1])
    T = vel.shape[1]
    rvel = torch.bmm(part_rotation[:, 1:].reshape(-1, 3, 3),
                     part_rotation[:, :-1].transpose(-1, -2).reshape(-1, 3, 3))
    transfer = part_mass.unsqueeze(-1).unsqueeze(1) * \
        torch.cross(part_centroid[:, :-1], vel, dim=-1)
    moi_seq = part_moi.unsqueeze(1).repeat(1, T, 1, 1, 1)\
       .reshape(-1, 3, 3).cuda()
    R = part_rotation[:, :-1].reshape(-1, 3, 3).float()
    part_I = torch.bmm(torch.bmm(R, moi_seq), R.transpose(-2, -1))
    w = rotation_matrix_to_angle_axis(rvel).reshape(-1, 3).unsqueeze(-1).float()
    Li = torch.bmm(part_I, w).reshape(B, T, -1, 3)
    am = torch.sum(Li, axis=2) + torch.sum(transfer, axis=2) # B, T, 3
    return am, lm


def batch_fft(am, T, batched_dct):
    am = am.transpose(0, 1).reshape(T, -1)
    fft = batched_dct(am)
    fft = fft.reshape(T, -1, 3).transpose(0, 1)
    return fft


def compute_tmo(data, specs):
    batched_dct = torch.func.vmap(dct.dct, in_dims=-1, out_dims=-1)
    mode = specs.get('mode', 'train')
    pred_root = data[f'{mode}_out_orient_q_tp'] # T, B, 4
    pred_trans = data[f'{mode}_out_trans_tp']
    pred_am, pred_lm = compute_momentum_root(data, pred_root, pred_trans)
    gt_am, gt_lm = compute_momentum_root(data, data[f'orient_q_tp'], data['trans_tp'])
    gt_am_v = gt_am[:, 1:] - gt_am[:, :-1]
    pred_am_v = pred_am[:, 1:] - pred_am[:, :-1]
    gt_lm_v = gt_lm[:, 1:] - gt_lm[:, :-1]
    pred_lm_v = pred_lm[:, 1:] - pred_lm[:, :-1]
    pred_fft = batch_fft(pred_am, pred_am.shape[1], batched_dct)
    gt_fft = batch_fft(gt_am, gt_am.shape[1], batched_dct)

    am_diff = (pred_am - gt_am).pow(2).sum(-1).sum(-1).mean()
    lm_diff = (pred_lm - gt_lm).pow(2).sum(-1).sum(-1).mean()
    am_v_diff = (pred_am_v - gt_am_v).pow(2).sum(-1).sum(-1).mean()
    lm_v_diff = (pred_lm_v - gt_lm_v).pow(2).sum(-1).sum(-1).mean()
    am_fft_diff = (pred_fft - gt_fft).pow(2).sum(-1).sum(-1).mean()

    total_loss = am_diff * 100 + lm_diff * 100 + am_v_diff * 100 + lm_v_diff * 100 + am_fft_diff
    return total_loss


def compute_trans_mse(data, specs):
    mode = specs.get('mode', 'train')
    use_frame_loss_mask = specs.get('use_frame_loss_mask', False)
    diff = data[f'{mode}_out_trans_tp'] - data[f'trans_tp']
    if use_frame_loss_mask:
        diff *= data['frame_loss_mask'].transpose(0, 1)
    mse = diff.pow(2).sum(-1).mean()
    return mse


def compute_orient_angle_loss(data, specs):
    mode = specs.get('mode', 'train')
    use_frame_loss_mask = specs.get('use_frame_loss_mask', False)
    angle = quat_angle_diff(data[f'{mode}_out_orient_q_tp'], data[f'orient_q_tp'])
    if use_frame_loss_mask:
        angle *= data['frame_loss_mask'].transpose(0, 1).squeeze(-1)
    angle_loss = angle.pow(2).mean()
    return angle_loss


def compute_orient_6d_loss(data, specs):
    mode = specs.get('mode', 'train')
    use_frame_loss_mask = specs.get('use_frame_loss_mask', False)
    diff = data[f'{mode}_out_orient_6d_tp'] - data[f'orient_6d_tp']
    if use_frame_loss_mask:
        diff *= data['frame_loss_mask'].transpose(0, 1)
    mse = diff.pow(2).sum(-1).mean()
    return mse


def compute_vae_z_kld(data, specs):
    clamp_before_mean = specs.get('clamp_before_mean', True)
    kld = data['q_z_dist'].kl(data['p_z_dist'])
    kld = kld.sum(-1)
    if clamp_before_mean:
        kld = kld.clamp_min_(specs['min_clip']).mean()
    else:
        kld = kld.mean().clamp_min_(specs['min_clip'])
    return kld


def compute_local_orient_heading(data, specs):
    local_traj = data[f'train_out_local_traj_tp']
    local_orient = local_traj[..., 3:-2]
    if local_orient.shape[-1] == 6:
        local_orient = rot6d_to_quat(local_orient)
    heading = get_heading(local_orient)
    mse = heading.pow(2).mean()
    return mse


def compute_dheading(data, specs):
    local_traj = data[f'train_out_local_traj_tp']
    local_heading_vec = local_traj[..., -2:]
    heading = vec_to_heading(local_heading_vec)
    mse = heading.pow(2).mean()
    return mse


loss_func_dict = {
    'trans_mse': compute_trans_mse,
    'orient_angle': compute_orient_angle_loss,
    'vae_z_kld': compute_vae_z_kld,
    'orient_6d': compute_orient_6d_loss,
    'local_orient_heading': compute_local_orient_heading,
    'dheading': compute_dheading,
    'tmo': compute_tmo
}