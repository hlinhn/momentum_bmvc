import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from scipy.ndimage import gaussian_filter1d
import pickle
import numpy as np
import glob 
import h5py
from loguru import logger
import roma

import sys
from lib.utils.konia_transform import (
    angle_axis_to_rotation_matrix,
    rotation_matrix_to_angle_axis
)
import os
import json


def get_seq(seq, idx, seq_len, append=0):
    if append == 0:
        return seq[idx: idx + seq_len].astype(np.float32)
    else:
        tile_shape = [1] * len(seq.shape)
        tile_shape[0] = append
        seq_data = np.vstack([seq[idx:], np.tile(seq[[-1]], tile_shape)]).astype(np.float32)
        return seq_data


def blend(source, target, seqlen, degree):
    source_copy = np.copy(source)
    target_body = torch.tensor(target[:seqlen, 6:75])
    source_body = torch.tensor(source[:seqlen, 6:75])
    blended = roma.utils.rotvec_slerp(source_body.reshape(-1, 3), target_body.reshape(-1, 3), torch.tensor(degree))
    source_copy[:seqlen, 6:75] = blended.reshape(seqlen, -1)
    return source_copy


def get_am_data(info, start, seq_len, orig_len):
    mass = info.get("mass")[:]
    moi = info.get("moi")[:]
    centroid_seq = info.get("centroid")
    part_rot_seq = info.get("rotation")
    center_seq = info.get("center")
    am_seq = info.get("AM")
    if start + seq_len <= orig_len:
        append = 0
        am_append = 0
    else:
        append = seq_len - (orig_len - start)
        am_append = append
    centroid = get_seq(centroid_seq, start, seq_len, append)
    part_rot = get_seq(part_rot_seq, start, seq_len, append)
    center = get_seq(center_seq, start, seq_len, append)
    am = get_seq(am_seq, start, seq_len - 1, am_append)
    return mass, moi, centroid, part_rot, center, am


class AMASSDataset(Dataset):

    def __init__(self, dataset_dir, split, cfg=None, training=True, seq_len=64, ntime_per_epoch=10000):
        self.cfg = cfg
        self.data_file = h5py.File(f"{dataset_dir}/amass_{split}.h5", "r")
        self.jpos_file = h5py.File(f"{dataset_dir}/amass_{split}_jpos.h5", "r")
        self.am_file = h5py.File(f"{dataset_dir}/amass_{split}_momentum.h5", "r")
        
        self.sequences = list(self.data_file.keys())
        self.split = split
        self.training = training
        self.seq_len = seq_len
        self.ntime_per_epoch = ntime_per_epoch
        self.epoch_init_seed = None
        # compute sampling probablity
        self.seq_lengths = np.array([x.shape[0] for x in self.data_file.values()])
        if cfg is not None and cfg.seq_sampling_method == 'length':
            self.seq_prob = self.seq_lengths / self.seq_lengths.sum()
        else:
            self.seq_prob = None

    def __len__(self):
       return self.ntime_per_epoch // self.seq_len

    def set_seq_len(self, seq_len):
        self.seq_len = seq_len        

    def random_sample(self, idx=0):
        if self.epoch_init_seed is None:
            # the above step is necessary for lightning's ddp parallel computing because each node gets a subset (idx) of the dataset
            self.epoch_init_seed = (np.random.get_state()[1][0] * len(self) + idx) % int(1e8)
            np.random.seed(self.epoch_init_seed)
        
        sind = np.random.choice(len(self.sequences), p=self.seq_prob)
        seq = self.sequences[sind]
        seq_jpos, seq_jpos_noshape = self.jpos_file.get(seq)
        seq_am = self.am_file.get(seq)

        orig_seq_data = self.data_file.get(seq)
        if self.seq_len <= orig_seq_data.shape[0]:
            fr_start = np.random.randint(orig_seq_data.shape[0] - self.seq_len + 1)
            append = 0
            frame_loss_mask = np.ones((self.seq_len, 1)).astype(np.float32)
            eff_seq_len = self.seq_len   # effective seq
        else:
            fr_start = 0
            eff_seq_len = orig_seq_data.shape[0]  # effective seq
            append = self.seq_len - eff_seq_len
            frame_loss_mask = np.zeros((self.seq_len, 1)).astype(np.float32)
            frame_loss_mask[:eff_seq_len] = 1.0

        seq_data = get_seq(orig_seq_data, fr_start, self.seq_len, append)
        jpos = get_seq(seq_jpos, fr_start, self.seq_len, append)
        jpos_noshape = get_seq(seq_jpos_noshape, fr_start, self.seq_len, append)
        mass, moi, centroid, part_rot, center, am = get_am_data(seq_am, fr_start, self.seq_len, orig_seq_data.shape[0])

        data = {
            'trans': seq_data[:, :3],
            'pose': seq_data[:, 3:75],
            'shape': seq_data[:, 75:],
            'seq_name': seq,
            'frame_loss_mask': frame_loss_mask,
            'fr_start': fr_start,
            'eff_seq_len': eff_seq_len,
            'joint_pos_shape': jpos[:, 1:, :].reshape(jpos.shape[0], -1),
            'joint_pos_noshape': jpos_noshape[:, 1:, :].reshape(jpos_noshape.shape[0], -1),
            'part_mass': mass,
            'part_moi': moi,
            'centroid': centroid,
            'center': center,
            'part_rotation': part_rot,
            'am': am
        }

        # mask
        self.generate_mask(data)

        # gaussian smoothing for data augmentation
        if self.cfg is not None and self.cfg.pose_gaussian_smooth is not None:
            in_body_pose = seq_data[:, 6:75]
            d = self.cfg.pose_gaussian_smooth
            if np.random.binomial(1, d['prob']):
                sigma = np.random.uniform(d['sigma_lb'], d['sigma_ub'])
                # print('smoothing', sigma)
                in_body_pose = gaussian_filter1d(in_body_pose.copy(), sigma=sigma, axis=0, mode='nearest')
            in_body_pose *= data['pose_mask'][:, 3:]
            data['in_body_pose'] = in_body_pose
        return data

    def generate_mask(self, data):
        mask_methods = self.cfg.data_mask_methods if self.cfg is not None else dict()
        pose_mask = np.ones_like(data['pose'])
        frame_mask = np.ones(data['pose'].shape[0]).astype(np.float32)
        for method, specs in mask_methods.items():
            if method == 'drop_frames':
                preserve_first_n = specs.get('preserve_first_n', 1)
                preserve_last_n = specs.get('preserve_last_n', 0)
                drop_len = np.random.randint(specs['min_drop_len'], specs['max_drop_len'] + 1)
                start_fr_min = preserve_first_n
                start_fr_max = min(data['pose'].shape[0] - drop_len + 1 - preserve_last_n, data['eff_seq_len'])
                start_fr = np.random.randint(start_fr_min, start_fr_max)
                end_fr = min(start_fr + drop_len, pose_mask.shape[0])
                pose_mask[start_fr: end_fr] = 0.0
                frame_mask[start_fr: end_fr] = 0.0
                data['num_drop_fr'] = end_fr - start_fr
        data['pose_mask'] = pose_mask
        data['frame_mask'] = frame_mask

    def __getitem__(self, idx):
        return self.random_sample(idx)


if __name__ == "__main__":

    np.random.seed(0)
    torch.manual_seed(0)
    amass_dir = 'datasets/amass_processed/v5'

    dataset = AMASSDataset(amass_dir, 'test', seq_len=100)

    batch_size = 5
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    batch = next(iter(dataloader))
    print(batch.keys())
    B, T, _ = batch["joint_pos_shape"].shape
    joint_pos = batch["joint_pos_shape"]
    with_gt = angle_axis_to_rotation_matrix(batch['pose'][:, :, :3])
    with_gt = with_gt.unsqueeze(-3).expand(-1, -1, 23, -1, -1)
    rotated_joint = torch.bmm(
        with_gt.reshape(-1, 3, 3),
        joint_pos.reshape(-1, 3, 1)
    ).reshape(B, T, -1)
    with_zero = torch.concatenate([torch.zeros(B, T, 3).to(joint_pos.device), rotated_joint], dim=-1).reshape(B, T, -1, 3)
    with_trans = with_zero + batch['trans'].unsqueeze(-2)
    pred_jitter = torch.norm(
        (with_trans[:, 3:] - 3 * with_trans[:, 2:-1] + 3 * with_trans[:, 1:-2] - with_trans[:, :-3]) * (30**3),
        dim=2,
    ).mean(dim=-1) / 10.0

    print(pred_jitter.mean())
