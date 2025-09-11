import torch
import torch.nn as nn
from smplx import SMPL, SMPLH
import numpy as np
import os
import roma
import torch_dct as dct
from path_variable import SMPL_PATH, SMPLH_PATH, SEGMENTATION_PATH


PART_ORDER = ['hips', 'leftUpLeg', 'rightUpLeg', 
              'spine', 'leftLeg', 'rightLeg', 
              'spine1', 'leftFoot', 'rightFoot', 
              'upperBody', 'leftToeBase', 'rightToeBase', 
              'neck', 'head', 
              'leftArm', 'rightArm', 
              'leftForeArm', 'rightForeArm', 
              'leftHand_f', 'rightHand_f']


def rotation_matrix_to_angle_axis(rotation_matrix: torch.Tensor) -> torch.Tensor:
    r"""Convert 3x3 rotation matrix to Rodrigues vector.

    Args:
        rotation_matrix: rotation matrix.

    Returns:
        Rodrigues vector transformation.

    Shape:
        - Input: :math:`(N, 3, 3)`
        - Output: :math:`(N, 3)`

    Example:
        >>> input = torch.rand(2, 3, 3)  # Nx3x3
        >>> output = rotation_matrix_to_angle_axis(input)  # Nx3
    """
    if not isinstance(rotation_matrix, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(rotation_matrix)}")

    if not rotation_matrix.shape[-2:] == (3, 3):
        raise ValueError(f"Input size must be a (*, 3, 3) tensor. Got {rotation_matrix.shape}")
    quaternion: torch.Tensor = rotation_matrix_to_quaternion(rotation_matrix)
    return quaternion_to_angle_axis(quaternion)


def quaternion_to_angle_axis(
    quaternion: torch.Tensor, eps: float = 1.0e-6
) -> torch.Tensor:
    """Convert quaternion vector to angle axis of rotation.

    The quaternion should be in (x, y, z, w) or (w, x, y, z) format.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        quaternion: tensor with quaternions.
        order: quaternion coefficient order. Note: 'xyzw' will be deprecated in favor of 'wxyz'.

    Return:
        tensor with angle axis of rotation.

    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`

    Example:
        >>> quaternion = torch.rand(2, 4)  # Nx4
        >>> angle_axis = quaternion_to_angle_axis(quaternion)  # Nx3
    """

    if not quaternion.shape[-1] == 4:
        raise ValueError(f"Input must be a tensor of shape Nx4 or 4. Got {quaternion.shape}")

    # unpack input and compute conversion
    q1: torch.Tensor = torch.tensor([])
    q2: torch.Tensor = torch.tensor([])
    q3: torch.Tensor = torch.tensor([])
    cos_theta: torch.Tensor = torch.tensor([])

    cos_theta = quaternion[..., 0]
    q1 = quaternion[..., 1]
    q2 = quaternion[..., 2]
    q3 = quaternion[..., 3]

    sin_squared_theta: torch.Tensor = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta: torch.Tensor = torch.sqrt((sin_squared_theta).clamp_min(eps))
    two_theta: torch.Tensor = 2.0 * torch.where(
        cos_theta < 0.0, torch_safe_atan2(-sin_theta, -cos_theta), torch_safe_atan2(sin_theta, cos_theta)
    )

    k_pos: torch.Tensor = safe_zero_division(two_theta, sin_theta, eps)
    k_neg: torch.Tensor = 2.0 * torch.ones_like(sin_theta)
    k: torch.Tensor = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis: torch.Tensor = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis


def torch_safe_atan2(y, x, eps: float = 1e-6):
    y_abs = y.abs()
    x_abs = x.abs()
    mask = (y_abs < eps) & (x_abs < eps)
    y = torch.where(mask, y + eps, y)
    return torch.atan2(y, x)


def rotation_matrix_to_quaternion(
    rotation_matrix: torch.Tensor, eps: float = 1.0e-6
) -> torch.Tensor:
    r"""Convert 3x3 rotation matrix to 4d quaternion vector.

    The quaternion vector has components in (w, x, y, z) or (x, y, z, w) format.

    .. note::
        The (x, y, z, w) order is going to be deprecated in favor of efficiency.

    Args:
        rotation_matrix: the rotation matrix to convert.
        eps: small value to avoid zero division.
        order: quaternion coefficient order. Note: 'xyzw' will be deprecated in favor of 'wxyz'.

    Return:
        the rotation in quaternion.

    Shape:
        - Input: :math:`(*, 3, 3)`
        - Output: :math:`(*, 4)`

    Example:
        >>> input = torch.rand(4, 3, 3)  # Nx3x3
        >>> output = rotation_matrix_to_quaternion(input, eps=torch.finfo(input.dtype).eps,
        ...                                        order=QuaternionCoeffOrder.WXYZ)  # Nx4
    """
    if not isinstance(rotation_matrix, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(rotation_matrix)}")

    if not rotation_matrix.shape[-2:] == (3, 3):
        raise ValueError(f"Input size must be a (*, 3, 3) tensor. Got {rotation_matrix.shape}")

    rotation_matrix_vec: torch.Tensor = rotation_matrix.view(*rotation_matrix.shape[:-2], 9)

    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.chunk(rotation_matrix_vec, chunks=9, dim=-1)

    trace: torch.Tensor = m00 + m11 + m22

    def trace_positive_cond():
        sq = torch.sqrt((trace + 1.0).clamp_min(eps)) * 2.0  # sq = 4 * qw.
        qw = 0.25 * sq
        qx = safe_zero_division(m21 - m12, sq)
        qy = safe_zero_division(m02 - m20, sq)
        qz = safe_zero_division(m10 - m01, sq)
        return torch.cat((qw, qx, qy, qz), dim=-1)

    def cond_1():
        sq = torch.sqrt((1.0 + m00 - m11 - m22).clamp_min(eps)) * 2.0  # sq = 4 * qx.
        qw = safe_zero_division(m21 - m12, sq)
        qx = 0.25 * sq
        qy = safe_zero_division(m01 + m10, sq)
        qz = safe_zero_division(m02 + m20, sq)
        return torch.cat((qw, qx, qy, qz), dim=-1)

    def cond_2():
        sq = torch.sqrt((1.0 + m11 - m00 - m22).clamp_min(eps)) * 2.0  # sq = 4 * qy.
        qw = safe_zero_division(m02 - m20, sq)
        qx = safe_zero_division(m01 + m10, sq)
        qy = 0.25 * sq
        qz = safe_zero_division(m12 + m21, sq)
        return torch.cat((qw, qx, qy, qz), dim=-1)

    def cond_3():
        sq = torch.sqrt((1.0 + m22 - m00 - m11).clamp_min(eps)) * 2.0  # sq = 4 * qz.
        qw = safe_zero_division(m10 - m01, sq)
        qx = safe_zero_division(m02 + m20, sq)
        qy = safe_zero_division(m12 + m21, sq)
        qz = 0.25 * sq
        return torch.cat((qw, qx, qy, qz), dim=-1)

    where_2 = torch.where(m11 > m22, cond_2(), cond_3())
    where_1 = torch.where((m00 > m11) & (m00 > m22), cond_1(), where_2)

    quaternion: torch.Tensor = torch.where(trace > 0.0, trace_positive_cond(), where_1)
    return quaternion


def safe_zero_division(numerator: torch.Tensor, denominator: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    denominator = denominator.clone()
    denominator = torch.where(denominator.abs() < eps, torch.tensor(eps, device=denominator.device), denominator)
    return numerator / denominator


def compute_inertia(part, data, part_faces, all_vertices):
    vertices = data[part]
    faces = part_faces.get(part)
    estimated_centroid = torch.mean(all_vertices[0][vertices], dim=0, keepdims=True)
    part_vertices = all_vertices[0] - estimated_centroid
    concat_faces = faces.reshape(-1)
    mats = part_vertices[concat_faces].reshape(-1, 3, 3)
    dets = torch.abs(torch.det(mats))
    centroid_cont = torch.mean(mats, dim=1)
    centroid = torch.sum(centroid_cont * torch.unsqueeze(dets, dim=1) / 6, dim=0) / 4
    total_volume = torch.sum(dets) / 6
    vert_sum = torch.sum(mats, dim=1)
    volumes = dets / 6

    sq = mats ** 2
    xy = mats[:, :, 0] * mats[:, :, 1]
    yz = mats[:, :, 1] * mats[:, :, 2]
    zx = mats[:, :, 2] * mats[:, :, 0]
    ssq = vert_sum ** 2
    sxy = vert_sum[:, 0] * vert_sum[:, 1]
    syz = vert_sum[:, 1] * vert_sum[:, 2]
    szx = vert_sum[:, 2] * vert_sum[:, 0]
    inertia_mat = torch.zeros((3, 3)).to(total_volume.device)
    inertia_mat[0, 0] = torch.sum((torch.sum(sq[:, :, [1, 2]], dim=(1, 2)) + torch.sum(ssq[:, [1, 2]], dim=1)) * volumes)
    inertia_mat[1, 1] = torch.sum((torch.sum(sq[:, :, [0, 2]], dim=(1, 2)) + torch.sum(ssq[:, [0, 2]], dim=1)) * volumes)
    inertia_mat[2, 2] = torch.sum((torch.sum(sq[:, :, [1, 0]], dim=(1, 2)) + torch.sum(ssq[:, [1, 0]], dim=1)) * volumes)
    inertia_mat[0, 1] = -torch.sum((torch.sum(xy, dim=1) + sxy) * volumes)
    inertia_mat[0, 2] = -torch.sum((torch.sum(zx, dim=1) + szx) * volumes)
    inertia_mat[1, 2] = -torch.sum((torch.sum(yz, dim=1) + syz) * volumes)
    inertia_mat[1, 0] = inertia_mat[0, 1]
    inertia_mat[2, 0] = inertia_mat[0, 2]
    inertia_mat[2, 1] = inertia_mat[1, 2]
    # assume rho = mass / volume, mass = 1
    return inertia_mat / 20, total_volume    


def calculate_part_com(vertices, part, partition):
    vertices_idx = partition[part]
    estimated_centroid = torch.mean(vertices[:, vertices_idx], dim=1, keepdims=True)
    return estimated_centroid


def calculate_part_rotation(rot_mats, parents, upright=False):
    rotation_chain = [rot_mats[:, 0]]
    for i in range(1, parents.shape[0]):
        curr_res = torch.matmul(rotation_chain[parents[i]],
                                rot_mats[:, i])
        rotation_chain.append(curr_res)
    rotation_chain = torch.stack(rotation_chain, dim=1)
    return rotation_chain


class MomentumCalculator(nn.Module):
    def __init__(self, num_iter=100, to_cuda=False, smplh=False):
        super().__init__()
        self.num_iter = num_iter
        if not smplh:
            model_path = SMPL_PATH
            self.smpl_model = SMPL(model_path)
        else:
            model_path = SMPLH_PATH
            self.smpl_model = SMPLH(model_path, use_pca=False)
        self.smplh = smplh
        self.device = torch.device('cpu')
        if to_cuda:
            self.smpl_model = self.smpl_model.cuda()
            self.device = torch.device('cuda')
        fv_path = SEGMENTATION_PATH
        self.fv_data = np.load(fv_path, allow_pickle=True)
        self.fv_data_verts = self.fv_data.item().get('vertices')
        self.part_parents = np.array([-1, 0, 0, 0, 1, 2, 3, 
                                      4, 5, 6, 7, 8, 9, 12, 
                                      9, 9, 14, 15, 16, 17])
        self.parent_joints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
                             11, 12, 15, 16, 17, 18, 19, 20, 21]

    def canon_body(self, shape=None):
        device = self.device
        if shape is None:
            if self.smplh:
                shape = torch.zeros((1, 16)).float().to(device)
            else:
                shape = torch.zeros((1, 10)).float().to(device)
        shape = shape.to(self.device)
        empty_pose = torch.zeros((1, 69)).float().to(device)
        no_rot = torch.zeros((1, 3)).float().to(device)
        no_trans = torch.zeros((1, 3)).float().to(device)
        return self.mesh_body(empty_pose, no_rot, no_trans, shape)
    
    def run_smpl(self, pose, betas, grot, gtrans):
        if not self.smplh:
            body = self.smpl_model(
                body_pose=pose.float(),
                betas=betas,
                global_orient=grot.float(),
                transl=gtrans.float()
            )
        else:
            T = pose.shape[0]
            empty_hand = torch.zeros((T, 45)).float().to(self.device)
            body = self.smpl_model(
                betas=betas,
                global_orient=grot.float(),
                transl=gtrans.float(),
                body_pose=pose[:, :63].float(),
                left_hand_pose=empty_hand,
                right_hand_pose=empty_hand
            )
        return body

    def mesh_body(self, pose, grot, gtrans, shape=None):
        if shape is None:
            shape = self.shape
            shape = shape.to(self.device)
        T = pose.shape[0]
        assert torch.is_tensor(shape)
        if len(shape.shape) == 1:
            betas = shape.unsqueeze(0).float().repeat(T, 1)
        if shape.shape[0] == 1:
            betas = shape.float().repeat(T, 1)
        else:
            betas = shape
        if gtrans is None:
            gtrans = torch.zeros((T, 3)).float().to(self.device)
        return self.run_smpl(pose, betas, grot, gtrans)
    
    def mesh_body_zero_hand(self, pose, grot, gtrans, shape=None, aa=False):
        if shape is None:
            shape = self.shape
            shape = shape.to(self.device)
        T = pose.shape[0]
        assert torch.is_tensor(shape)
        if len(shape.shape) == 1:
            betas = shape.unsqueeze(0).float().repeat(T, 1)
        if shape.shape[0] == 1:
            betas = shape.float().repeat(T, 1)
        else:
            betas = shape
        betas = betas.to(self.device)
        if gtrans is None:
            gtrans = torch.zeros((T, 3)).float().to(self.device)
        zero_hand = torch.zeros((T, 6)).float().to(self.device)
        if not aa:
            pose_aa = rotation_matrix_to_angle_axis(pose.reshape(-1, 3, 3)).reshape(T, 63)
            grot_aa = rotation_matrix_to_angle_axis(grot.reshape(-1, 3, 3)).reshape(T, 3)
        else:
            pose_aa = pose
            grot_aa = grot
        total_pose = torch.cat([pose_aa, zero_hand], dim=-1)
        return self.run_smpl(total_pose, betas, grot_aa, gtrans)

    def get_part_info(self, body_shape):
        body_mesh = self.canon_body(body_shape)
        vert_segm = self.fv_data.item().get('vertices')
        face_segm = self.fv_data.item().get('faces')
        full_body_verts = torch.cat([body_mesh.vertices,
                                     body_mesh.joints[:, :22]], axis=1)
        zero_centroid = torch.cat([calculate_part_com(full_body_verts, part, self.fv_data_verts)
                                   for part in PART_ORDER], dim=1)
        parent_joints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
                         11, 12, 15, 16, 17, 18, 19, 20, 21]
        relative_vector = zero_centroid - body_mesh.joints[0, parent_joints]
        self.part_parent_joint = parent_joints
        part_data = [compute_inertia(part, vert_segm, face_segm, full_body_verts) 
                     for part in PART_ORDER]
        part_moi, part_mass = list(zip(*part_data))
        part_mass = torch.stack(part_mass, dim=0)
        mass_sum = torch.sum(part_mass)
        self.part_mass = part_mass / mass_sum # to have mass = 1
        self.part_moi = torch.stack(part_moi, dim=0)
        self.zero_centroid = zero_centroid
        self.relative_vector = relative_vector
        self.canon_joints = body_mesh.joints
        self.shape = body_shape
        self.shape = self.shape.to(self.device)

    def precompute_part_centroid(self, pose):
        T = pose.shape[0]
        zero_grot = torch.zeros((T, 3))
        body_meshes = self.mesh_body(pose, zero_grot, None, self.shape)
        verts = torch.cat([body_meshes.vertices, body_meshes.joints[:, :22]], axis=1)
        centroids = torch.cat([calculate_part_com(verts, part, self.fv_data_verts)
                               for part in PART_ORDER], dim=1)
        return centroids, body_meshes.joints[:, [0]]

    def centroid_from_mesh(self, body_meshes):
        verts = torch.cat([body_meshes.vertices, body_meshes.joints[:, :22]], axis=1)
        centroids = torch.cat([calculate_part_com(verts, part, self.fv_data_verts)
                               for part in PART_ORDER], dim=1)
        return centroids

    def part_centroid(self, pose, grot, gtrans, return_jpos=False):
        body_meshes = self.mesh_body(pose, grot, gtrans, self.shape)
        verts = torch.cat([body_meshes.vertices, body_meshes.joints[:, :22]], axis=1)
        centroids = torch.cat([calculate_part_com(verts, part, self.fv_data_verts)
                               for part in PART_ORDER], dim=1)
        if not return_jpos:
            return centroids
        else:
            return centroids, body_meshes.joints[:, :22]

    def setup_shape(self, body_shape):
        self.get_part_info(body_shape=body_shape)

    def setup_zero_shape(self):
        zero_shape = torch.zeros((1, 10))
        self.setup_shape(zero_shape)

    def part_rotation(self, pose, grot):
        T = pose.shape[0]
        rot_mats = roma.rotvec_to_rotmat(pose.reshape(-1, 3))
        grot_mat = roma.rotvec_to_rotmat(grot.reshape(-1, 3)).reshape(T, 1, 3, 3)
        rot_mats = rot_mats.reshape(T, -1, 3, 3)
        replaced_arm = torch.bmm(rot_mats[:, [12, 13]].reshape(-1, 3, 3), 
                                 rot_mats[:, [15, 16]].reshape(-1, 3, 3))
        part_rot_mats = torch.cat([
            grot_mat,
            rot_mats[:, :12],
            rot_mats[:, [14]],
            replaced_arm.reshape(T, 2, 3, 3),
            rot_mats[:, 17:21]
        ], dim=1)
        return calculate_part_rotation(part_rot_mats, self.part_parents)

    def linear_momentum(self, pose, grot, gtrans):
        pred_centroids = self.part_centroid(pose, grot, gtrans)
        vel = pred_centroids[1:] - pred_centroids[:-1]
        momentum = self.part_mass.unsqueeze(-1).unsqueeze(0) * vel
        momentum = torch.sum(momentum, axis=1)
        return momentum
    
    def linear_momentum_from_centroids(self, centroids):
        vel = centroids[1:] - centroids[:-1]
        momentum = self.part_mass.unsqueeze(-1).unsqueeze(0) * vel
        momentum = torch.sum(momentum, axis=1)
        return momentum

    def linear_momentum_from_mesh(self, mesh_body):
        verts = torch.cat([mesh_body.vertices, mesh_body.joints[:, :22]], axis=1)
        centroids = torch.cat([calculate_part_com(verts, part, self.fv_data_verts)
                               for part in PART_ORDER], dim=1)
        vel = centroids[1:] - centroids[:-1]
        momentum = self.part_mass.unsqueeze(-1).unsqueeze(0) * vel
        momentum = torch.sum(momentum, axis=1)
        return momentum

    def momentum_precomp(self, pred_centroids, pred_rotation, mass, moi):
        vel = pred_centroids[1:] - pred_centroids[:-1]
        rvel = torch.bmm(pred_rotation[1:].reshape(-1, 3, 3),
                         pred_rotation[:-1].transpose(-1, -2).reshape(-1, 3, 3))
        transfer = mass.unsqueeze(-1).unsqueeze(0) * \
            torch.cross(pred_centroids[:-1], vel, dim=-1)
        momentum = mass.unsqueeze(-1).unsqueeze(0) * vel
        momentum = torch.sum(momentum, axis=1)
        moi_seq = moi.unsqueeze(0).repeat(vel.shape[0], 1, 1, 1)\
            .reshape(-1, 3, 3)
        R = pred_rotation[:-1].reshape(-1, 3, 3).float()
        part_moi = torch.bmm(torch.bmm(R, moi_seq), 
                             R.transpose(-2, -1))
        w = roma.rotmat_to_rotvec(rvel).reshape(-1, 3).unsqueeze(-1).float()
        Li = torch.bmm(part_moi, w)
        Li = Li.reshape(vel.shape[0], -1, 3)
        local_total = torch.sum(Li, axis=1)
        transfer_total = torch.sum(transfer, axis=1)
        return local_total + transfer_total, momentum

    def angular_momentum_from_centroids(self, pred_centroids, pred_rotation):
        vel = pred_centroids[1:] - pred_centroids[:-1]
        rvel = torch.bmm(pred_rotation[1:].reshape(-1, 3, 3),
                         pred_rotation[:-1].transpose(-1, -2).reshape(-1, 3, 3))
        transfer = self.part_mass.unsqueeze(-1).unsqueeze(0) * \
            torch.cross(pred_centroids[:-1], vel, dim=-1)
        moi_seq = self.part_moi.unsqueeze(0).repeat(vel.shape[0], 1, 1, 1)\
            .reshape(-1, 3, 3)
        R = pred_rotation[:-1].reshape(-1, 3, 3).float()
        part_moi = torch.bmm(torch.bmm(R, moi_seq), 
                             R.transpose(-2, -1))
        w = rotation_matrix_to_angle_axis(rvel).reshape(-1, 3).unsqueeze(-1).float()
        Li = torch.bmm(part_moi, w)
        Li = Li.reshape(vel.shape[0], -1, 3)
        local_total = torch.sum(Li, axis=1)
        transfer_total = torch.sum(transfer, axis=1)
        return local_total + transfer_total

    def momenta_single_component(self, centroid, rotation, part):
        vel = centroid[1:] - centroid[:-1]
        rvel = torch.bmm(rotation[1:].reshape(-1, 3, 3),
                         rotation[:-1].transpose(-1, -2).reshape(-1, 3, 3))
        transfer = self.part_mass[part] * \
            torch.cross(centroid[:-1], vel, dim=-1)
        lin_momentum = self.part_mass[part] * vel
        R = rotation[:-1].float()
        moi_part = self.part_moi[part].unsqueeze(0).expand_as(R)
        part_moi = torch.bmm(torch.bmm(R, moi_part),
                             R.transpose(-2, -1))
        w = roma.rotmat_to_rotvec(rvel).reshape(-1, 3).unsqueeze(-1).float()
        Li = torch.bmm(part_moi, w)
        Li = Li.reshape(vel.shape[0], 3)
        return Li, transfer, lin_momentum

    def momenta_components_from_mesh(self, body_meshes, pose, grot):
        pred_centroids = self.centroid_from_mesh(body_meshes)
        pred_rotation = self.part_rotation(pose, grot)
        return self.momenta_components_helper(pred_centroids, pred_rotation)

    def momenta_components_helper(self, pred_centroids, pred_rotation, return_sum=True):
        vel = pred_centroids[1:] - pred_centroids[:-1]
        rvel = torch.bmm(pred_rotation[1:].reshape(-1, 3, 3),
                         pred_rotation[:-1].transpose(-1, -2).reshape(-1, 3, 3))
        transfer = self.part_mass.unsqueeze(-1).unsqueeze(0) * \
            torch.cross(pred_centroids[:-1], vel, dim=-1)
        momentum = self.part_mass.unsqueeze(-1).unsqueeze(0) * vel
        moi_seq = self.part_moi.unsqueeze(0).repeat(vel.shape[0], 1, 1, 1)\
            .reshape(-1, 3, 3)
        R = pred_rotation[:-1].reshape(-1, 3, 3).float()
        part_moi = torch.bmm(torch.bmm(R, moi_seq), 
                             R.transpose(-2, -1))
        w = roma.rotmat_to_rotvec(rvel).reshape(-1, 3).unsqueeze(-1).float()
        Li = torch.bmm(part_moi, w)
        Li = Li.reshape(vel.shape[0], -1, 3)
        if return_sum:
            local_total = torch.sum(Li, axis=1)
            transfer_total = torch.sum(transfer, axis=1)
            momentum = torch.sum(momentum, axis=1)
            return local_total + transfer_total, momentum
        return Li + transfer, momentum

    def momenta_components(self, pose, grot, gtrans=None, return_sum=True):
        pred_centroids = self.part_centroid(pose, grot, gtrans)
        pred_rotation = self.part_rotation(pose, grot)
        return self.momenta_components_helper(pred_centroids, pred_rotation, return_sum)

    def angular_momentum(self, pose, grot, gtrans=None, return_sum=True):
        pred_centroids = self.part_centroid(pose, grot, gtrans)
        pred_rotation = self.part_rotation(pose, grot)
        vel = pred_centroids[1:] - pred_centroids[:-1]
        rvel = torch.bmm(pred_rotation[1:].reshape(-1, 3, 3),
                         pred_rotation[:-1].transpose(-1, -2).reshape(-1, 3, 3))
        transfer = self.part_mass.unsqueeze(-1).unsqueeze(0) * \
            torch.cross(pred_centroids[:-1], vel, dim=-1)
        moi_seq = self.part_moi.unsqueeze(0).repeat(vel.shape[0], 1, 1, 1)\
            .reshape(-1, 3, 3)
        R = pred_rotation[:-1].reshape(-1, 3, 3).float()
        part_moi = torch.bmm(torch.bmm(R, moi_seq), 
                             R.transpose(-2, -1))
        w = roma.rotmat_to_rotvec(rvel).reshape(-1, 3).unsqueeze(-1).float()
        Li = torch.bmm(part_moi, w)
        Li = Li.reshape(vel.shape[0], -1, 3)
        local_total = torch.sum(Li, axis=1)
        transfer_total = torch.sum(transfer, axis=1)
        if return_sum:
            return local_total + transfer_total
        return Li + transfer

    def hf_energy(self, pose_body, root):
        am = self.angular_momentum(pose_body, root)
        fft = torch.zeros_like(am)
        for i in range(3):
            fft[:, i] = dct.dct(am[:, i])
        hf = torch.sum(fft[40:] ** 2) / torch.sum(fft ** 2)
        return hf

    def hf_components_momenta(self, pose_body, grot, gtrans, shape=None):
        if shape is None:
            self.setup_zero_shape()
        else:
            self.setup_shape(shape)
        am = self.angular_momentum(pose_body, grot, gtrans)
        lm = self.linear_momentum(pose_body, grot, gtrans)
        return self.hf_components(am, lm)

    def hf_components(self, am, lm):
        am_fft = self.calculate_fft(am)
        lm_fft = self.calculate_fft(lm)
        T = int(0.4 * am.shape[0])
        am_hf = torch.abs(am_fft[T:]).mean()
        lm_hf = torch.abs(lm_fft[T:]).mean()
        return am_hf, lm_hf

    def calculate_fft(self, time_signal):
        fft = torch.zeros_like(time_signal)
        for i in range(3):
            fft[:, i] = dct.dct(time_signal[:, i])
        return fft
    
    def calculate_dft(self, time_signal):
        fft = torch.zeros_like(time_signal)
        for i in range(3):
            fft[:, i] = torch.fft.fft(time_signal[:, i]).real
        return fft
