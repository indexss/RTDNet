import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock, ResNet
import logging
import numpy as np
from collections import OrderedDict
import functools
import wandb
import random
import os
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Subset
from datasources import EVESequences_train, EVESequences_test, EVESequences_val
from core import DefaultConfig
from torch.optim.lr_scheduler import ExponentialLR

from torch.utils.tensorboard import SummaryWriter
import time
from datetime import datetime, timezone
from model.eyenet import EyeNet

logger = logging.getLogger(__name__)
config = DefaultConfig()

hp = {
    'eye_net_rnn_num_features': 128,
    'eye_net_rnn_num_cells': 1,
    'batch_size': 16,
    'epoch_num': 10,
    'sequence_length': 30,
    'saving_dir': './saving',
}

def pitchyaw_to_vector(a):
    if a.shape[1] == 2:
        sin = torch.sin(a)
        cos = torch.cos(a)
        return torch.stack([cos[:, 0] * sin[:, 1], sin[:, 0], cos[:, 0] * cos[:, 1]], dim=1)
    elif a.shape[1] == 3:
        return F.normalize(a)
    else:
        raise ValueError('Do not know how to convert tensor of size %s', a.shape)

def pitchyaw_to_vector_with_seq(a):
    # a: batch_size x seq_len x 2
    if a.shape[-1] == 2:
        sin = torch.sin(a)
        cos = torch.cos(a)
        x = cos[..., 0] * sin[..., 1]
        y = sin[..., 0]
        z = cos[..., 0] * cos[..., 1]
        return torch.stack([x, y, z], dim=-1)
    elif a.shape[-1] == 3:
        return F.normalize(a, dim=-1)
    else:
        raise ValueError('Do not know how to convert tensor of size %s', a.shape)


def to_screen_coordinates(origin, direction, rotation, reference_dict):
    direction = pitchyaw_to_vector(direction)

    # Negate gaze vector back (to camera perspective)
    direction = -direction

    # De-rotate gaze vector
    inv_rotation = torch.transpose(rotation, 1, 2)
    direction = direction.reshape(-1, 3, 1)
    direction = torch.matmul(inv_rotation, direction)

    # Transform values
    inv_camera_transformation = reference_dict['inv_camera_transformation']
    direction = apply_rotation(inv_camera_transformation, direction)
    # direction: batch_szie, seq_len, 3
    origin = apply_transformation(inv_camera_transformation, origin)
    # origin: batch_size, seq_len, 3

    # Intersect with z = 0
    recovered_target_2D = get_intersect_with_zero(origin, direction)
    PoG_mm = recovered_target_2D

    # Convert back from mm to pixels
    ppm_w = reference_dict['pixels_per_millimeter'][:, 0]
    ppm_h = reference_dict['pixels_per_millimeter'][:, 1]
    PoG_px = torch.stack([
        #               16x2                      16x1
        torch.clamp(recovered_target_2D[:, 0] * ppm_w,
                    0.0, float(config.actual_screen_size[0])),
        torch.clamp(recovered_target_2D[:, 1] * ppm_h,
                    0.0, float(config.actual_screen_size[1]))
    ], axis=-1)
    # print('---------------------test PoG_mm shape:', PoG_mm.shape)
    # print('---------------------test PoG_px shape:', PoG_px.shape)
    return PoG_mm, PoG_px

def to_screen_coordinates_with_seq(origin, direction, rotation, reference_dict):
    # origin: batch_size x seq_len x 3
    # direction: batch_size x seq_len x 2
    # rotation: batch_size x seq_len x 3 x 3
    # reference_dict: dict, each item: batch_size x seq_len x (item)
    batch_size = direction.shape[0]
    seq_len = direction.shape[1]
    direction = pitchyaw_to_vector_with_seq(direction)

    # Negate gaze vector back (to camera perspective)
    direction = -direction

    # De-rotate gaze vector
    inv_rotation = torch.transpose(rotation, 2, 3)
    direction = direction.reshape(batch_size, seq_len, 3, 1)
    direction = torch.matmul(inv_rotation, direction)

    # Transform values
    inv_camera_transformation = reference_dict['inv_camera_transformation']
    direction = apply_rotation_with_seq(inv_camera_transformation, direction)
    origin = apply_transformation_with_seq(inv_camera_transformation, origin)

    # print("origin shape13:", origin.shape)
    # print("direction shape13:", direction.shape)

    # Intersect with z = 0
    recovered_target_2D = get_intersect_with_zero_with_seq(origin, direction)
    PoG_mm = recovered_target_2D # batch_size x seq_len x 2

    # Convert back from mm to pixels
    ppm_w = reference_dict['pixels_per_millimeter'][:, :, 0]
    ppm_h = reference_dict['pixels_per_millimeter'][:, :, 1]
    PoG_px = torch.stack([
        torch.clamp(recovered_target_2D[..., 0] * ppm_w,
                    0.0, float(config.actual_screen_size[0])),
        torch.clamp(recovered_target_2D[..., 1] * ppm_h,
                    0.0, float(config.actual_screen_size[1]))
    ], axis=-1)
    # print('---------------------test PoG_mm shape:', PoG_mm.shape)
    # print('---------------------test PoG_px shape:', PoG_px.shape)
    return PoG_mm, PoG_px




def apply_rotation(T, vec):
    if vec.shape[1] == 2:
        vec = pitchyaw_to_vector(vec)
    vec = vec.reshape(-1, 3, 1)
    R = T[:, :3, :3]
    return torch.matmul(R, vec).reshape(-1, 3)

def apply_rotation_with_seq(T, vec):
    # vec: batch_size x seq_len x 3
    batch_size = vec.shape[0]
    seq_len = vec.shape[1]
    if vec.shape[-1] == 2:
        vec = pitchyaw_to_vector_with_seq(vec)
    vec = vec.reshape(batch_size, seq_len, 3, 1)
    R = T[:, :, :3, :3]
    return torch.matmul(R, vec).reshape(batch_size, seq_len, 3)

nn_plane_normal = None
nn_plane_other = None

def get_intersect_with_zero(o, g):
    """Intersects a given gaze ray (origin o and direction g) with z = 0."""
    global nn_plane_normal, nn_plane_other
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if nn_plane_normal is None:
        nn_plane_normal = torch.tensor([0, 0, 1], dtype=torch.float32, device=device).view(1, 3, 1)
        nn_plane_other = torch.tensor([1, 0, 0], dtype=torch.float32, device=device).view(1, 3, 1)

    # Define plane to intersect with
    n = nn_plane_normal
    a = nn_plane_other
    g = g.view(-1, 3, 1)
    o = o.view(-1, 3, 1)
    numer = torch.sum(torch.mul(a - o, n), dim=1)

    # Intersect with plane using provided 3D origin
    denom = torch.sum(torch.mul(g, n), dim=1) + 1e-7
    t = torch.div(numer, denom).view(-1, 1, 1)
    return (o + torch.mul(t, g))[:, :2, 0]

nn_plane_normal = None
nn_plane_other = None
def get_intersect_with_zero_with_seq(o, g):
    # print("g shape1:", g.shape)
    # print("o shape1:", o.shape)
    # o: batch_size x seq_len x 3
    # g: batch_size x seq_len x 3
    batch_size = o.shape[0]
    seq_len = o.shape[1]
    global nn_plane_normal, nn_plane_other
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 确保全局变量已初始化
    if nn_plane_normal is None:
        nn_plane_normal = torch.tensor([0, 0, 1], dtype=torch.float32, device=device).view(1, 1, 3, 1)
        nn_plane_other = torch.tensor([1, 0, 0], dtype=torch.float32, device=device).view(1, 1, 3, 1)

    # 调整输入的形状以匹配期望的操作
    g = g.unsqueeze(-1)  # batch_size x seq_len x 3 x 1
    o = o.unsqueeze(-1)  # batch_size x seq_len x 3 x 1
    # print("g shape2:", g.shape)
    # print("o shape2:", o.shape)

    # Define plane to intersect with
    n = nn_plane_normal
    a = nn_plane_other

    
    # print("n shape:", n.shape)
    # print("a shape:", a.shape)
    numer = torch.sum((a - o) * n, dim=2)  
    denom = torch.sum(g * n, dim=2) + 1e-7  # batch_size x seq_len x 1
    # print("numer shape:", numer.shape)
    # print("denom shape:", denom.shape)
    t = numer / denom  # batch_size x seq_len x 1
    t = t.unsqueeze(-1)  # batch_size x seq_len x 1 x 1 

    # print("o shape5:", o.shape)
    # print("t shape5:", t.shape)
    # print("g shape5:", g.shape)

    intersection = o + t * g  # batch_size x seq_len x 3 x 1
    return intersection.squeeze(-1)[..., :2]  # batch_size x seq_len x 2
    

def apply_transformation(T, vec):
    batch_size = vec.shape[0]
    seq_len = vec.shape[1]
    if vec.shape[1] == 2:
        vec = pitchyaw_to_vector(vec)
    # print("vec shape1:", vec.shape)
    vec = vec.reshape(batch_size, seq_len, 3, 1)
    # print("vec shape2:", vec.shape)
    h_vec = F.pad(vec, pad=(0, 0, 0, 1), value=1.0)
    # print("T shape:", T.shape)
    # print("h_vec shape:", h_vec.shape)
    return torch.matmul(T, h_vec)[:, :3, 0]

def apply_transformation_with_seq(T, vec):
    # T: batch_size x seq_len x 4 x 4
    # vec: batch_size x seq_len x 3
    batch_size = vec.shape[0]
    seq_len = vec.shape[1]
    if vec.shape[-1] == 2:
        vec = pitchyaw_to_vector_with_seq(vec)
    vec = vec.reshape(batch_size, seq_len, 3, 1)
    h_vec = F.pad(vec, pad=(0, 0, 0, 1), value=1.0)
    # batch_size, seq_len, 3
    return torch.matmul(T, h_vec)[:, :, :3, 0]

def sliding_window(tensor, window_size, step_size):
    # tensor: batch_size x seq_len x C x H x W
    batch_size = tensor.shape[0]
    seq_len = tensor.shape[1]
    C = tensor.shape[2]
    H = tensor.shape[3]
    W = tensor.shape[4]
    window_size = min(window_size, seq_len)
    step_size = min(step_size, window_size)
    windows = []
    for i in range(0, seq_len - window_size + 1, step_size):
        window = tensor[:, i:i + window_size, :, :, :]
        windows.append(window)
    return windows

def set_seed(seed=0):
    random.seed(seed) 
    np.random.seed(seed)  
    os.environ['PYTHONHASHSEED'] = str(seed)  
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)

def get_all_vectors_from_points(a):
    batch_size = a.shape[0]
    seq_len = a.shape[1]
    all_vectors = a[:, 1:] - a[:, :-1]
    first_vector = a[:, 0] - a[:, -1]
    all_vectors = torch.cat((first_vector.unsqueeze(1), all_vectors), dim=1)
    return all_vectors

def get_all_distance(gt, pd):
    # gt: batch_size x seq_len x 2
    # pd: batch_size x seq_len x 2
    distances = torch.norm(gt - pd, dim=2)
    return distances
    

def cal_angular_through_vectors(a1, a2):
    epsilon = 1e-10
    _to_degrees = 180. / np.pi
    batch_size = a1.shape[0]
    seq_len = a1.shape[1]
    vector_size = a1.shape[2]
    ang = torch.sum(a1 * a2, dim=2) / (torch.norm(a1, dim=2) * torch.norm(a2, dim=2) + epsilon)
    last_ang = torch.sum(a2[:, -1] * a1[:, 0], dim=1) / (torch.norm(a2[:, -1], dim=1) * torch.norm(a1[:, 0], dim=1) + epsilon)
    ang = torch.cat((ang, last_ang.unsqueeze(1)), dim=1)
    ang = F.hardtanh_(ang, min_val=-0.5 + 1e-8, max_val=0.5 - 1e-8)
    return torch.acos(ang) * _to_degrees

def subset_test(model, test_data):
    model.eval()
    test_dataloader = test_data['eve_val']['dataloader']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lg = []
    rg = []
    # lp = []
    # rp = []
    lgcm = []
    lgpx = []
    rgcm = []
    rgpx = []
    with torch.no_grad():
        for i, test_data in enumerate(test_dataloader):
            for k, v in test_data.items():
                if isinstance(v, torch.Tensor):
                    test_data[k] = v.to(device)
            test_output_dict = model(test_data)
            # print(test_output_dict)
            left_gaze_diff = test_output_dict['loss_ang_left_g_initial'].item()
            right_gaze_diff = test_output_dict['loss_ang_right_g_initial'].item()
            # left_pupil_diff = test_output_dict['loss_l1_left_pupil_size'].item()
            # right_pupil_diff = test_output_dict['loss_l1_right_pupil_size'].item()
            left_PoG_diff_cm = test_output_dict['euc_loss_left_PoG_cm_initial'].item()
            right_PoG_diff_cm = test_output_dict['euc_loss_right_PoG_cm_initial'].item()
            left_PoG_diff_px = test_output_dict['euc_loss_left_PoG_px_initial'].item()
            right_PoG_diff_px = test_output_dict['euc_loss_right_PoG_px_initial'].item()

            lg.append(left_gaze_diff)
            rg.append(right_gaze_diff)
            # lp.append(left_pupil_diff)
            # rp.append(right_pupil_diff)
            lgcm.append(left_PoG_diff_cm)
            lgpx.append(left_PoG_diff_px)
            rgcm.append(right_PoG_diff_cm)
            rgpx.append(right_PoG_diff_px)
    # print(test_output_dict)
    lg_mean = np.mean(lg).item()
    rg_mean = np.mean(rg).item()
    # lp_mean = np.mean(lp).item()
    # rp_mean = np.mean(rp).item()
    lgcm_mean = np.mean(lgcm).item()
    lgpx_mean = np.mean(lgpx).item()
    rgcm_mean = np.mean(rgcm).item()
    rgpx_mean = np.mean(rgpx).item()

    print({
        'test_lg_mean': lg_mean,
        'test_rg_mean': rg_mean,
        # 'test_lp_mean': lp_mean,
        # 'test_rp_mean': rp_mean,
        'test_lgcm_mean': lgcm_mean,
        'test_lgpx_mean': lgpx_mean,
        'test_rgcm_mean': rgcm_mean,
        'test_rgpx_mean': rgpx_mean
    })

    return {
        'test_lg_mean': lg_mean,
        'test_rg_mean': rg_mean,
        # 'test_lp_mean': lp_mean,
        # 'test_rp_mean': rp_mean,
        'test_lgcm_mean': lgcm_mean,
        'test_lgpx_mean': lgpx_mean,
        'test_rgcm_mean': rgcm_mean,
        'test_rgpx_mean': rgpx_mean
    }


def mask_tensor(input_tensor, mask_indices, mask_type='noise'):
    '''
    将输入张量的指定索引位置的值mask掉
    输入:input_tensor(batch_size, sequence_length, channel, height, width)
    mask_indices: 要mask的索引位置,列表形式,如[4,9,14,19,24,29]
    mask_type: mask的类型:'black'表示mask为黑色,'white'表示mask为白色,'noise'表示mask为随机噪声
    输出:mask后的张量
    注意:为了节省显存没使用clone复制tensor(能否节省存疑),因此会改变原始tensor的值
    '''

    # 确保输入张量的维度正确
    assert input_tensor.dim() == 5 
    
    if mask_type == 'black':
        for idx in mask_indices:
            input_tensor[:, idx, :, :, :] = 1

    elif mask_type == 'white':
        for idx in mask_indices:
            input_tensor[:, idx, :, :, :] = -1

    elif mask_type == 'noise':
        batch_size, sequence_length, channel, height, width = input_tensor.shape
        noise = torch.rand(batch_size, channel, height, width) * 2 - 1
        for idx in mask_indices:
            input_tensor[:, idx, :, :, :] = noise
    
    return input_tensor
