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
from utils.loss import BaseLossWithValidity, L1Loss, AngularLoss, EuclideanLoss
from utils.util_functions import pitchyaw_to_vector
from utils.util_functions import to_screen_coordinates, apply_rotation, apply_transformation, get_intersect_with_zero
from utils.util_functions import to_screen_coordinates_with_seq

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

l1_loss = L1Loss()
angular_loss = AngularLoss()
euclidean_loss = EuclideanLoss()

class EyeNetFrame(nn.Module):
    def __init__(self):
        super(EyeNetFrame, self).__init__()
        self.eye_net = EyeNet()

    def from_g_to_PoG_history(self, sub_input_dict, sub_output_dict, input_suffix, output_suffix):
        for side in ('left', 'right'):
            origin = (sub_output_dict[side + '_o']
                        if side + '_o' in sub_output_dict else sub_input_dict[side + '_o'])
            direction = sub_output_dict[side + '_g_' + input_suffix]
            rotation = (sub_output_dict[side + '_R']
                        if side + '_R' in sub_output_dict else sub_input_dict[side + '_R'])
            PoG_mm, PoG_px = to_screen_coordinates(origin, direction, rotation, sub_input_dict)
            sub_output_dict[side + '_PoG_cm_' + output_suffix] = 0.1 * PoG_mm
            sub_output_dict[side + '_PoG_px_' + output_suffix] = PoG_px

        # Step 1b) Calculate average PoG
        sub_output_dict['PoG_px_' + output_suffix] = torch.mean(torch.stack([
            sub_output_dict['left_PoG_px_' + output_suffix],
            sub_output_dict['right_PoG_px_' + output_suffix],
        ], axis=-1), axis=-1)
        sub_output_dict['PoG_cm_' + output_suffix] = torch.mean(torch.stack([
            sub_output_dict['left_PoG_cm_' + output_suffix],
            sub_output_dict['right_PoG_cm_' + output_suffix],
        ], axis=-1), axis=-1)
        sub_output_dict['PoG_mm_' + output_suffix] = \
            10.0 * sub_output_dict['PoG_cm_' + output_suffix]

    def from_g_to_PoG_history_with_seq(self, input_dict, output_dict, input_suffix, output_suffix):
        for side in ('left', 'right'):
            origin = (output_dict[side + '_o']
                        if side + '_o' in output_dict else input_dict[side + '_o'])
            direction = output_dict[side + '_g_' + input_suffix]
            rotation = (output_dict[side + '_R']
                        if side + '_R' in output_dict else input_dict[side + '_R'])
            PoG_mm, PoG_px = to_screen_coordinates_with_seq(origin, direction, rotation, input_dict)
            output_dict[side + '_PoG_cm_' + output_suffix] = 0.1 * PoG_mm
            output_dict[side + '_PoG_px_' + output_suffix] = PoG_px


    def forward(self, full_input_dict):
        seq_len = hp['sequence_length']
        left_previous_states = None
        right_previous_states = None
        intermediate_dicts = []

        # set PoG ground truth
        for side in ['left', 'right']:
            full_input_dict[side + '_PoG_cm_tobii'] = torch.mul(
                full_input_dict[side + '_PoG_tobii'],
                0.1 * full_input_dict['millimeters_per_pixel'],
            ).detach()
            full_input_dict[side + '_PoG_cm_tobii_validity'] = \
                full_input_dict[side + '_PoG_tobii_validity']

        for t in range(seq_len):
            # 取出时间步t字典中所有的东西
            sub_input_dict = {}
            for k, v in full_input_dict.items():
                if isinstance(v, torch.Tensor):
                    sub_v = v[:, t, :] if v.ndim > 2 else v[:, t]
                    sub_input_dict[k] = sub_v
            sub_output_dict = {}

            left_g_initial, left_pupil_size, left_output_states = self.eye_net(sub_input_dict, side='left',
                                                                               initial_states=left_previous_states)
            left_previous_states = left_output_states
            right_g_initial, right_pupil_size, right_output_states = self.eye_net(sub_input_dict, side='right',
                                                                                  initial_states=right_previous_states)
            right_previous_states = right_output_states

            sub_output_dict['left_g_initial'] = left_g_initial
            sub_output_dict['left_pupil_size'] = left_pupil_size
            sub_output_dict['right_g_initial'] = right_g_initial
            sub_output_dict['right_pupil_size'] = right_pupil_size

            # self.from_g_to_PoG_history(sub_input_dict=sub_input_dict, sub_output_dict=sub_output_dict, input_suffix='initial', output_suffix='initial')
            intermediate_dicts.append(sub_output_dict)

        full_intermediate_dict = {}
        for k in intermediate_dicts[0].keys():
            sample = intermediate_dicts[0][k]
            if not isinstance(sample, torch.Tensor):
                continue
            full_intermediate_dict[k] = torch.stack([
                intermediate_dicts[i][k] for i in range(hp['sequence_length'])
            ], axis=1)
        self.from_g_to_PoG_history_with_seq(full_input_dict, full_intermediate_dict, input_suffix='initial', output_suffix='initial')

        output_dict = {}
        for k in full_intermediate_dict.keys():
            output_dict[k] = full_intermediate_dict[k]
        
        ########
        for side in ['left', 'right']:



            input_key = side + '_g_tobii'
            interm_key = side + '_g_initial'
            output_key = side + '_g_initial'
            output_dict['loss_ang_' + output_key] = angular_loss(full_intermediate_dict[interm_key], input_key,
                                                                 full_input_dict)

            input_key = side + '_p'
            interm_key = side + '_pupil_size'
            output_dict['loss_l1_' + interm_key] = l1_loss(full_intermediate_dict[interm_key], input_key,
                                                           full_input_dict)
            
            input_key = side + '_PoG_cm_tobii'
            interm_key = side + '_PoG_cm_initial'
            output_key = side + '_PoG_cm_initial'
            output_dict['euc_loss_' + output_key] = euclidean_loss(full_intermediate_dict[interm_key], input_key, full_input_dict)

            input_key = side + '_PoG_tobii'
            interm_key = side + '_PoG_px_initial'
            output_dict['euc_loss_' + interm_key] = euclidean_loss(full_intermediate_dict[interm_key], input_key, full_input_dict)
            
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        full_loss = torch.zeros(()).to(device)
        full_loss += (output_dict['loss_ang_left_g_initial'] + output_dict['loss_ang_right_g_initial'])
        full_loss += (output_dict['loss_l1_left_pupil_size'] + output_dict['loss_l1_right_pupil_size'])

        output_dict['full_loss'] = full_loss
        
        return output_dict