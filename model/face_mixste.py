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
from functools import partial

from torch.utils.tensorboard import SummaryWriter
import time
from datetime import datetime, timezone
from model.eyenet import EyeNet
from utils.loss import BaseLossWithValidity, L1Loss, AngularLoss, EuclideanLoss
from utils.util_functions import pitchyaw_to_vector
from utils.util_functions import to_screen_coordinates, apply_rotation, apply_transformation, get_intersect_with_zero
from utils.util_functions import to_screen_coordinates_with_seq
from einops import rearrange

logger = logging.getLogger(__name__)
config = DefaultConfig()

hp = {
    'eye_net_rnn_num_features': 128,
    'eye_net_rnn_num_cells': 1,
    'batch_size': 16,
    'epoch_num': 10,
    # 'sequence_length': 30,
    'saving_dir': './saving',

    'face_net_features': 500,
    'embed_dim_ratio' : None,
    'mlp_ratio': 2,
    'sequence_length': 30,

}

l1_loss = L1Loss()
angular_loss = AngularLoss()
euclidean_loss = EuclideanLoss()

class FaceMixSTE(nn.Module):
    def __init__(self):
        super(FaceMixSTE, self).__init__()
        # self.norm_layer = partial(nn.LayerNorm, eps=1e-6)
        
        self.cnn_layer = ResNet(block=BasicBlock,
                                layers=[2, 2, 2, 2],
                                num_classes=hp['face_net_features'],
                                norm_layer=nn.InstanceNorm2d)
        
        self.fc_common = nn.Sequential(
            # Use Head Pose
            nn.Linear(hp['face_net_features'] + 2, hp['face_net_features']),
            nn.SELU(inplace=True),
            nn.Linear(hp['face_net_features'], hp['face_net_features'])
        )

        #TODO: 在forward的时候确定维度
        self.spatial_pos_embedding = nn.Parameter(torch.zeros(1, hp['face_net_features']))
        self.temporal_pos_embedding = nn.Parameter(torch.zeros(1, hp['sequence_length']))
        
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hp['face_net_features'],
                nhead=10,
                dim_feedforward=hp['face_net_features']*hp['mlp_ratio'],
                dropout=0.1,
                activation='gelu',
                batch_first=True
            ),
            num_layers=1
        )

        self.transformer_decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hp['sequence_length'],
                nhead=10,
                dim_feedforward=hp['sequence_length']*hp['mlp_ratio'],
                dropout=0.1,
                activation='gelu',
                batch_first=True
            ),
            num_layers=1
        )

        self.fc_to_gaze_left = nn.Sequential(
            nn.Linear(hp['face_net_features'], hp['face_net_features']),
            nn.SELU(inplace=True),
            nn.Linear(hp['face_net_features'], out_features=2, bias=False),
            nn.Tanh()
        )

        self.fc_to_gaze_right = nn.Sequential(
            nn.Linear(hp['face_net_features'], hp['face_net_features']),
            nn.SELU(inplace=True),
            nn.Linear(hp['face_net_features'], out_features=2, bias=False),
            nn.Tanh()
        )

        nn.init.zeros_(self.fc_to_gaze_left[-2].weight)
        nn.init.zeros_(self.fc_to_gaze_right[-2].weight)

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
        # seq_len = hp['sequence_length']
        # print("---------------------------------")
        # print(full_input_dict['face_patch'].shape)
        # batch_size, seq_len, _, _, _ = full_input_dict['face_patch'].shape
        batch_size = full_input_dict['face_patch'].shape[0]
        seq_len = full_input_dict['face_patch'].shape[1]
        C = full_input_dict['face_patch'].shape[2]
        H = full_input_dict['face_patch'].shape[3]
        W = full_input_dict['face_patch'].shape[4]
        left_previous_states = None
        right_previous_states = None
        intermediate_dicts = []

        for side in ['left', 'right']:
            full_input_dict[side + '_PoG_cm_tobii'] = torch.mul(
                full_input_dict[side + '_PoG_tobii'],
                0.1 * full_input_dict['millimeters_per_pixel'],
            ).detach()
            full_input_dict[side + '_PoG_cm_tobii_validity'] = \
                full_input_dict[side + '_PoG_tobii_validity']
        
        input_images_seq = full_input_dict['face_patch']
        input_images_seq = input_images_seq.view(batch_size * seq_len, C, H, W)
        output_of_cnn = self.cnn_layer(input_images_seq)
        output_of_cnn = output_of_cnn.view(batch_size, seq_len, hp['face_net_features'])
        output_of_cnn_with_head_pos = torch.cat([output_of_cnn, full_input_dict['face_h']], dim=2)
        initial_features_seq = self.fc_common(output_of_cnn_with_head_pos)

        # Transformer_1
        transformer_encoder_input_1 = initial_features_seq
        transformer_encoder_input_1 += self.spatial_pos_embedding
        transformer_encoder_output_1 = self.transformer_encoder(transformer_encoder_input_1)

        transformer_decoder_input_1 = rearrange(transformer_encoder_output_1, 'b t f -> b f t')
        transformer_decoder_input_1 += self.temporal_pos_embedding
        transformer_decoder_output_1 = self.transformer_decoder(transformer_decoder_input_1)
        
        # Transformer_2
        transformer_encoder_input_2 = rearrange(transformer_decoder_output_1, 'b f t -> b t f')
        transformer_encoder_input_2 += self.spatial_pos_embedding
        transformer_encoder_output_2 = self.transformer_encoder(transformer_encoder_input_2)

        transformer_decoder_input_2 = rearrange(transformer_encoder_output_2, 'b t f -> b f t')
        transformer_decoder_input_2 += self.temporal_pos_embedding
        transformer_decoder_output_2 = self.transformer_decoder(transformer_decoder_input_2)

        transformer_output = rearrange(transformer_decoder_output_2, 'b f t -> b t f')


        # to gaze
        left_g_initial = self.fc_to_gaze_left(transformer_output)
        right_g_initial = self.fc_to_gaze_right(transformer_output)

        full_intermediate_dict = {}
        full_intermediate_dict['left_g_initial'] = left_g_initial
        full_intermediate_dict['right_g_initial'] = right_g_initial
        self.from_g_to_PoG_history_with_seq(full_input_dict, full_intermediate_dict, input_suffix='initial', output_suffix='initial')

        output_dict = {}
        for k in full_intermediate_dict.keys():
            output_dict[k] = full_intermediate_dict[k]

        for side in ['left', 'right']:



            input_key = side + '_g_tobii'
            interm_key = side + '_g_initial'
            output_key = side + '_g_initial'
            output_dict['loss_ang_' + output_key] = angular_loss(full_intermediate_dict[interm_key], input_key,
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

        output_dict['full_loss'] = full_loss
        
        return output_dict
        