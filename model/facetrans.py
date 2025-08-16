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
from utils.util_functions import to_screen_coordinates_with_seq, get_all_vectors_from_points, cal_angular_through_vectors, sliding_window, get_all_distance
from typing import Any, Callable, List, Optional, Type, Union
from torch import Tensor
from datasources.EyediapTest import EyediapLoaderTest

logger = logging.getLogger(__name__)
config = DefaultConfig()

hp = {
    'eye_net_rnn_num_features': 128,
    'eye_net_rnn_num_cells': 1,
    'batch_size': 16,
    'epoch_num': 10,
    'sequence_length': 30,
    'saving_dir': './saving',

    'face_net_features': 512,
    'embed_dim_ratio' : None,
    'mlp_ratio': 2,

}

l1_loss = L1Loss()
angular_loss = AngularLoss()
euclidean_loss = EuclideanLoss()

class Facetrans(nn.Module):
    def __init__(self):
        super(Facetrans, self).__init__()
        # self.norm_layer = partial(nn.LayerNorm, eps=1e-6)
        
        self.cnn_layer = ResNet(block=BasicBlock,
                                layers=[2, 2, 2, 2],
                                num_classes=hp['face_net_features'],
                                norm_layer=nn.InstanceNorm2d)

        # self.cnn_layer = MyResNet18()
        
        self.fc_common = nn.Sequential(
            # Use Head Pose
            nn.Linear(hp['face_net_features'] + 2, hp['face_net_features']),
            nn.SELU(inplace=True),
            nn.Linear(hp['face_net_features'], hp['face_net_features'])
        )

        #TODO: 在forward的时候确定维度
        self.spatial_pos_embedding = nn.Parameter(torch.randn(1, hp['face_net_features']))
        
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hp['face_net_features'],
                nhead=8,
                dim_feedforward=hp['face_net_features']*hp['mlp_ratio'],
                dropout=0.1,
                activation='gelu',
                batch_first=True
            ),
            num_layers=1
        )

        # self.fc_to_gaze_left = nn.Sequential(
        #     nn.Linear(hp['face_net_features'], hp['face_net_features']),
        #     nn.SELU(inplace=True),
        #     nn.Linear(hp['face_net_features'], out_features=2, bias=False),
        #     nn.Tanh()
        # )

        # self.fc_to_gaze_right = nn.Sequential(
        #     nn.Linear(hp['face_net_features'], hp['face_net_features']),
        #     nn.SELU(inplace=True),
        #     nn.Linear(hp['face_net_features'], out_features=2, bias=False),
        #     nn.Tanh()
        # )

        # nn.init.zeros_(self.fc_to_gaze_left[-2].weight)
        # nn.init.zeros_(self.fc_to_gaze_right[-2].weight)

        self.fc_to_gaze_face = nn.Sequential(
            nn.Linear(hp['face_net_features'], hp['face_net_features']),
            nn.SELU(inplace=True),
            nn.Linear(hp['face_net_features'], out_features=2, bias=False),
            nn.Tanh()
        )
        nn.init.zeros_(self.fc_to_gaze_face[-2].weight)

    def from_g_to_PoG_history_with_seq(self, input_dict, output_dict, input_suffix, output_suffix):
        side = 'face'
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

        ## for side in ['face']:
        #     full_input_dict[side + '_PoG_cm_tobii'] = torch.mul(
        #         full_input_dict[side + '_PoG_tobii'],
        #         0.1 * full_input_dict['millimeters_per_pixel'],
        #     ).detach()
        #     full_input_dict[side + '_PoG_cm_tobii_validity'] = \
        ##         full_input_dict[side + '_PoG_tobii_validity']
        
        input_images_seq = full_input_dict['face_patch']
        input_images_seq = input_images_seq.view(batch_size * seq_len, C, H, W)
        # output_of_cnn, feature1, feature2, feature3, feature4 = self.cnn_layer(input_images_seq)
        output_of_cnn = self.cnn_layer(input_images_seq)
        # print(output_of_cnn.shape)
        # print(feature1.shape)
        # print(feature2.shape)
        # print(feature3.shape)
        # print(feature4.shape)
        output_of_cnn = output_of_cnn.view(batch_size, seq_len, hp['face_net_features'])
        output_of_cnn_with_head_pos = torch.cat([output_of_cnn, full_input_dict['face_h']], dim=2)
        initial_features_seq = self.fc_common(output_of_cnn_with_head_pos)

        # Add spatial positional encoding
        initial_features_seq += self.spatial_pos_embedding

        # Transformer
        transformer_output = self.transformer_encoder(initial_features_seq)

        # to gaze
        # left_g_initial = self.fc_to_gaze_left(transformer_output)
        # right_g_initial = self.fc_to_gaze_right(transformer_output)
        face_g_initial = self.fc_to_gaze_face(transformer_output)

        full_intermediate_dict = {}
        # full_intermediate_dict['left_g_initial'] = left_g_initial
        # full_intermediate_dict['right_g_initial'] = right_g_initial
        full_intermediate_dict['face_g_initial'] = face_g_initial
        full_intermediate_dict['face_patch'] = full_input_dict['face_patch']
        ## self.from_g_to_PoG_history_with_seq(full_input_dict, full_intermediate_dict, input_suffix='initial', output_suffix='initial')

        output_dict = {}
        for k in full_intermediate_dict.keys():
            output_dict[k] = full_intermediate_dict[k]

        for side in ['face']:

            input_key = side + '_g_tobii'
            interm_key = side + '_g_initial'
            output_key = side + '_g_initial'
            output_dict['loss_ang_' + output_key] = angular_loss(full_intermediate_dict[interm_key], input_key,
                                                                 full_input_dict)
            
            ## input_key = side + '_PoG_cm_tobii'
            # interm_key = side + '_PoG_cm_initial'
            # output_key = side + '_PoG_cm_initial'
            # output_dict['euc_loss_' + output_key] = euclidean_loss(full_intermediate_dict[interm_key], input_key, full_input_dict)

            # input_key = side + '_PoG_tobii'
            # interm_key = side + '_PoG_px_initial'
            # output_dict['euc_loss_' + interm_key] = euclidean_loss(full_intermediate_dict[interm_key], input_key, full_input_dict)

            # input_key = side + '_PoG_cm_tobii'
            # # print(full_input_dict['left_PoG_cm_tobii'].shape)
            # # 16 x 30 x 2
            # interm_key = side + '_PoG_cm_initial'
            # output_key = side + '_ada_loss'
            # gt_vectors = get_all_vectors_from_points(full_input_dict[input_key])
            # pd_vectors = get_all_vectors_from_points(full_intermediate_dict[interm_key])
            # gt_angulars = cal_angular_through_vectors(gt_vectors[:, :-1], gt_vectors[:, 1:])
            # # print(gt_angulars.shape)
            # # 16 * 30
            # pd_angulars = cal_angular_through_vectors(pd_vectors[:, :-1], pd_vectors[:, 1:])
            # gt_angulars_mean = torch.mean(gt_angulars)
            # pd_angulars_mean = torch.mean(pd_angulars)
            # output_dict[output_key] = torch.abs(gt_angulars_mean - pd_angulars_mean)

            # input_key = side + '_PoG_cm_tobii'
            # interm_key = side + '_PoG_cm_initial'
            # output_key = side + '_weighted_ada_loss'
            # # 16 x 30 x 1
            ## distances = get_all_distance(full_input_dict[input_key], full_intermediate_dict[interm_key])
            # # gt_angulars: 16x30
            # # pd_angulars: 16x30
            # # angulars_diff: 16x30
            # angulars_diff = torch.abs(gt_angulars - pd_angulars)
            # distances加权angulars_diff
            # weighted_angulars_diff = angulars_diff * distances
            # weighted_angulars_diff = angulars_diff + distances
            # output_dict[output_key] = torch.mean(weighted_angulars_diff)





        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        full_loss = torch.zeros(()).to(device)
        full_loss += output_dict['loss_ang_face_g_initial'] 
        # full_loss += 0.01*(output_dict['left_ada_loss'] + output_dict['right_ada_loss'])
        # full_loss += 0.0005*output_dict['face_weighted_ada_loss']

        output_dict['full_loss'] = full_loss
        
        return output_dict
    
def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    
class MyResNet18(nn.Module):
    def __init__(self):
        super(MyResNet18, self).__init__()
        self.norm_layer = nn.InstanceNorm2d
        self.inplanes = 64
        self.dilation = 1

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self.norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.downsample1 = nn.Sequential(
            nn.Conv2d(5*64, 5, 1), # 29 x 64
            nn.InstanceNorm2d(5)
        )

        self.downsample1_1 = nn.Sequential(
            nn.Conv2d(9*5, 30, 1), # 29 x 64
            nn.InstanceNorm2d(30)
        )

        self.downsample1_1_1 = nn.Sequential(
            nn.Conv2d(9*5, 30, 1), # 29 x 64
            nn.InstanceNorm2d(30)
        )

        self.downsample2 = nn.Sequential(
            nn.Conv2d(5*128, 5, 1), # 29 * 128
            nn.InstanceNorm2d(5)
        )

        self.downsample2_1 = nn.Sequential(
            nn.Conv2d(9*5, 30, 1), # 29 x 128
            nn.InstanceNorm2d(30)
        )

        self.downsample2_1_1 = nn.Sequential(
            nn.Conv2d(9*5, 30, 1), # 29 x 128
            nn.InstanceNorm2d(30)
        )

        self.downsample3 = nn.Sequential(
            nn.Conv2d(5*256, 5, 1), # 29 x 256
            nn.InstanceNorm2d(5)
        )

        self.downsample3_1 = nn.Sequential(
            nn.Conv2d(9*5, 30, 1), # 29 x 256
            nn.InstanceNorm2d(30)
        )

        self.downsample3_1_1 = nn.Sequential(
            nn.Conv2d(9*5, 30, 1), # 29 x 256
            nn.InstanceNorm2d(30)
        )

        self.downsample4 = nn.Sequential(
            nn.Conv2d(5*512, 5, 1), # 29 x 256
            nn.InstanceNorm2d(5)
        )

        self.downsample4_1 = nn.Sequential(
            nn.Conv2d(9*5, 30, 1), # 29 x 256
            nn.InstanceNorm2d(30)
        )

        self.downsample4_1_1 = nn.Sequential(
            nn.Conv2d(9*5, 30, 1), # 29 x 256
            nn.InstanceNorm2d(30)
        )

        self.fusionblock1 = BasicBlock(5*64, 5, downsample=self.downsample1)
        self.fusionblock1_1 = BasicBlock(9*5, 30, downsample=self.downsample1_1)
        self.fusionblock1_1_1 = BasicBlock(9*5, 30, downsample=self.downsample1_1_1)
        self.fusionblock2 = BasicBlock(5*128, 5, downsample=self.downsample2)
        self.fusionblock2_1 = BasicBlock(9*5, 30, downsample=self.downsample2_1)
        self.fusionblock2_1_1 = BasicBlock(9*5, 30, downsample=self.downsample2_1_1)
        self.fusionblock3 = BasicBlock(5*256, 5, downsample=self.downsample3)
        self.fusionblock3_1 = BasicBlock(9*5, 30, downsample=self.downsample3_1)
        self.fusionblock3_1_1 = BasicBlock(9*5, 30, downsample=self.downsample3_1_1)
        self.fusionblock4 = BasicBlock(5*512, 5, downsample=self.downsample4)
        self.fusionblock4_1 = BasicBlock(9*5, 30, downsample=self.downsample4_1)
        self.fusionblock4_1_1 = BasicBlock(9*5, 30, downsample=self.downsample4_1_1)

        self.layer1 = self._make_layer(BasicBlock, 64, 2)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, 512)
    
    def forward(self, x: Tensor) -> Tensor:
        batch_size = 16
        seq_len = 30

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        feature1 = self.layer1(x)
        feature1_c = feature1.shape[1]
        feature1_h = feature1.shape[2]
        feature1_w = feature1.shape[3]

        
        # (16 x 30) x 64 x 64 x 64 -> 16 x 30 x 64 x 64 x 64
        feature1 = feature1.reshape(batch_size, seq_len, feature1_c, feature1_h, feature1_w)
        # 16 x 29 x 64 x 64 x 64
        difference1 = feature1[:, 1:] - feature1[:, :-1]
        # 9 x (16 x 5 x 64 x 64 x 64)
        difference1_windows = sliding_window(difference1, window_size=5, step_size = 3)
        diff1_out = []
        for diff1 in difference1_windows:
            # 16 x (5x64) x 64 x 64 
            diff1 = diff1.reshape(batch_size, 5*feature1_c, feature1_h, feature1_w)
            diff1 = self.fusionblock1(diff1)
            # 16 x 5 x 64 x 64
            diff1_out.append(diff1)
        
        # 16 x 45 x 64 x 64
        difference1_bind = torch.cat(diff1_out, dim=1)
        difference1 = self.fusionblock1_1(difference1_bind)
        difference1 = difference1.reshape(batch_size, seq_len, 1, feature1_h, feature1_w)
        feature1 = feature1 + difference1
        feature1 = feature1.reshape(batch_size*seq_len, feature1_c, feature1_h, feature1_w)
        

        feature2 = self.layer2(feature1)
        feature2_c = feature2.shape[1]
        feature2_h = feature2.shape[2]
        feature2_w = feature2.shape[3]

        # (16 x 30) x 128 x 32 x 32 -> 16 x 30 x 128 x 32 x 32
        feature2 = feature2.reshape(batch_size, seq_len, feature2_c, feature2_h, feature2_w)
        # 16 x 29 x 128 x 32 x 32
        difference2 = feature2[:, 1:] - feature2[:, :-1]
        difference2_windows = sliding_window(difference2, window_size=5, step_size = 3)
        diff2_out = []
        for diff2 in difference2_windows:
            # 16 x (5x128) x 32 x 32 
            diff2 = diff2.reshape(batch_size, 5*feature2_c, feature2_h, feature2_w)
            diff2 = self.fusionblock2(diff2)
            # 16 x 5 x 32 x 32
            diff2_out.append(diff2)
        difference2_bind = torch.cat(diff2_out, dim=1)
        difference2 = self.fusionblock2_1(difference2_bind)
        difference2 = difference2.reshape(batch_size, seq_len, 1, feature2_h, feature2_w)
        feature2 = feature2 + difference2
        feature2 = feature2.reshape(batch_size*seq_len, feature2_c, feature2_h, feature2_w)




        feature3 = self.layer3(feature2)
        feature3_c = feature3.shape[1]
        feature3_h = feature3.shape[2]
        feature3_w = feature3.shape[3]

        # (16 x 30) x 256 x 16 x 16 -> 16 x 30 x 256 x 16 x 16
        feature3 = feature3.reshape(batch_size, seq_len, feature3_c, feature3_h, feature3_w)
        # 16 x 29 x 256 x 16 x 16
        difference3 = feature3[:, 1:] - feature3[:, :-1]
        difference3_windows = sliding_window(difference3, window_size=5, step_size = 3)
        diff3_out = []
        for diff3 in difference3_windows:
            # 16 x (5x256) x 16 x 16 
            diff3 = diff3.reshape(batch_size, 5*feature3_c, feature3_h, feature3_w)
            diff3 = self.fusionblock3(diff3)
            # 16 x 5 x 16 x 16
            diff3_out.append(diff3)
        difference3_bind = torch.cat(diff3_out, dim=1)
        difference3 = self.fusionblock3_1(difference3_bind)
        difference3 = difference3.reshape(batch_size, seq_len, 1, feature3_h, feature3_w)
        feature3 = feature3 + difference3
        feature3 = feature3.reshape(batch_size*seq_len, feature3_c, feature3_h, feature3_w)
        

        feature4 = self.layer4(feature3)
        feature4_c = feature4.shape[1]
        feature4_h = feature4.shape[2]
        feature4_w = feature4.shape[3]
        feature4 = feature4.reshape(batch_size, seq_len, feature4_c, feature4_h, feature4_w)
        difference4 = feature4[:, 1:] - feature4[:, :-1]
        
        difference4_windows = sliding_window(difference4, window_size=5, step_size = 3)
        
        diff4_out = []
        for diff4 in difference4_windows:
            diff4 = diff4.reshape(batch_size, 5*feature4_c, feature4_h, feature4_w)
            diff4 = self.fusionblock4(diff4)
            diff4_out.append(diff4)
        
        # bottom up
        difference4_bind = torch.cat(diff4_out, dim=1)
        
        
        difference4_bind_up = F.interpolate(difference4_bind, scale_factor=2, mode='nearest')
        
        
        difference3_bind = difference4_bind_up + difference3_bind
        difference3_bind_up = F.interpolate(difference3_bind, scale_factor=2, mode='nearest')
        difference2_bind = difference3_bind_up + difference2_bind
        difference2_bind_up = F.interpolate(difference2_bind, scale_factor=2, mode='nearest')
        # difference1_bind = difference2_bind_up + difference1_bind
        difference1_bind = difference2_bind_up

        difference1_bind = self.fusionblock1_1_1(difference1_bind)
        difference2_bind = self.fusionblock2_1_1(difference2_bind)
        difference3_bind = self.fusionblock3_1_1(difference3_bind)

       
        difference1_bind = difference1_bind.reshape((batch_size*seq_len), 1, feature1_h, feature1_w)
        feature1 = feature1 + difference1_bind
        feature2 = self.layer2(feature1)
        difference2_bind = difference2_bind.reshape((batch_size*seq_len), 1, feature2_h, feature2_w)
        feature2 = feature2 + difference2_bind
        feature3 = self.layer3(feature2)
        difference3_bind = difference3_bind.reshape((batch_size*seq_len), 1, feature3_h, feature3_w)
        feature3 = feature3 + difference3_bind
        feature4 = self.layer4(feature3)

        x = self.avgpool(feature4)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        

        return x, feature1, feature2, feature3, feature4

    def _make_layer(
        self,
        block: BasicBlock,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self.norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, 1, 64, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=1,
                    base_width=64,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)
        
    
def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )
        

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
    