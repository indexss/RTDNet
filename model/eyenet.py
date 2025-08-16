import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock, ResNet
import logging
import numpy as np
from collections import OrderedDict
import functools
# import wandb
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

class EyeNet(nn.Module):
    def __init__(self):
        super(EyeNet, self).__init__()
        num_features = hp['eye_net_rnn_num_features']

        # ResNet-18
        self.cnn_layer = ResNet(block=BasicBlock,
                                layers=[2, 2, 2, 2],
                                num_classes=num_features,
                                norm_layer=nn.InstanceNorm2d)

        # Fully Connected
        self.fc_common = nn.Sequential(
            
            # Use Head Pose
            nn.Linear(num_features + 2, num_features),
            nn.SELU(inplace=True),
            nn.Linear(num_features, num_features)
        )

        # GRU Cell
        self.rnn_cell = nn.GRUCell(input_size=hp['eye_net_rnn_num_features'],
                                   hidden_size=hp['eye_net_rnn_num_features'])

        # Fully Connected to Gaze Direction
        self.fc_to_gaze = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.SELU(inplace=True),
            nn.Linear(num_features, out_features=2, bias=False),
            nn.Tanh()
        )

        # Fully Connected to Pupil Size
        self.fc_to_pupil = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.SELU(inplace=True),
            nn.Linear(num_features, out_features=1),
            nn.ReLU(inplace=True)
        )

        # Set gaze layer weights to zero as otherwise this can
        # explode early in training
        nn.init.zeros_(self.fc_to_gaze[-2].weight)

        # for layer in self.fc_to_pupil:
        #     if isinstance(layer, nn.Linear):
        #         nn.init.xavier_uniform_(layer.weight)

    def forward(self, input_sub_dict, side, initial_states=None):
        # 对于EyeNet这个模块，它应当只处理单张图片，这样的灵活性最高
        # 当遇到视频的时候，我们只需要将其搞成一个序列，再逐个输入，通过initial_states的传递保证连续性
        # 所以EyeNet只接受单张图片（或者batch_size * 单张图片，batch的支持由pytorch自动实现）

        input_image = input_sub_dict[side + '_eye_patch']

        # CNN
        initial_features = self.cnn_layer(input_image)

        # concat head pose
        initial_features = torch.cat([initial_features, input_sub_dict[side + '_h']], axis=1)

        # Fully Connected
        initial_features = self.fc_common(initial_features)

        # RNN
        rnn_features = initial_features
        previous_states = None
        if initial_states is not None:
            previous_states = initial_states
        states = self.rnn_cell(rnn_features, previous_states)

        if isinstance(states, tuple):
            rnn_features = states[0]
            output_states = states
        else:
            rnn_features = states
            output_states = states
        features = rnn_features

        # To Gaze and pupil size
        gaze_prediction = np.pi / 2 * self.fc_to_gaze(features)
        pupil_size = self.fc_to_pupil(features).reshape(-1)

        return gaze_prediction, pupil_size, output_states