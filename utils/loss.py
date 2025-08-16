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
from utils.util_functions import pitchyaw_to_vector

logger = logging.getLogger(__name__)
config = DefaultConfig()

class BaseLossWithValidity(object):

    def calculate_loss(self, predictions, ground_truth):
        raise NotImplementedError('Must implement BaseLossWithValidity::calculate_loss')

    def calculate_mean_loss(self, predictions, ground_truth):
        return torch.mean(self.calculate_loss(predictions, ground_truth))

    def __call__(self, predictions, gt_key, reference_dict):
        # Since we deal with sequence data, assume B x T x F (if ndim == 3)
        batch_size = predictions.shape[0]

        individual_entry_losses = []
        num_valid_entries = 0

        for b in range(batch_size):
            # Get sequence data for predictions and GT
            entry_predictions = predictions[b]
            entry_ground_truth = reference_dict[gt_key][b]

            # If validity values do not exist, return simple mean
            # NOTE: We assert for now to catch unintended errors,
            #       as we do not expect a situation where these flags do not exist.
            validity_key = gt_key + '_validity'
            assert (validity_key in reference_dict)
            # if validity_key not in reference_dict:
            #     individual_entry_losses.append(torch.mean(
            #         self.calculate_mean_loss(entry_predictions, entry_ground_truth)
            #     ))
            #     continue

            # Otherwise, we need to set invalid entries to zero
            validity = reference_dict[validity_key][b].float()
            losses = self.calculate_loss(entry_predictions, entry_ground_truth)

            # Some checks to make sure that broadcasting is not hiding errors
            # in terms of consistency in return values
            assert (validity.ndim == losses.ndim)
            assert (validity.shape[0] == losses.shape[0])

            # Make sure to scale the accumulated loss correctly
            num_valid = torch.sum(validity)
            accumulated_loss = torch.sum(validity * losses)
            if num_valid > 1:
                accumulated_loss /= num_valid
            num_valid_entries += 1
            individual_entry_losses.append(accumulated_loss)

        # Merge all loss terms to yield final single scalar
        return torch.sum(torch.stack(individual_entry_losses)) / float(num_valid_entries)


class L1Loss(BaseLossWithValidity):

    def calculate_loss(self, a, b):
        assert (a.ndim == b.ndim)
        if a.ndim > 1:
            return torch.mean(torch.abs(a - b), axis=tuple(range(1, a.ndim)))
        else:
            return torch.abs(a - b)


class AngularLoss(BaseLossWithValidity):
    _to_degrees = 180. / np.pi

    def calculate_loss(self, a, b):
        a = pitchyaw_to_vector(a)
        b = pitchyaw_to_vector(b)
        sim = F.cosine_similarity(a, b, dim=1, eps=1e-8)
        sim = F.hardtanh_(sim, min_val=-1 + 1e-8, max_val=1 - 1e-8)
        return torch.acos(sim) * self._to_degrees
    
class EuclideanLoss(BaseLossWithValidity):

    def calculate_loss(self, a, b):
        assert(a.ndim == b.ndim)
        assert(a.ndim > 1)
        squared_difference = torch.pow(a - b, 2)
        ssd = torch.sum(squared_difference, axis=tuple(range(1, a.ndim)))
        return torch.sqrt(ssd)