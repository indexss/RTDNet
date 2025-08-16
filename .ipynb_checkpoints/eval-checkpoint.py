"""
test的时候batch size是128，有60(上下 忘了)个batch。如果你在跑这份代码的时候发现电脑有点死机或者干脆DE崩了，那我建议你重启一下然后直接用命令行跑。
因为我就崩了好几次，每次都是重启后第一次跑就好了。我猜测是因为显存/RAM没有释放干净，所以重启后就好了。
"""
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
from datasources.EyediapTest import EyediapLoaderTest

from torch.utils.data import DataLoader, Subset
from datasources import EVESequences_train, EVESequences_test, EVESequences_val
from core import DefaultConfig
from torch.optim.lr_scheduler import ExponentialLR

from torch.utils.tensorboard import SummaryWriter
import time
from datetime import datetime, timezone
from model.eyenet import EyeNet
from model.eyenet_frame import EyeNetFrame
from utils.loss import BaseLossWithValidity, L1Loss, AngularLoss, EuclideanLoss
from utils.util_functions import pitchyaw_to_vector
from utils.util_functions import to_screen_coordinates, apply_rotation, apply_transformation, get_intersect_with_zero
from utils.util_functions import set_seed
from core.training import init_datasets_eval
from core import DefaultConfig, training
from datasources import EVESequencesBase, EVESequences_train, EVESequences_val, EVESequences_test  # for i,data in enumerate(test_dataloader):
from model.facenet_transformer import FaceNetTransformer
from model.face_mixste import FaceMixSTE

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

def do_final_test(path):
    set_seed(1024)
    config = DefaultConfig()
    train_dataset_paths = [
            ('eve_train', EVESequences_train, '/root/autodl-tmp/eve_dataset/eve_dataset', config.train_stimuli, config.train_cameras),  # noqa
        ]
    validation_dataset_paths = [
        ('eve_val', EVESequences_val, '/root/autodl-tmp/eve_dataset/eve_dataset', config.test_stimuli, config.test_cameras),
    ]
    train_data, test_data = init_datasets_eval(train_dataset_paths, validation_dataset_paths)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: {}".format(torch.cuda.get_device_name(device)))

    # model = EyeNetFrame().to(device)
    model = FaceNetTransformer().to(device)
    # model = FaceMixSTE().to(device)
    model.load_state_dict(torch.load(path))
    model.eval()

    
    test_dataloader = test_data['eve_val']['dataloader']

    #left gaze, right gaze, left pupil, right pupil, left PoG cm, left PoG px, right PoG cm, right PoG px
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
            if i == 478:
                break
            for k, v in test_data.items():
                if isinstance(v, torch.Tensor):
                    test_data[k] = v.to(device)
                    # print(k, v.shape)
            # break
            test_output_dict = model(test_data)

            # if i == 0:
            #     # print(test_output_dict)
            #     print(test_output_dict['face_patch'].shape)
            #     serial = test_output_dict['face_patch'][0]
            #     serial = serial.permute(0, 2, 3, 1)
            #     # print picture
            #     for i in range(30):
            #         plt.imshow(serial[i].cpu().numpy())
            #         plt.savefig(f'./pics/{i}.png')
            #         plt.close()
            # print(test_output_dict)
            left_gaze_diff = test_output_dict['loss_ang_face_g_initial'].item()
            # right_gaze_diff = test_output_dict['loss_ang_right_g_initial'].item()
            # left_pupil_diff = test_output_dict['loss_l1_left_pupil_size'].item()
            # right_pupil_diff = test_output_dict['loss_l1_right_pupil_size'].item()
            # left_PoG_diff_cm = test_output_dict['euc_loss_face_PoG_cm_initial'].item()
            # right_PoG_diff_cm = test_output_dict['euc_loss_right_PoG_cm_initial'].item()
            # left_PoG_diff_px = test_output_dict['euc_loss_face_PoG_px_initial'].item()
            # right_PoG_diff_px = test_output_dict['euc_loss_right_PoG_px_initial'].item()

            lg.append(left_gaze_diff)
            # rg.append(right_gaze_diff)
            # lp.append(left_pupil_diff)
            # rp.append(right_pupil_diff)
            # lgcm.append(left_PoG_diff_cm)
            # lgpx.append(left_PoG_diff_px)
            # rgcm.append(right_PoG_diff_cm)
            # rgpx.append(right_PoG_diff_px)

            print(f'--------------Batch {i}---------------------')
            print(f'face Eye Gaze Dir diff: {left_gaze_diff}°')
            # print(f'face Eye PoG cm diff: {left_PoG_diff_cm} cm')
            # print(f'face Eye PoG px diff: {left_PoG_diff_px} px')
            # print(f'Left Eye Pupil Size diff: {left_pupil_diff} mm')
            
            # print(f'Right Eye Gaze Dir diff: {right_gaze_diff}°')
            # print(f'Right Eye PoG cm diff: {right_PoG_diff_cm} cm')
            # print(f'Right Eye PoG px diff: {right_PoG_diff_px} px')
            # print(f'Right Eye Pupil Size diff: {right_pupil_diff} mm')
            
            
    # print(test_output_dict)
    lg_mean = np.mean(lg).item()
    # rg_mean = np.mean(rg).item()
    # lp_mean = np.mean(lp).item()
    # rp_mean = np.mean(rp).item()
    # lgcm_mean = np.mean(lgcm).item()
    # lgpx_mean = np.mean(lgpx).item()
    # rgcm_mean = np.mean(rgcm).item()
    # rgpx_mean = np.mean(rgpx).item()
    print('-----------------Summary-----------------')
    print(f'Mean face Eye Gaze Dir diff: {lg_mean}°')
    # print(f'Mean Left Eye Pupil Size diff: {lp_mean} mm')
    # print(f'Mean face Eye PoG cm diff: {lgcm_mean} cm')
    # print(f'Mean face Eye PoG px diff: {lgpx_mean} px')
    
    # print(f'Mean Right Eye Gaze Dir diff: {rg_mean}°')
    # print(f'Mean Right Eye Pupil Size diff: {rp_mean} mm')
    # print(f'Mean Right Eye PoG cm diff: {rgcm_mean} cm')
    # print(f'Mean Right Eye PoG px diff: {rgpx_mean} px')
    final_dict = {
        'final_left_gaze_diff': lg_mean,
        # 'final_left_pupil_diff': lp_mean,
        # 'final_left_pog_cm_diff': lgcm_mean,
        # 'final_left_pog_px_diff': lgpx_mean,
        # 'final_right_gaze_diff': rg_mean,
        # 'final_right_pupil_diff': rp_mean,
        # 'final_right_pog_cm_diff': rgcm_mean,
        # 'final_right_pog_px_diff': rgpx_mean,
    }
    return final_dict

class RunningStatistics(object):
    def __init__(self):
        self.losses = OrderedDict()

    def add(self, key, value):
        if key not in self.losses:
            self.losses[key] = []
        self.losses[key].append(value)

    def means(self):
        return OrderedDict([
            (k, np.mean(v)) if len(v) > 0 else (k, v) for k, v in self.losses.items()
        ])

    def reset(self):
        for key in self.losses.keys():
            self.losses[key] = []

    def __str__(self):
        fmtstr = ', '.join(['%s: %.6f (%.6f)' % (k, v[-1], self.means()[k]) for k, v in self.losses.items()])
        return fmtstr

def eyediap_mid_test(args, device, run=None, model=None):
     # eyediap_dataset = EyediapLoader(source_path=eyediap_path, config=args, transforms = None)
    eyediap_dataset = EyediapLoaderTest(source_path="/root/autodl-tmp/EyeDiap_Face_256", config=args, transforms = None)
    num_batches = len(eyediap_dataset) // 16
    test_eyediap_dataloader = DataLoader(
        eyediap_dataset,
        batch_size = 16,
        shuffle = True,
        num_workers = 4,
        pin_memory = True,
        collate_fn = my_collate
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(torch.cuda.get_device_name(device)))

    model.eval()
    set_seed(0)
    test_losses = RunningStatistics()
    for i, input_data in enumerate(test_eyediap_dataloader):
        print(f'------------------Batch {i}--------------------')
        for k, v in input_data.items():
            if isinstance(v, torch.Tensor):
                input_data[k] = v.detach().to(device, non_blocking=True)
        output_dict = model(input_data)
        print("gaze_diff", output_dict['loss_ang_face_g_initial'].detach().cpu().numpy())
        test_losses.add('%s' % 'gaze_diff', output_dict['loss_ang_face_g_initial'].detach().cpu().numpy())
    test_loss_means = test_losses.means()
    print('Test Losses for Eyediap test data %s' % (', '.join(['%s: %.6f' % v for v in test_loss_means.items()])))
    # print(test_loss_means['gaze_diff'])
    return test_loss_means['gaze_diff']

def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) ==0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

if __name__ == '__main__':
    path='/root/autodl-tmp/projects/m2ff/saving/eveframe_epoch_9_batch_2034_time_20240824_093923.pth'
    do_final_test(path)

    