import os
import torch
# from utils.core_utils import RunningStatistics
# from datasources.EVE import EVEDataset_test
# from datasources.Gaze360 import Gaze360Loader
# from datasources.Eyediap import EyediapLoader
from datasources.EyediapTest import EyediapLoaderTest
from torch.utils.data import DataLoader
# from utils.train_utils import my_collate
from tqdm import tqdm

import argparse
import json
import numpy as np
import os
import torch
from collections import OrderedDict
from torch.utils.data import DataLoader
from tqdm import tqdm
from argparse import Namespace
import torch.nn.functional as F
import random
# from models.facenet_transformer import FaceNetTransformer
from model.facenet_transformer import FaceNetTransformer
import time
from datetime import datetime, timezone
import wandb
from utils.wandb_setup import init_wandb

eyediap_path = "/root/autodl-tmp/EyeDiap_Face_256"

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

def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) ==0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

def set_seed(seed=0):
    random.seed(seed) 
    np.random.seed(seed)  
    os.environ['PYTHONHASHSEED'] = str(seed)  
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)

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
    eyediap_dataset = EyediapLoaderTest(source_path=eyediap_path, config=args, transforms = None)
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
                # print(k, v.shape)
        # print(input_data['face_h'][0])
        # input_data['face_h'] = -input_data['face_h']
        # print(input_data['face_h'][0])
        output_dict = model(input_data)
        print("gaze_diff", output_dict['loss_ang_face_g_initial'].detach().cpu().numpy())
        test_losses.add('%s' % 'gaze_diff', output_dict['loss_ang_face_g_initial'].detach().cpu().numpy())
    test_loss_means = test_losses.means()
    print('Test Losses for Eyediap test data %s' % (', '.join(['%s: %.6f' % v for v in test_loss_means.items()])))
    # print(test_loss_means['gaze_diff'])
    return test_loss_means['gaze_diff']

# def eyediap_test(args, device, run=None, datapath=None):
#     # eyediap_dataset = EyediapLoader(source_path=eyediap_path, config=args, transforms = None)
#     eyediap_dataset = EyediapLoaderTest(source_path=eyediap_path, config=args, transforms = None)
#     num_batches = len(eyediap_dataset) // 16
#     test_eyediap_dataloader = DataLoader(
#         eyediap_dataset,
#         batch_size = 16,
#         shuffle = True,
#         num_workers = 4,
#         pin_memory = True,
#         collate_fn = my_collate
#     )
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("Using device: {}".format(torch.cuda.get_device_name(device)))

#     model = FaceNetTransformer().to(device)
#     checkpoint = torch.load(datapath)
#     # model.load_state_dict(checkpoint['model_state_dict'])
#     model.load_state_dict(checkpoint)
#     # model.load_state_dict(torch.load(datapath))
#     # model.train()
#     model.eval()
#     set_seed(0)
#     test_losses = RunningStatistics()
#     for i, input_data in enumerate(test_eyediap_dataloader):
#         print(f'------------------Batc123h {i}--------------------')
#         for k, v in input_data.items():
#             if isinstance(v, torch.Tensor):
#                 input_data[k] = v.detach().to(device, non_blocking=True)
#         print(1232123212321)
#         print(input_data['face_h'])
#         input_data['face_h'] = -input_data['face_h']
#         print(input_data['face_h'])
#         output_dict = model(input_data)
#         print("gaze_diff", output_dict['loss_ang_face_g_initial'].detach().cpu().numpy())
#         test_losses.add('%s' % 'gaze_diff', output_dict['loss_ang_face_g_initial'].detach().cpu().numpy())
#     test_loss_means = test_losses.means()
#     print('Test Losses for Eyediap test data %s' % (', '.join(['%s: %.6f' % v for v in test_loss_means.items()])))
    
        
                   


if __name__ == "__main__":
    
    defualt_config = json.load(open('/root/autodl-tmp/projects/move2-face/core/my1.json'))
    config = {**defualt_config}
    config = Namespace(**config)
    config.tanh = True
    config.learning_rate = config.base_learning_rate * config.batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # run = init_wandb()
    # path = "/root/autodl-tmp/projects/move-idp/saving/eyediap_iter_11999_time_20240823_091346.pth"
    # path = "/root/autodl-tmp/projects/move-idp/saving/eyediap_iter_15999_time_20240823_091346.pth"
    # path = "/root/autodl-tmp/projects/move-idp/saving/eyediap_iter_19999_time_20240823_091346.pth"
    # path = "/root/autodl-tmp/projects/move-idp/saving/eyediap_iter_23999_time_20240823_091346.pth"
    # path = "/root/autodl-tmp/projects/move-idp/saving/eyediap_iter_27999_time_20240823_091346.pth"
    # path = "/root/autodl-tmp/projects/move-idp/saving/eyediap_iter_31999_time_20240823_091346.pth"
    # path = "/root/autodl-tmp/projects/move-idp/saving/eyediap_iter_35999_time_20240823_091346.pth"
    # path = "/root/autodl-tmp/projects/move-idp/saving/eyediap_iter_39999_time_20240823_091346.pth"
    # path = "/root/autodl-tmp/projects/move-idp/saving/eyediap_iter_43999_time_20240823_080219.pth"
    # path = "/root/autodl-tmp/projects/move-idp/saving/eyediap_iter_47999_time_20240823_080219.pth"
    path_list = [
        "/root/autodl-tmp/projects/move2-face/saving/eveframe_epoch_9_batch_2034_time_20240824_093923.pth"
    ]
    model = FaceNetTransformer().to(device)
    model.load_state_dict(torch.load(path_list[0]))
    eyediap_mid_test(config, device, None, model)
    # for path in path_list:
    #     print(f'------------------Path {path}--------------------')
    #     eyediap_test(config, device, None, datapath=path)
    
    # wandb.finish()
    
