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
from model.eyenet_frame import EyeNetFrame
from model.facenet_transformer import FaceNetTransformer
from utils.util_functions import pitchyaw_to_vector, subset_test
from utils.loss import BaseLossWithValidity, L1Loss, AngularLoss
from core.training import init_datasets
from utils.wandb_setup import init_wandb
from eval import do_final_test
from model.face_mixste import FaceMixSTE

logger = logging.getLogger(__name__)
config = DefaultConfig()

hp = {
    'eye_net_rnn_num_features': 128,
    'eye_net_rnn_num_cells': 1,
    'batch_size': config.batch_size,
    'epoch_num': 10,
    'sequence_length': 30,
    'saving_dir': './saving',
}

if __name__ == '__main__':

    run = init_wandb()
    logging.basicConfig(level=logging.INFO)
    logging.info('Start training')
    logging.info('Hyperparameters: %s', hp)

    ############## train end 1###########################

    train_dataset_paths = [
        ('eve_train',
         EVESequences_train,
         '/root/autodl-tmp/eve_dataset/eve_dataset',
         #  '/media/linlishi/Extend/EVE/eve_dataset',
        #  '/media/linlishi/Extend/EVE/eve_dataset',
         # '/home/luanfuzi/dataset/eve_dataset',
        #  '/run/media/linlishi/Extend/EVE/eve_dataset',
        # '/Volumes/Extend/EVE/eve_dataset',
         ['image', 'video', 'wikipedia'],
         ['basler', 'webcam_l', 'webcam_c', 'webcam_r']),  # noqa
    ]

    val_dataset_paths = [
        ('eve_val', EVESequences_val,
        '/root/autodl-tmp/eve_dataset/eve_dataset',
         #  '/media/linlishi/Extend/EVE/eve_dataset',
        #  '/media/linlishi/Extend/EVE/eve_dataset',
         # '/home/luanfuzi/dataset/eve_dataset',
        #  '/run/media/linlishi/Extend/EVE/eve_dataset',
        # '/Volumes/Extend/EVE/eve_dataset',
        
         ['image', 'video', 'wikipedia'],
         ['basler', 'webcam_l', 'webcam_c', 'webcam_r']),
    ]

    train_data, test_data = init_datasets(train_dataset_paths, val_dataset_paths)
    
    # mydata = None
    # for i,data in enumerate(test_data['eve_val']['dataloader']):
    #     mydata = data
    #     break
    # for k, v in mydata.items():
    #     print(k, v.shape)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: {}".format(torch.cuda.get_device_name(device)))
    
    # model = EyeNetFrame().to(device)
    model = FaceNetTransformer().to(device)
    model.load_state_dict(torch.load('/root/autodl-tmp/projects/MyEVE-Transformer/saving/eveframe_epoch_9_batch_4069_time_20240622_103221.pth'))
    # model = FaceMixSTE().to(device)
    model.train()

    learning_rate = 0.0000625 * (0.5**10)

    optimizers = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.005,
    )

    num_epochs = 5

    timestamp = time.time()
    utc_time = datetime.fromtimestamp(timestamp, timezone.utc)
    # writer = SummaryWriter('./tensorboard/UTC' + utc_time.strftime('%Y%m%d_%H%M%S'))

    dataloader = train_data['eve_train']['dataloader']
    dataset = train_data['eve_train']['dataset']
    len_dataset = len(dataset)
    batch_size = config.batch_size
    num_batches = len_dataset // batch_size
    current_step = 0

    for epoch in range(num_epochs):
        model.train()
        # print learning rate now
        # print(f'Learning rate for epoch {epoch + 1} is {optimizers.state_dict()["param_groups"][0]["lr"]}')
        for i, data in enumerate(dataloader):
            # if i == 10:
            #     break
            optimizers.zero_grad()
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    data[k] = v.to(device)
            output_dict = model(data)
            loss = output_dict['full_loss']
            loss.backward()
            optimizers.step()
            if i == num_batches // 2:
                # lr_now = optimizers.state_dict()["param_groups"][0]["lr"]
                # lr_next = lr_now * 0.5
                # for param_group in optimizers.param_groups:
                #     param_group['lr'] = lr_next
                path = "./saving/eveframe_epoch_" + str(epoch) +"_batch_" + str(i)+"_time_"+utc_time.strftime('%Y%m%d_%H%M%S')+".pth"
                torch.save(model.state_dict(), path)
                artifact = wandb.Artifact('model', type='model')
                artifact.add_file(path)
                run.log_artifact(artifact)

            # wandb
            wandb.log({"full_loss": output_dict['full_loss'].item(),
                    #    "loss_l1_left_pupil_size": output_dict['loss_l1_left_pupil_size'].item(),
                    #    'loss_l1_right_pupil_size': output_dict['loss_l1_right_pupil_size'].item(),
                       'loss_ang_left_g_initial': output_dict['loss_ang_left_g_initial'].item(),
                       'loss_ang_right_g_initial': output_dict['loss_ang_right_g_initial'].item(),
                       'learning_rate': optimizers.state_dict()["param_groups"][0]["lr"],
                       'loss_euc_left_PoG_cm_initial': output_dict['euc_loss_left_PoG_cm_initial'].item(),
                       'loss_euc_right_PoG_cm_initial': output_dict['euc_loss_right_PoG_cm_initial'].item(),
                       'loss_euc_left_PoG_px_initial': output_dict['euc_loss_left_PoG_px_initial'].item(),
                       'loss_euc_right_PoG_px_initial': output_dict['euc_loss_right_PoG_px_initial'].item(),
                       })
            # print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{i}/{4069}], Loss: {output_dict}")
            lr = optimizers.state_dict()["param_groups"][0]["lr"]
            print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{num_batches}], LR: {lr}")
        lr_now = optimizers.state_dict()["param_groups"][0]["lr"]
        lr_next = lr_now * 0.5
        for param_group in optimizers.param_groups:
            param_group['lr'] = lr_next
        
        # do a small subset test
        # test_dict = subset_test(model, test_data)
        # wandb.log(test_dict)

        # save model
        path = "./saving/eveframe_epoch_" + str(epoch) +f"_batch_{num_batches}"+"_time_"+utc_time.strftime('%Y%m%d_%H%M%S')+".pth"
        torch.save(model.state_dict(), path)
        artifact = wandb.Artifact('model', type='model')
        artifact.add_file(path)
        run.log_artifact(artifact)

    model_parameters_final_path = "./saving/eveframe_epoch_" + str(num_epochs-1) +f"_batch_{num_batches}"+"_time_"+utc_time.strftime('%Y%m%d_%H%M%S')+".pth"
    final_dict = do_final_test(model_parameters_final_path)
    wandb.log(final_dict)
    
    print('Finished Training')
    wandb.finish()
    
    os.system("shutdown now -h")
    ############## train end 2###########################



