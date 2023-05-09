import os
import pprint
import argparse
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import tqdm
from utils.model_utils import loader, saver, only_model_saver
from astnet import eval_
import sys
import MSloss
import datasets
import models as models
from utils import utils
from config import config, update_config
import core
import numpy as np
from MSloss import MSSSIM
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from tensorboardX import SummaryWriter
import core
import csv
from datetime import datetime



def train():

    # creating out put dirs
    
    paths = dict(log_dir="%s/%s/%s" % (args.proj_root,'log', config.DATASET.DATASET),
                 ckpt_dir="%s/%s/%s" % (args.proj_root,'ckpt', config.DATASET.DATASET))

    os.makedirs(paths["ckpt_dir"], exist_ok=True)
    os.makedirs(paths["log_dir"], exist_ok=True)
    print(paths["ckpt_dir"])

    intensity_loss = core.Intensity_Loss().to(device=config.GPUS[0])
    gradient_loss = core.Gradient_Loss(3).to(device=config.GPUS[0])
    
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.determinstic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    model = models.get_net(config)
    model=model.to(device=config.GPUS[0])

    train_dataset = eval('datasets.get_train_data')(config)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        # batch_size=config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=True,
        num_workers=config.WORKERS,
        pin_memory=True
        
    )

    learing_rate = 0.0002
    optimizer = torch.optim.Adam(model.parameters(), lr=learing_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 100], gamma=0.5, last_epoch=-1)

    loss_func_mse = nn.MSELoss(reduction='none')
    
    ef = config.MODEL.ENCODED_FRAMES
    df = config.MODEL.DECODED_FRAMES
    fp = ef + df 

    step = 0
    epoch_last = 0
    running_loss_all = 0.0
    running_loss_kl = 0.0
    running_loss_frame = 0.0
    running_loss_grad = 0.0
    prev_epoch = 0
    epoch_now = 0

    resume = args.resume 
    if resume:
        ckpt_path = os.path.join(paths["ckpt_dir"], config.DATASET.DATASET + config.MODEL.NAME) + '-' + str(args.epoch) + '.pth'
        print("continue checkpoint path", ckpt_path)
        print(config.TRAIN.END_EPOCH)
        ep = int((ckpt_path.split('-')[-1]).split('.')[0])
        if (ep) >= config.TRAIN.END_EPOCH:
            print("Error ckpt greater than epochs :/ ")
        elif os.path.isfile(ckpt_path) and ep < config.TRAIN.END_EPOCH:
            epoch_now = ep
            
            print("Resume from checkpoint...")
            checkpoint = torch.load(ckpt_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # epoch_now = int(ckpt_path.split('-')[-1])
            print("====>Epoch ofloaded checkpoint",epoch_now)
        else:
            print("====>no checkpoint found.")

    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")
    filename = os.path.join(paths["log_dir"], config.DATASET.DATASET + config.MODEL.NAME + dt_string + '.csv' )

    with open(filename, 'w') as csvfile: 
        # creating a csv writer object 
        csvwriter = csv.writer(csvfile) 
        csvwriter.writerow(['grad_l','inte_l','MSSIM','GLT','step','epoch','video_num'])

    for epoch in tqdm.tqdm(range(config.TRAIN.BEGIN_EPOCH + epoch_now,config.TRAIN.END_EPOCH)):
        
        runningLoss = []
        best_auroc = -1
        for i, data in enumerate(train_loader):

            video, _ = core.decode_input(input=data, train=True)

            video = [frame.to(device=config.GPUS[0]) for frame in video]

            j = 0
            for f in tqdm.tqdm(range(len(video) - fp)):
                j += 1
                inputs = video[f:f + fp]
                output = model(inputs)
                target = video[f + fp:f + fp + 1][0]

                # if ((f % 50 == 0) and (i % 10 == 0)):
                #     plt.imshow(np.transpose(torch.squeeze(output.cpu(),dim=0).detach().numpy(),[1,2,0]))
                #     plt.show()    
                #     plt.imshow(np.transpose(torch.squeeze(target.cpu(),dim=0).detach().numpy(),[1,2,0]))
                #     plt.show()
                
                mse_imgs = torch.mean(loss_func_mse((output[0] + 1) / 2, (target[0] + 1) / 2)).item()
                grad_l = gradient_loss(output,target) #梯度约束
                grad_l.requires_grad_(True)
                inte_l = intensity_loss(output,target) #强度约束
                inte_l.requires_grad_(True)

                ms_ssim_out =MSloss.msssim( target,output)
                ms_ssim_out.requires_grad_(True)

                G_l_t = 1. * inte_l + 1. * grad_l+1. * ms_ssim_out
                G_l_t.requires_grad_(True)

                # if ((f % 50 == 0) and (i % 10 == 0)):
                #     print("epoch/120:",epoch,"  ",j,"/",len(video) - fp,"    ",G_l_t.item())
                
                with open(filename, 'a') as csvfile: 
                    # creating a csv writer object 
                    csvwriter = csv.writer(csvfile) 
                    csvwriter.writerow([grad_l.clone().detach().item(), inte_l.clone().detach().item(), ms_ssim_out.clone().detach().item(), G_l_t.clone().detach().item(), step+1, epoch+1, i])        
                    
                optimizer.zero_grad() 
                G_l_t.backward()
                optimizer.step()

                step += 1
            
            runningLoss.append(G_l_t.clone().detach().item())
        
        # deprecated

        # checkpoint = {"model_state_dict": model.state_dict(),
        #                   "optimizer_state_dict": optimizer.state_dict(),
        #                   "epoch": epoch}
        # path_checkpoint = "/home/nitr/Devansh/astnet/astnet_all/checkpoint/{}_model.pth".format(epoch)
        # torch.save(checkpoint, path_checkpoint)
        prev_epoch = step+1
        scheduler.step()
        print("loss ", np.average(runningLoss))
        
        # saving stats

        if epoch % config.TRAIN.SAVEEVERY == config.TRAIN.SAVEEVERY - 1:
            model_save_path = os.path.join(paths["ckpt_dir"], config.DATASET.DATASET + config.MODEL.NAME)
            saver(model.state_dict(), optimizer.state_dict(), model_save_path, epoch + 1, step, max_to_save=5)

            # computer training stats
            stats_save_path = os.path.join(paths["ckpt_dir"], "training_stats_%d.npy" % (epoch + 1))



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='ASTNet for Anomaly Detection')

    parser.add_argument("--proj_root", type=str, default="/home/nitr/Devansh/astnet/astMod/workingBase", help='project root path')
    parser.add_argument('--cfg', help='experiment configuration filename', required=True, type=str)
    parser.add_argument('--model-file', help='model parameters',  type=str)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--epoch', type=int)
    parser.add_argument('opts', 
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    sys.path.append(os.getcwd())
    update_config(config, args)
    
    train()
