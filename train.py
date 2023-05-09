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

def parse_args():

    parser = argparse.ArgumentParser(description='ASTNet for Anomaly Detection')

    parser.add_argument('--cfg', help='experiment configuration filename',
                        required=True, type=str)
    parser.add_argument('--model-file', help='model parameters', type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    # update_config(config, args)
    return args
writer = SummaryWriter(f'tensorboard_log/ped2_batchsize16')
def decode_input(input, train=True):
    video = input['video']
    video_name = input['video_name']
    inputs = video[:-1]
    target = video[-1]
    return inputs, target

class Intensity_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, gen_frames, gt_frames):
        return torch.mean(torch.abs((gen_frames - gt_frames) ** 2))


class Gradient_Loss(nn.Module):
    def __init__(self, channels):
        super().__init__()

        pos = torch.from_numpy(np.identity(channels, dtype=np.float32))
        neg = -1 * pos
        # Note: when doing conv2d, the channel order is different from tensorflow, so do permutation.
        self.filter_x = torch.stack((neg, pos)).unsqueeze(0).permute(3, 2, 0, 1).cuda()
        self.filter_y = torch.stack((pos.unsqueeze(0), neg.unsqueeze(0))).permute(3, 2, 0, 1).cuda()

    def forward(self, gen_frames, gt_frames):
        # Do padding to match the  result of the original tensorflow implementation
        gen_frames_x = nn.functional.pad(gen_frames, [0, 1, 0, 0])
        gen_frames_y = nn.functional.pad(gen_frames, [0, 0, 0, 1])
        gt_frames_x = nn.functional.pad(gt_frames, [0, 1, 0, 0])
        gt_frames_y = nn.functional.pad(gt_frames, [0, 0, 0, 1])

        gen_dx = torch.abs(nn.functional.conv2d(gen_frames_x, self.filter_x))
        gen_dy = torch.abs(nn.functional.conv2d(gen_frames_y, self.filter_y))
        gt_dx = torch.abs(nn.functional.conv2d(gt_frames_x, self.filter_x))
        gt_dy = torch.abs(nn.functional.conv2d(gt_frames_y, self.filter_y))

        grad_diff_x = torch.abs(gt_dx - gen_dx)
        grad_diff_y = torch.abs(gt_dy - gen_dy)

        return torch.mean(grad_diff_x + grad_diff_y)
def collate_fn(batch):
    #  batch是一个列表，其中是一个一个的元组，每个元组是dataset中_getitem__的结果
    batch = list(zip(*(batch)))
    video= torch.tensor(batch[0], dtype=torch.int32)
    video_name = batch[1]
    del batch
    return video, video_name
def train():
    intensity_loss = Intensity_Loss().cuda()
    gradient_loss = Gradient_Loss(3).cuda()
    

    logger, final_output_dir, tb_log_dir = \
        utils.create_logger(config, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.determinstic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    config.defrost()
    config.MODEL.INIT_WEIGHTS = True
    config.freeze()

    gpus = list(config.GPUS)
    model = models.get_net(config)


    logger.info('Model: {}'.format(model.get_name()))
    model=model.cuda()

    #model = nn.DataParallel(model, device_ids=gpus).cuda()
    #model = nn.DataParallel(model, device_ids=gpus).cuda(device=gpus[0])
    logger.info('Epoch: '.format(args.model_file))

    train_dataset = eval('datasets.get_train_data')(config)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        #batch_size=config.TRAIN.BATCH_SIZE_PER_GPU * len(gpus),
        shuffle=True,
        num_workers=config.WORKERS,
        pin_memory=True
        #collate_fn=collate_fn
    )
    # 定义损失函数和优化器

    learing_rate = 0.0002
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learing_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 100], gamma=0.5, last_epoch=-1)
    #train
    # 训练模型
    loss_func_mse = nn.MSELoss(reduction='none')
    psnr_list = []
    ef = config.MODEL.ENCODED_FRAMES
    df = config.MODEL.DECODED_FRAMES
    fp = ef + df 
    # best_acc=0
    # resume = False  # 设置是否需要从上次的状态继续训练
    # if resume:
    #     if os.path.isfile("/media/test/02ca50dc-830d-4673-9e13-afa0e5e097a8/Python_test/download/astnet-main/datasets/seg_weights/wider_resnet38.pth.tar"):
    #         print("Resume from checkpoint...")
    #         checkpoint = torch.load("/media/test/02ca50dc-830d-4673-9e13-afa0e5e097a8/Python_test/download/astnet-main/datasets/seg_weights/wider_resnet38.pth.tar")
    #         model.load_state_dict(checkpoint['state_dict'])
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         epoch_now = checkpoint['epoch'] + 1
    #         print("====>loaded checkpoint (epoch{})".format(checkpoint['epoch']))
    #     else:
    #         print("====>no checkpoint found.")
    #         epoch_now = 0  # 如果没进行训练过，初始训练epoch值为1

    for epoch_i in tqdm.tqdm(range(0,120)):
        runningLoss = []
        with torch.no_grad():
            for i, data in enumerate(train_loader):
                optimizer.zero_grad()
                video, video_name = decode_input(input=data, train=True)
                
                video = [frame.to(device=config.GPUS[0]) for frame in video]
                
                psnrs = []
                j = 0
                # 前向传播、计算损失和反向传播
                # print(len(video))
                for f in range(len(video) - fp):
                    #print(1)
                    j = j + 1
                    inputs = video[f:f + fp]
                    output = model(inputs)
                    target = video[f + fp:f + fp + 1][0]  # 预测的真值
                    mse_imgs = torch.mean(loss_func_mse((output[0] + 1) / 2, (target[0] + 1) / 2)).item()
                    test_acc_1=mse_imgs
                    psnr = utils.psnr_park(mse_imgs)
                    psnrs.append(float(psnr))
                    grad_l = gradient_loss(output,target) #梯度约束
                    grad_l.requires_grad_(True)
                    inte_l = intensity_loss(output,target) #强度约束
                    inte_l.requires_grad_(True)
                    ms_ssim_loss = MSSSIM()
                    #ms_ssim_loss =((1 - pytorch_msssim.msssim(x, recon_x))) / 2
                    #ms_ssim_out = ms_ssim_loss(output, target)
                    ms_ssim_out =MSloss.msssim( target,output)
                    ms_ssim_out.requires_grad_(True)
                    ms_ssim_value = ms_ssim_out.item()
                    G_l_t = 1. * inte_l + 1. * grad_l+1. * ms_ssim_out
                    # Train generator
                    G_l_t.requires_grad_(True)
                    writer.add_scalar('梯度约束', grad_l.item(), global_step=epoch_i)
                    writer.add_scalar('强度约束', inte_l.item(), global_step=epoch_i)
                    writer.add_scalar('MS_SSIM', ms_ssim_value, global_step=epoch_i)
                    writer.add_scalar('总损失', G_l_t, global_step=epoch_i)
                    writer.add_scalar('psnr', test_acc_1, global_step=epoch_i)

                    if ((f % 50 == 0) and (i % 10 == 0)):
                        print("epoch/120:",epoch_i,"  ",j,"/",len(video) - fp,"    ",G_l_t.item())
                    G_l_t.backward()
                    optimizer.step()
            # 打印训练进度
                # if (i % 10 == 0):
                #     print('[%d,loss: %.3f]' %( i + 1, G_l_t.item()))
                #print(scheduler.get_last_lr())
                runningLoss.append(G_l_t.clone().detach().item())
                
            # 保存断点
            checkpoint = {"model_state_dict": model.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict(),
                          "epoch": epoch_i}
            path_checkpoint = "/home/nitr/Devansh/astnet/astnet_all/checkpoint/{}_model.pth".format(epoch_i)
            torch.save(checkpoint, path_checkpoint)
        scheduler.step()
        print(scheduler.get_last_lr())
        print("loss ", np.average(runningLoss))

    torch.save(model.state_dict(), "model_test1.pth")

if __name__ == '__main__':
    args = parse_args()
    update_config(config, args)
    train()
