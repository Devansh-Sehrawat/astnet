from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import tqdm

import torch
import torch.nn as nn

from utils import utils
import numpy as np
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)


def decode_input(input, train=True):
    video = input['video']
    video_name = input['video_name']
    # print(video_name)
    # exit(1)

    if train:
        inputs = video[:-1]
        target = video[-1]
        return inputs, target
        # return video, video_name
    else:   # TODO: bo sung cho test
        return video, video_name


def inference(config, data_loader, model):
    loss_func_mse = nn.MSELoss(reduction='none')

    model.eval()
    psnr_list = []
    ef = config.MODEL.ENCODED_FRAMES
    df = config.MODEL.DECODED_FRAMES
    fp = ef + df
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            print('[{}/{}]'.format(i+1, len(data_loader)))
            psnr_video = []

            video, video_name = decode_input(input=data, train=False)
            video = [frame.to(device=config.GPUS[0]) for frame in video]
            for f in tqdm.tqdm(range(len(video) - fp)):

                inputs = video[f:f + fp]
                output = model(inputs)
                target = video[f + fp:f + fp + 1][0]
                
                
                # if ((f % 50 == 0) and (i % 10 == 0)):
                #     plt.imshow(np.transpose(torch.squeeze(output.cpu(),dim=0).numpy(),[1,2,0]))
                #     plt.show()    
                #     plt.imshow(np.transpose(torch.squeeze(target.cpu(),dim=0).numpy(),[1,2,0]))
                #     plt.show()
                
                
                # compute PSNR for each frame
                mse_imgs = torch.mean(loss_func_mse((output[0] + 1) / 2, (target[0] + 1) / 2)).item()
                psnr = utils.psnr_park(mse_imgs)
                psnr_video.append(psnr)

            psnr_list.append(psnr_video)
    return psnr_list

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
