from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from MSloss import MSSSIM
import io
import logging
import os
import MSloss
import numpy as np
import tqdm

import torch
import torch.nn as nn

from utils import utils
import cv2
from PIL import Image
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def decode_input(input, train=True):
    video = input['video']
    video_name = input['video_name']

    if train:
        inputs = video[:-1]
        target = video[-1]
        return inputs, target
        # return video, video_name
    else:   # TODO: bo sung cho test
        return video, video_name
def visualize_grid_attention_v2(img, save_path, attention_mask, ratio=1, cmap="jet", save_image=False,
                             save_original_image=False, quality=200):
    # img = Image.open(img, mode='r')
    img_h, img_w = img.size[0], img.size[1]
    plt.subplots(nrows=1, ncols=1, figsize=(0.02 * img_h, 0.02 * img_w))

    # scale the image
    img_h, img_w = int(img.size[0] * ratio), int(img.size[1] * ratio)
    img = img.resize((img_h, img_w))
    plt.imshow(img, alpha=1)
    plt.axis('off')

    # normalize the attention map
    mask = cv2.resize(attention_mask, (img_h, img_w))
    normed_mask = mask / mask.max()
    normed_mask = (normed_mask * 255).astype('uint8')
    plt.imshow(normed_mask, alpha=0.5, interpolation='nearest', cmap=cmap)

    if save_image:
        # build save path
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        img_name =  "with_attention.jpg"
        img_with_attention_save_path = os.path.join(save_path, img_name)

        # pre-process and save image
        print("save image to: " + save_path + " as " + img_name)
        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(img_with_attention_save_path, dpi=quality)

    if save_original_image:
        # build save path
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        # save original image file
        print("save original image at the same time")
        img_name = "_original.jpg"
        original_image_save_path = os.path.join(save_path, img_name)
        img.save(original_image_save_path, quality=quality)

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
def inference(config, data_loader, model):
    intensity_loss = Intensity_Loss().cuda()
    gradient_loss = Gradient_Loss(3).cuda()
    fig = plt.figure("Image")
    manager = plt.get_current_fig_manager()
    manager.window.setGeometry = (550, 200, 600, 500)
    plt.xlabel('frames')
    plt.ylabel('psnr')
    plt.title('psnr curve')
    plt.grid(ls='--')
    cv2.namedWindow('target frames', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('target frames', 384, 384)
    cv2.moveWindow("target frames", 100, 100)
    cv2.namedWindow('difference map', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('difference map', 384, 384)
    cv2.moveWindow('difference map', 100, 550)

    loss_func_mse = nn.MSELoss(reduction='none')

    model.eval()
    psnr_list = []
    ef = config.MODEL.ENCODED_FRAMES
    df = config.MODEL.DECODED_FRAMES
    fp = ef + df
    # img_test=cv2.imread('/media/test/02ca50dc-830d-4673-9e13-afa0e5e097a8/Python_test/download/astnet-main/result/4.jpg', 1)[:, :, ::-1]
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            print('[{}/{}]'.format(i+1, len(data_loader)))
            psnr_video = []
            video, video_name = decode_input(input=data, train=False)
            video = [frame.to(device=config.GPUS[0]) for frame in video]

            name = video_name
            fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
            video_writer = cv2.VideoWriter(f'results/{name}_video.avi', fourcc, 30, (600,430))
            curve_writer = cv2.VideoWriter(f'results/{name}_curve.avi', fourcc, 30, (600, 430))
            js = []
            plt.clf()
            ax = plt.axes(xlim=(0, len(video)), ylim=(10, 45))
            line, = ax.plot([], [], '-b')
            heatmap_writer = cv2.VideoWriter(f'results/{name}_heatmap.avi', fourcc, 30, (600,430))
            optmap1_writer = cv2.VideoWriter(f'results/{name}_optmap1.avi', fourcc, 30, (600, 430))
            optmap2_writer = cv2.VideoWriter(f'results/{name}_optmap1.avi', fourcc, 30, (600, 430))
            psnrs = []
            j=0


            for f in tqdm.tqdm(range(len(video) - fp)):
                j=j+1
                inputs = video[f:f + fp]
                output = model(inputs)

                target = video[f + fp:f + fp + 1][0]  # 预测的真值
                target1=target.squeeze(0)
                last=video[f + fp -1:f + fp][0]  #输入的最后一帧真值
                last1=last.squeeze(0)
                output_pred=output.squeeze(0)
                #last1=last1.resize(3,224,224)
                #img2=last1*0.9+img_test
                #cv2.imwrite('/media/test/02ca50dc-830d-4673-9e13-afa0e5e097a8/Python_test/download/astnet-main/result/4.jpg', img2)


                # compute PSNR for each frame
                mse_imgs = torch.mean(loss_func_mse((output[0] + 1) / 2, (target[0] + 1) / 2)).item()
                psnr = utils.psnr_park(mse_imgs)
                psnr_video.append(psnr)
                psnrs.append(float(psnr))

                grad_l = gradient_loss(output, target)  # 梯度约束
                grad_l.requires_grad_(True)

                inte_l = intensity_loss(output, target)  # 强度约束
                inte_l.requires_grad_(True)
                # ms_ssim_loss =((1 - pytorch_msssim.msssim(x, recon_x))) / 2
                # ms_ssim_out = ms_ssim_loss(output, target)
                ms_ssim_out = MSloss.msssim(target, output)
                ms_ssim_out.requires_grad_(True)
                ms_ssim_value = ms_ssim_out.item()
                G_l_t = 1. * inte_l + 1. * grad_l + 1. * ms_ssim_out

                js.append(j)
                line.set_xdata(js)  # This keeps the existing figure and updates the X-axis and Y-axis data,
                line.set_ydata(psnrs)  # which is faster, but still not perfect.
                plt.pause(0.001)  # show curve
                buffer = io.BytesIO()  # Write curve frames from buffer.
                fig.canvas.print_png(buffer)
                buffer.write(buffer.getvalue())
                curve_img = np.array(Image.open(buffer))[..., (2, 1, 0)]
                curve_writer.write(curve_img)

                cv2_pred = ((output_pred + 1) * 127.5).permute(1, 2, 0).contiguous().cpu()
                cv2_frame1 = ((target1 + 1) * 127.5).permute(1, 2, 0).contiguous().cpu()
                cv2_frame=cv2_frame1.numpy().astype('uint8')
                cv2.imshow('target frames', cv2_frame)
                cv2.waitKey(1)  # show video
                video_writer.write(cv2_frame)  # Write original video frames.
                cv2_frame_pred = cv2_pred.numpy().astype('uint8')
                cv2.imshow('pred frames', cv2_frame_pred)
                cv2.waitKey(1)  # show video
                video_writer.write(cv2_frame_pred)  # Write original video frames.



                diff_map = torch.sum(torch.abs(output - target).squeeze(), 0)
                diff_map -= diff_map.min()  # Normalize to 0 ~ 255.
                diff_map /= diff_map.max()
                diff_map *= 255
                diff_map = diff_map.cpu().detach().numpy().astype('uint8')
                heat_map = cv2.applyColorMap(diff_map, cv2.COLORMAP_JET)
                cv2.imshow('difference map', heat_map)
                cv2.waitKey(1)
                heatmap_writer.write(heat_map)

                opt_map1 = torch.sum(torch.abs(target-last).squeeze(), 0)
                opt_map1 -= opt_map1.min()  # Normalize to 0 ~ 255.
                opt_map1 /= opt_map1.max()
                opt_map1 *= 255
                opt_map1 = opt_map1.cpu().detach().numpy().astype('uint8')
                heat_map1 = cv2.applyColorMap(opt_map1, cv2.COLORMAP_JET)
                cv2.imshow('opt map1:gt-gt', heat_map1)
                cv2.waitKey(1)
                optmap1_writer.write(opt_map1)

                opt_map2 = torch.sum(torch.abs(output - last).squeeze(), 0)
                opt_map2 -= opt_map2.min()  # Normalize to 0 ~ 255.
                opt_map2 /= opt_map2.max()
                opt_map2 *= 255
                opt_map2 = opt_map2.cpu().detach().numpy().astype('uint8')
                heat_map2 = cv2.applyColorMap(opt_map2, cv2.COLORMAP_JET)
                cv2.imshow('opt map2:predict-gt', heat_map2)
                cv2.waitKey(1)
                optmap2_writer.write(opt_map2)
                #flow-flow=(target-last)-(output - last)=target-output


            psnr_list.append(psnr_video)
    return psnr_list

