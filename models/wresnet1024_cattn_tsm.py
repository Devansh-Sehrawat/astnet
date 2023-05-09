import logging
import os.path

import torch
import torch.nn as nn
from networks.wider_resnet import wresnet
from networks.helper import ConvBnRelu, ConvTransposeBnRelu, initialize_weights

from networks.helper import TemporalShift, ChannelAttention
import einops
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision.transforms import Compose, Normalize, ToTensor
a=0
logger = logging.getLogger(__name__)
def preprocess_image(img: np.ndarray, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) -> torch.Tensor:
    preprocessing = Compose([
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])
    return preprocessing(img.copy()).unsqueeze(0)
def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.
    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")
    img=img.reshape(224,224,3)
    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

class ASTNet(nn.Module):
    def get_name(self):
        return self.model_name

    def __init__(self, config):
        super(ASTNet, self).__init__()
        frames = config.MODEL.ENCODED_FRAMES
        final_conv_kernel = config.MODEL.EXTRA.FINAL_CONV_KERNEL
        self.model_name = config.MODEL.NAME

        logger.info('=> ' + self.model_name + '_1024: (CATTN + TSM) - Ped2')

        self.wrn38 = wresnet(config, self.model_name, pretrained=True)

        channels = [4096, 2048, 1024, 512, 256, 128]
        
        # changing chanel 0 to 1 in conv_x2 

        self.conv_x8 = nn.Conv2d(channels[0] * frames, channels[1], kernel_size=1, bias=False)  # 4096 * frames -> 2048
        self.conv_x2 = nn.Conv2d(channels[4] * frames, channels[4], kernel_size=1, bias=False)  # 256 * frames -> 256
        self.conv_x1 = nn.Conv2d(channels[5] * frames, channels[5], kernel_size=1, bias=False)  # 128 * frames -> 128

        self.up8 = ConvTransposeBnRelu(channels[1], channels[2], kernel_size=2)   # 2048          -> 1024
        self.up4 = ConvTransposeBnRelu(channels[2] + channels[4], channels[3], kernel_size=2)   # 1024  +   256 -> 512
        self.up2 = ConvTransposeBnRelu(channels[3] + channels[5], channels[4], kernel_size=2)   # 512   +   128 -> 256

        self.tsm_left = TemporalShift(n_segment=4, n_div=16, direction='left')

        self.attn8 = ChannelAttention(channels[2])
        self.attn4 = ChannelAttention(channels[3])
        self.attn2 = ChannelAttention(channels[4])

        self.final = nn.Sequential(
            ConvBnRelu(channels[4], channels[5], kernel_size=1, padding=0),
            ConvBnRelu(channels[5], channels[5], kernel_size=3, padding=1),
            nn.Conv2d(channels[5], 3,
                      kernel_size=final_conv_kernel,
                      padding=1 if final_conv_kernel == 3 else 0,
                      bias=False)
        )

        initialize_weights(self.conv_x1, self.conv_x2, self.conv_x8)
        initialize_weights(self.up2, self.up4, self.up8)
        initialize_weights(self.attn2, self.attn4, self.attn8)
        initialize_weights(self.final)

    def forward(self, x):
        a=0
        x1s, x2s, x8s = [], [], []

        for xi in x:
            a=a+1
            x1, x2, x8 = self.wrn38(xi)

            x8s.append(x8)
            x2s.append(x2)
            x1s.append(x1)

        x8_1=torch.cat(x8s, dim=1)
        x8 = self.conv_x8(torch.cat(x8s, dim=1))
        x2 = self.conv_x2(torch.cat(x2s, dim=1))
        x1 = self.conv_x1(torch.cat(x1s, dim=1))

        print("convolution of the frames")
        print(x8.shape)
        print(x2.shape)
        print(x1.shape)

        left = self.tsm_left(x8)
        print("temporal shift")
        print(left.shape)

        x8 = x8 + left
        print("temporal concat spatial")
        print(x8.shape)


        x = self.up8(x8)   
        print("first deconv")                            # 2048          -> 1024, 24, 40
        print(x.shape)                         # 2048          -> 1024, 24, 40
        
        x = self.attn8(x)
        print("first attention")
        print(x.shape)

        """
        rgb_img = F.interpolate(xi, size=(224, 224), mode='bilinear').cpu()
        img = np.float32(rgb_img) / 255
        #input = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        feature_grad = x
        feature_weight = einops.reduce(feature_grad, 'b c h w -> b c', 'mean')
        grad_cam = x * feature_weight.unsqueeze(-1).unsqueeze(-1)  # (b c h w) * (b c 1 1) -> (b c h w)
        grad_cam = F.relu(torch.sum(grad_cam, dim=1)).unsqueeze(dim=1)  # (b c h w) -> (b h w) -> (b 1 h w)
        grad_cam = F.interpolate(grad_cam, size=(224, 224), mode='bilinear')  # (b 1 h w) -> (b 1 224 224) -> (224 224)
        grad_cam = grad_cam[0, 0, :].cpu()
        #img=x8s[0].cpu()
        cam_image = show_cam_on_image(img, grad_cam.detach().numpy())

        img_name=str(a)+'.jpg'
        path=os.path.join('/media/test/02ca50dc-830d-4673-9e13-afa0e5e097a8/Python_test/download/astnet-main/result',img_name)
        cv2.imwrite(path, cam_image)
        """
        print("first attention concat ",torch.cat([x2, x], dim=1).shape)
        x = self.up4(torch.cat([x2, x], dim=1))     # 1024 + 256    -> 512, 48, 80
        print("second deconv")
        print(x.shape)
        
        x = self.attn4(x)
        print("second attention")
        print(x.shape)

        print("second attention concat ",torch.cat([x1, x], dim=1).shape)
        x = self.up2(torch.cat([x1, x], dim=1))     # 512 + 128     -> 256, 96, 160
        print("third deconv")
        print(x.shape)
        
        x = self.attn2(x)
        print("third attention")
        print(x.shape)

        print("final o/p")
        x =self.final(x)
        print(x.shape)
        exit()
        return x
