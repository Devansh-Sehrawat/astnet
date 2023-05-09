import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import scipy.signal as signal

import glob
import collections
import math
import pathlib
import warnings
from itertools import repeat
from types import FunctionType
from typing import Any, BinaryIO, List, Optional, Tuple, Union
import torch
from PIL import Image, ImageColor, ImageDraw, ImageFont
import cv2 
import csv


def draw_roc_curve(fpr, tpr, auc, psnr_dir):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")

    plt.savefig(os.path.join(psnr_dir, "auroc.png"))
    plt.close()

def draw_prc_curve(rec, prc, auc, psnr_dir):
    plt.figure()
    plt.plot(rec, prc, color='darkorange', lw=2, label='PRC curve (area = %0.4f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision Recall example')
    plt.legend(loc="lower right")

    plt.savefig(os.path.join(psnr_dir, "auprc.png"))
    plt.close()

def nonzero_intervals(vec):
    '''
    Find islands of non-zeros in the vector vec
    '''
    if len(vec) == 0:
        return []
    elif not isinstance(vec, np.ndarray):
        vec = np.array(vec)

    tmp1 = (vec == 0) * 1
    tmp = np.diff(tmp1)
    edges, = np.nonzero(tmp)
    edge_vec = [edges + 1]

    if vec[0] != 0:
        edge_vec.insert(0, [0])
    if vec[-1] != 0:
        edge_vec.append([len(vec)])
    edges = np.concatenate(edge_vec)
    return zip(edges[::2], edges[1::2])


def save_evaluation_curves(scores, labels, curves_save_path, video_frame_nums):
    """
    Draw anomaly score curves for each video and the overall ROC figure.
    """
    if not os.path.exists(curves_save_path):
        os.mkdir(curves_save_path)

    scores = scores.flatten()
    labels = labels.flatten()

    scores_each_video = {}
    labels_each_video = {}

    start_idx = 0
    for video_id in range(len(video_frame_nums)):
        scores_each_video[video_id] = scores[start_idx:start_idx + video_frame_nums[video_id]]
        scores_each_video[video_id] = signal.medfilt(scores_each_video[video_id], kernel_size=17)
        labels_each_video[video_id] = labels[start_idx:start_idx + video_frame_nums[video_id]]

        start_idx += video_frame_nums[video_id]

    truth = []
    preds = []
    for i in range(len(scores_each_video)):
        truth.append(labels_each_video[i])
        preds.append(scores_each_video[i])

    truth = np.concatenate(truth, axis=0)
    preds = np.concatenate(preds, axis=0)
    fpr, tpr, roc_thresholds = roc_curve(truth, preds, pos_label=1)
    prc, rec, prc_thresholds = precision_recall_curve(truth, preds, pos_label=1)
    auroc = auc(fpr, tpr)
    auprc = auc(rec, prc)
	
    
    
    #print(max(preds))
    #print(min(preds))
    #print(roc_thresholds)
    
    # Find the optimal threshold
    gmean = np.sqrt(tpr * (1 - fpr))

    index = np.argmax(gmean)
    thresholdOpt = round(roc_thresholds[index], ndigits = 4)
    gmeanOpt = round(gmean[index], ndigits = 4)

    # draw ROC figure
    draw_roc_curve(fpr, tpr, auroc, curves_save_path)
    draw_prc_curve(rec, prc, auprc, curves_save_path)
    
    filename = os.path.join(curves_save_path,"values.csv")
    rows = zip(fpr,tpr,roc_thresholds,prc,rec,prc_thresholds)
    
    with open(filename, 'w') as csvfile: 
    # creating a csv writer object 
        csvwriter = csv.writer(csvfile) 
        csvwriter.writerow(['fpr','tpr','roc_thresh','prc','rec','prc_thresh'])
        
        for row in rows:
            csvwriter.writerow(row)        
     
    for i in sorted(scores_each_video.keys()):
        plt.figure()

        x = range(0, len(scores_each_video[i]))
        plt.xlim([x[0], x[-1] + 5])

        # anomaly scores
        plt.plot(x, scores_each_video[i], color="blue", lw=2, label="Anomaly Score")

        # abnormal sections
        lb_one_intervals = nonzero_intervals(labels_each_video[i])
        for idx, (start, end) in enumerate(lb_one_intervals):
            plt.axvspan(start, end, alpha=0.5, color='red',
                        label="_" * idx + "Anomaly Intervals")

        plt.xlabel('Frames Sequence')
        plt.title('Test video #%d' % (i + 1))
        plt.legend(loc="upper left")
        plt.savefig(os.path.join(curves_save_path, "anomaly_curve_%d.png" % (i + 1)))
        plt.close()

    return auroc, auprc, thresholdOpt
    
def bbSave(frame_bbox,frame_score,thresh,dataset_name,bb_save_path,METADATA):

    if not os.path.exists(bb_save_path):
        os.mkdir(bb_save_path)

    
    video_dir_list = glob.glob(os.path.join("/home/nitr/Devansh/hf2vad/data",dataset_name,"testing/frames", '*'))
    video_dir_list.sort()
    video_dir_list = [ x for x in video_dir_list if "gt" not in x ]
    frame_bbox = frame_bbox.astype('int32')
    
    start_idx = 0
    for cur_video_id in range(METADATA[dataset_name]["testing_video_num"]):
        
        save_dir = os.path.join(bb_save_path,(video_dir_list[cur_video_id]).split('/')[-1])
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
            
        frame_dir_list = glob.glob(os.path.join(video_dir_list[cur_video_id],'*'))
        frame_dir_list.sort()
        frame_dir_list = frame_dir_list[4:]
        
        for i,_ in enumerate(frame_dir_list):
            im1 = cv2.imread(frame_dir_list[i])
            cur_img_name = (frame_dir_list[i]).split('/')[-1]
            im = torch.from_numpy(np.transpose(im1,[2,0,1]))
#             video_frame_path = os.path.join(of_save_dir, (frame_dir_list[i]).split('/')[-2])
            
            if frame_score[i+start_idx] >= thresh:
                imb=draw_bounding_boxes(im, (torch.from_numpy(frame_bbox[i+start_idx])).unsqueeze(0), width=3, colors=(255,255,0))
            else:
                imb = im1
            
            cv2.imwrite(os.path.join(save_dir,cur_img_name+'.png'),imb)
            
            
        start_idx += METADATA[dataset_name]["testing_frames_cnt"][cur_video_id] - 4

    
def draw_bounding_boxes(
    image: torch.Tensor,
    boxes: torch.Tensor,
    labels: Optional[List[str]] = None,
    colors: Optional[Union[List[Union[str, Tuple[int, int, int]]], str, Tuple[int, int, int]]] = None,
    fill: Optional[bool] = False,
    width: int = 1,
    font: Optional[str] = None,
    font_size: Optional[int] = None,
) -> torch.Tensor:

    """
    Draws bounding boxes on given image.
    The values of the input image should be uint8 between 0 and 255.
    If fill is True, Resulting Tensor should be saved as PNG image.
    Args:
        image (Tensor): Tensor of shape (C x H x W) and dtype uint8.
        boxes (Tensor): Tensor of size (N, 4) containing bounding boxes in (xmin, ymin, xmax, ymax) format. Note that
            the boxes are absolute coordinates with respect to the image. In other words: `0 <= xmin < xmax < W` and
            `0 <= ymin < ymax < H`.
        labels (List[str]): List containing the labels of bounding boxes.
        colors (color or list of colors, optional): List containing the colors
            of the boxes or single color for all boxes. The color can be represented as
            PIL strings e.g. "red" or "#FF00FF", or as RGB tuples e.g. ``(240, 10, 157)``.
            By default, random colors are generated for boxes.
        fill (bool): If `True` fills the bounding box with specified color.
        width (int): Width of bounding box.
        font (str): A filename containing a TrueType font. If the file is not found in this filename, the loader may
            also search in other directories, such as the `fonts/` directory on Windows or `/Library/Fonts/`,
            `/System/Library/Fonts/` and `~/Library/Fonts/` on macOS.
        font_size (int): The requested font size in points.
    Returns:
        img (Tensor[C, H, W]): Image Tensor of dtype uint8 with bounding boxes plotted.
    """

#     if not torch.jit.is_scripting() and not torch.jit.is_tracing():
#         _log_api_usage_once(draw_bounding_boxes)
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Tensor expected, got {type(image)}")
    elif image.dtype != torch.uint8:
        raise ValueError(f"Tensor uint8 expected, got {image.dtype}")
    elif image.dim() != 3:
        raise ValueError("Pass individual images, not batches")
    elif image.size(0) not in {1, 3}:
        raise ValueError("Only grayscale and RGB images are supported")
#         make changes here for multiple boxes
    elif (boxes[:,0] > boxes[:,2]).any() or (boxes[:,1] > boxes[:,3]).any():
        raise ValueError(
            "Boxes need to be in (xmin, ymin, xmax, ymax) format. Use torchvision.ops.box_convert to convert them"
        )

    num_boxes = boxes.shape[0]

    if num_boxes == 0:
        warnings.warn("boxes doesn't contain any box. No box was drawn")
        return image

    if labels is None:
        labels: Union[List[str], List[None]] = [None] * num_boxes  # type: ignore[no-redef]
    elif len(labels) != num_boxes:
        raise ValueError(
            f"Number of boxes ({num_boxes}) and labels ({len(labels)}) mismatch. Please specify labels for each box."
        )

    if colors is None:
        colors = _generate_color_palette(num_boxes)
    elif isinstance(colors, list):
        if len(colors) < num_boxes:
            raise ValueError(f"Number of colors ({len(colors)}) is less than number of boxes ({num_boxes}). ")
    else:  # colors specifies a single color for all boxes
        colors = [colors] * num_boxes

    colors = [(ImageColor.getrgb(color) if isinstance(color, str) else color) for color in colors]

    if font is None:
        if font_size is not None:
            warnings.warn("Argument 'font_size' will be ignored since 'font' is not set.")
        txt_font = ImageFont.load_default()
    else:
        txt_font = ImageFont.truetype(font=font, size=font_size or 10)

    # Handle Grayscale images
    if image.size(0) == 1:
        image = torch.tile(image, (3, 1, 1))

    ndarr = image.permute(1, 2, 0).cpu().numpy()
    img_to_draw = Image.fromarray(ndarr)
    img_boxes = boxes.to(torch.int64).tolist()

    if fill:
        draw = ImageDraw.Draw(img_to_draw, "RGBA")
    else:
        draw = ImageDraw.Draw(img_to_draw)

    for bbox, color, label in zip(img_boxes, colors, labels):  # type: ignore[arg-type]
        if fill:
            fill_color = color + (100,)
            draw.rectangle(bbox, width=width, outline=color, fill=fill_color)
        else:
            draw.rectangle(bbox, width=width, outline=color)

        if label is not None:
            margin = width + 1
            draw.text((bbox[0] + margin, bbox[1] + margin), label, fill=color, font=txt_font)

    return (np.array(img_to_draw)).astype('uint8')
#     return torch.from_numpy(np.array(img_to_draw)).permute(2, 0, 1).to(dtype=torch.uint8)

