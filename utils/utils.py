import os
import logging
import time
import math
import pickle
import numpy as np
from pathlib import Path
from sklearn import metrics
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

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



def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.DATASET
    model = cfg.MODEL.NAME
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    final_output_dir = root_output_dir / dataset / cfg_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / \
                          (cfg_name + '_' + time_str)
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)


def psnr_park(mse):
    return 10 * math.log10(1 / mse)


def anomaly_score(psnr, max_psnr, min_psnr):
    return (psnr - min_psnr) / (max_psnr - min_psnr)


def calculate_auc(config, psnr_list, mat, suffix):
    ef = config.MODEL.ENCODED_FRAMES
    df = config.MODEL.DECODED_FRAMES
    fp = ef + df  # number of frames to process

    scores = np.array([], dtype=np.float)
    labels = np.array([], dtype=np.int)

    for i in range(len(psnr_list)):
        score = anomaly_score(psnr_list[i], np.max(psnr_list[i]), np.min(psnr_list[i]))

        scores = np.concatenate((scores, score), axis=0)
        labels = np.concatenate((labels, mat[i][fp:]), axis=0)
    assert scores.shape == labels.shape, f'Ground truth has {labels.shape[0]} frames, BUT got {scores.shape[0]} detected frames!'
    fpr, tpr, roc_thresholds = metrics.roc_curve(labels, scores, pos_label=0)
    prc, rec, prc_thresholds = metrics.precision_recall_curve(labels, scores, pos_label=1)
    auroc = metrics.auc(fpr, tpr)
    auprc = metrics.auc(rec, prc)

    gmean = np.sqrt(tpr * (1 - fpr))

    index = np.argmax(gmean)
    thresholdOpt = round(roc_thresholds[index], ndigits = 4)
    gmeanOpt = round(gmean[index], ndigits = 4)

    eval_dir = os.path.join('./eval', (config.MODEL.NAME + config.DATASET.DATASET))
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)
    curves_save_path = os.path.join('./eval', (config.MODEL.NAME + config.DATASET.DATASET), 'anomaly_curves_%s' % suffix)
    if not os.path.exists(curves_save_path):
        os.mkdir(curves_save_path)



    # draw ROC figure
    draw_roc_curve(fpr, tpr, auroc, curves_save_path)
    draw_prc_curve(rec, prc, auprc, curves_save_path)
    
    # --------------if needed add csv path-----------------------
    # filename = os.path.join(curves_save_path,"values.csv")
    # rows = zip(fpr,tpr,roc_thresholds,prc,rec,prc_thresholds)
    
    # with open(filename, 'w') as csvfile: 
    # # creating a csv writer object 
    #     csvwriter = csv.writer(csvfile) 
    #     csvwriter.writerow(['fpr','tpr','roc_thresh','prc','rec','prc_thresh'])
        
    #     for row in rows:
    #         csvwriter.writerow(row)        


     
    #---------------- to do - saving individual curves--------------

    # for i in sorted(scores_each_video.keys()):
    #     plt.figure()

    #     x = range(0, len(scores_each_video[i]))
    #     plt.xlim([x[0], x[-1] + 5])

    #     # anomaly scores
    #     plt.plot(x, scores_each_video[i], color="blue", lw=2, label="Anomaly Score")

    #     # abnormal sections
    #     lb_one_intervals = nonzero_intervals(labels_each_video[i])
    #     for idx, (start, end) in enumerate(lb_one_intervals):
    #         plt.axvspan(start, end, alpha=0.5, color='red',
    #                     label="_" * idx + "Anomaly Intervals")

    #     plt.xlabel('Frames Sequence')
    #     plt.title('Test video #%d' % (i + 1))
    #     plt.legend(loc="upper left")
    #     plt.savefig(os.path.join(curves_save_path, "anomaly_curve_%d.png" % (i + 1)))
    #     plt.close()



    return auroc, auprc, fpr, tpr

