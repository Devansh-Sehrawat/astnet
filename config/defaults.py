from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from yacs.config import CfgNode as CN


_C = CN()

_C.OUTPUT_DIR = 'output'
_C.LOG_DIR = 'log'
_C.GPUS = (0, 1, 2, 3)
_C.WORKERS = 4
_C.PRINT_FREQ = 50
_C.SAVE_CHECKPOINT_FREQ = 5
_C.AUTO_RESUME = False
_C.PIN_MEMORY = True

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True  #在每个卷积操作前进预处理
_C.CUDNN.DETERMINISTIC = False #使用非确定性的优化算法，不能保证每次的结果都相同（导致结果的不可重复）
_C.CUDNN.ENABLED = True #使用CUDNN作为默认的深度神经网络库来加速卷积/池化等操作


# DATASET related params
_C.DATASET = CN()
_C.DATASET.ROOT = './datasets'
_C.DATASET.DATASET = 'ped2'
_C.DATASET.TRAINSET = 'train'
_C.DATASET.TESTSET = 'test'
_C.DATASET.NUM_INCHANNELS = 3
_C.DATASET.NUM_FRAMES = 4
_C.DATASET.FRAME_STEPS = 1
_C.DATASET.LOWER_BOUND = 500


# train
_C.TRAIN = CN()

_C.TRAIN.BATCH_SIZE_PER_GPU = 16
_C.TRAIN.SHUFFLE = True

_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 200
_C.TRAIN.RESUME = True
_C.TRAIN.CHECKPOINT = ''
_C.TRAIN.SAVEEVERY = 1

_C.TRAIN.OPTIMIZER = 'adam'

# sgd and
_C.TRAIN.MOMENTUM = 0.0 #不使用动量优化算法，即只考虑当前的梯度信息，不考虑之前的梯度信息
_C.TRAIN.WD = 0.0 #不使用L2正则化方法（权重衰减），会降低模型的拟合能力和泛化能力
_C.TRAIN.NESTEROV = False #不使用NESTEROV动量优化算法，每次更新参数时，只考虑当前的梯度信息和上一步的动量信息，而不考虑未来梯度信息

_C.TRAIN.LR_TYPE = 'linear'     # 'linear'  /   'step'  /   'multistep'
_C.TRAIN.LR = 0.0002
_C.TRAIN.LR_STEP = [40, 70]
_C.TRAIN.LR_FACTOR = 0.5


# test
_C.TEST = CN()

# size of images for each device
_C.TEST.BATCH_SIZE_PER_GPU = 1


# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = 'net'
_C.MODEL.INIT_WEIGHTS = True
_C.MODEL.PRETRAINED = '/media/test/02ca50dc-830d-4673-9e13-afa0e5e097a8/Python_test/download/astnet-main/datasets/seg_weights/wider_resnet38.pth.tar'
_C.MODEL.IMAGE_SIZE = [256, 256]  # width * height
_C.MODEL.MEMORY_SIZE = 3
_C.MODEL.ENCODED_FRAMES = 2
_C.MODEL.DECODED_FRAMES = 1
# _C.MODEL.SIGMA = 1.5


_C.MODEL.EXTRA = CN()
_C.MODEL.EXTRA.FINAL_CONV_KERNEL = 1


def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.freeze()


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)
