import argparse
import os
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

from experiment import Experiment

import faulthandler

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

faulthandler.enable()


"""
jsub -J stf-encoder -q qgpu -e error.txt -o output.txt python run.py --lr 1e-3 --num_workers 16 --batch_size 32 --epochs 200 --cuda --ngpu 4 --image_size 2720 3200 --patch_size 680 800 --patch_stride 200 --test_patch 680 800 --save_dir out --train_dir data --test_dir data
"""
import warnings
warnings.filterwarnings("ignore")

# 获取模型运行时必须的一些参数
parser = argparse.ArgumentParser(description='Acquire some parameters for fusion model')
parser.add_argument('--lr', type=float, default=1e-3, help='the initial learning rate')
parser.add_argument('--batch_size', type=int, default=8, help='input batch size for training')
parser.add_argument('--epochs', type=int, default=500, help='number of epochs to train')
parser.add_argument('--cuda', action='store_true', default=True,help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--num_workers', type=int, default=0, help='number of threads to load data')
parser.add_argument('--save_dir', type=Path, default=Path('./Tianjin/output'), help='the output directory')

# 获取对输入数据进行预处理时的一些参数
parser.add_argument('--train_dir', type=Path, default=Path('./Tianjin/data/train'), help='the training data directory')
parser.add_argument('--test_dir', type=Path, default=Path('./Tianjin/data/test'), help='the test data directory')
parser.add_argument('--image_size', type=int, nargs='+', default=[1200, 1200], help='the size of the coarse image (width, height)')
parser.add_argument('--patch_size', type=int, nargs='+',default=400, help='the fine image patch size for training model')
parser.add_argument('--patch_stride', type=int, nargs='+',default=200, help='the fine patch stride for image division')
parser.add_argument('--test_patch', type=int, nargs='+', default=40, help='the fine image patch size for fusion test')
# parser.add_argument('--image_size', type=int, nargs='+', default=[800, 800], help='the size of the coarse image (width, height)')
# parser.add_argument('--patch_size', type=int, nargs='+',default=400, help='the fine image patch size for training model')
# parser.add_argument('--patch_stride', type=int, nargs='+',default=400, help='the fine patch stride for image division')
# parser.add_argument('--test_patch', type=int, nargs='+', default=200, help='the fine image patch size for fusion test')
opt = parser.parse_args()

torch.manual_seed(2019)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(2019)

if opt.cuda and not torch.cuda.is_available():
    opt.cuda = False
else:
    cudnn.benchmark = True
    cudnn.deterministic = True

if __name__ == '__main__':
    experiment = Experiment(opt)
    if opt.epochs > 0:
        experiment.train(opt.train_dir, opt.patch_size, opt.patch_stride,
                         opt.batch_size, num_workers=opt.num_workers, epochs=opt.epochs)
    experiment.test(opt.test_dir, opt.test_patch, num_workers=opt.num_workers)
