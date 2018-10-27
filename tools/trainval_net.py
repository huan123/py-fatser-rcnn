# -*- coding: utf-8 -*-
# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Zheqi He, Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------
#使用voc训练时配置参数
# --weight
# data/imagenet_weights/vgg16.ckpt
# --imdb
# voc_2007_trainval
# --imdbval
# voc_2007_test
# --iters
# 110000
# --cfg
# experiments/cfgs/vgg16.yml
# --net
# vgg16
# --set
# ANCHOR_SCALES
# [8,16,32]

#使用coco
# --weight
# data/imagenet_weights/vgg16.ckpt
# --imdb
# coco_2014_train+coco_2014_valminusminival
# --imdbval
# coco_2014_minival
# --iters
# 490000
# --cfg
# experiments/cfgs/vgg16.yml
# --net
# vgg16
# --set
# ANCHOR_SCALES
# [4,8,16,32]

#自己的
# --weight
# data/imagenet_weights/vgg16.ckpt
# --imdb
# coco_2014_train+coco_2014_valminusminival
# --imdbval
# coco_2014_minival
# --iters
# 490
# --cfg
# experiments/cfgs/vgg16.yml
# --net
# vgg16
# --set
# ANCHOR_SCALES
# [4,8,16,32]
#


#train_net.py：使用fast rcnn，训练自己数据集的网络模型
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
#sys.path.insert(0, '/Users/huan/code/PycharmProjects/tf-faster-rcnn/lib')
import tools._init_paths
from model.train_val import get_training_roidb, train_net
from model.config import cfg, cfg_from_file, cfg_from_list, get_output_dir, get_output_tb_dir
from datasets.factory import get_imdb
import datasets.imdb
import argparse
import pprint
import numpy as np
import sys
import argparse
import tensorflow as tf
from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
from nets.mobilenet_v1 import mobilenetv1

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default=None, type=str)
  parser.add_argument('--weight', dest='weight',
                      help='initialize with pretrained model weights',
                      type=str)
  parser.add_argument('--imdb', dest='imdb_name',
                      help='dataset to train on',
                      default='voc_2007_trainval', type=str)
  parser.add_argument('--imdbval', dest='imdbval_name',
                      help='dataset to validate on',
                      default='voc_2007_test', type=str)
  parser.add_argument('--iters', dest='max_iters',
                      help='number of iterations to train',
                      default=70000, type=int)
  parser.add_argument('--tag', dest='tag',
                      help='tag of the model',
                      default=None, type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152, mobile',
                      default='res50', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)

  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)

  args = parser.parse_args()
  return args


def combined_roidb(imdb_names):
  """
  Combine multiple roidbs
  """
  #roidb是一个列表，列表的长度是读取的图片文件个数的两倍（注意，经过了图片翻转），列表中的每一个元素都是一个字典，
  #而且字典的key包括：'boxes' 、'gt_classes'、'gt_overlaps'、'flipped'（原始图片取值为：False，翻转之后的图片取值为：True）、'image'、'width'、'height'、'max_classes'、'max_overlaps'。
  def get_roidb(imdb_name):
    imdb = get_imdb(imdb_name)
    print('Loaded dataset `{:s}` for training'.format(imdb.name))
    imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
    print('Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD))
    roidb = get_training_roidb(imdb)
    return roidb

  roidbs = [get_roidb(s) for s in imdb_names.split('+')]
  roidb = roidbs[0]
  if len(roidbs) > 1:
    for r in roidbs[1:]:
      roidb.extend(r)
    tmp = get_imdb(imdb_names.split('+')[1])
    imdb = datasets.imdb.imdb(imdb_names, tmp.classes)
  else:
    imdb = get_imdb(imdb_names)
  return imdb, roidb


if __name__ == '__main__':
  args = parse_args()

  print('Called with args:')
  print(args)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  #print.pprint(cfg)
  print(cfg)

  np.random.seed(cfg.RNG_SEED)

  # train set
  imdb, roidb = combined_roidb(args.imdb_name)
  print('{:d} roidb entries'.format(len(roidb)))

  #todo 为了生成pkl，先将除了训练以外的注释掉
  # output directory where the models are saved
  output_dir = get_output_dir(imdb, args.tag)
  print('Output will be saved to `{:s}`'.format(output_dir))

  # tensorboard directory where the summaries are saved during training
  tb_dir = get_output_tb_dir(imdb, args.tag)
  print('TensorFlow summaries will be saved to `{:s}`'.format(tb_dir))

  # also add the validation set, but with no flipping images
  orgflip = cfg.TRAIN.USE_FLIPPED
  cfg.TRAIN.USE_FLIPPED = False
  #todo 测试被注释

  _, valroidb = combined_roidb(args.imdbval_name)
  print('{:d} validation roidb entries'.format(len(valroidb)))
  cfg.TRAIN.USE_FLIPPED = orgflip

  # load network
  if args.net == 'vgg16':
    net = vgg16()
  elif args.net == 'res50':
    net = resnetv1(num_layers=50)
  elif args.net == 'res101':
    net = resnetv1(num_layers=101)
  elif args.net == 'res152':
    net = resnetv1(num_layers=152)
  elif args.net == 'mobile':
    net = mobilenetv1()
  else:
    raise NotImplementedError

  train_net(net, imdb, roidb, valroidb, output_dir, tb_dir,
            pretrained_model=args.weight,
            max_iters=args.max_iters)
