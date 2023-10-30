import sys
import os
import warnings
import requests
import argparse

import torch
import torch.nn.functional as F
import numpy as np
import glob
import tqdm

import matplotlib.pyplot as plt
from PIL import Image
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

sys.path.append('.')
from util.ddp_utils import DatasetTest
from util import ddp_utils
from xdecoder.BaseModel import BaseModel
from xdecoder import build_model
from util.distributed import init_distributed
from util.arguments import load_opt_from_config_files
from collections import defaultdict
from scipy.ndimage.interpolation import zoom
import imgviz
import random


imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

def save_colored_mask(mask, save_path):
    lbl_pil = Image.fromarray(mask.astype(np.uint8), mode='P')
    color_map = imgviz.label_colormap()
    lbl_pil.putpalette(color_map.flatten())
    lbl_pil.save(save_path)


def get_args_parser():
    parser = argparse.ArgumentParser('COCO panoptic segmentation', add_help=False)
    parser.add_argument('--ckpt_path', type=str, help='path to ckpt', default='/mnt/paintercoco/output_dir_2/')
    parser.add_argument('--model', type=str, help='dir to ckpt',
                        default='painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1')
    parser.add_argument('--prompt', type=str, help='prompt image in train set',
                        default='/mnt/XN-Net/data/npy/imgs/Mri_amos_amos_0580-017.npy')
    parser.add_argument('--input_size', type=int, default=448)

    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--conf_files',
                        default="/mnt/paintercoco/configs/seem/seem_dino_lang.yaml",
                        metavar="FILE",
                        help='path to config file', )
    parser.add_argument('--support_size', default=8)
    return parser.parse_args()


def prepare_model():

    opt = load_opt_from_config_files(args.conf_files)
    opt = init_distributed(opt)

    model = BaseModel(opt, build_model(opt)).cuda()
    if os.path.exists(os.path.join(args.ckpt_path, 'checkpoint-0.pth')):
        model_dict = model.state_dict()
        checkpoint = torch.load(os.path.join(args.ckpt_path, 'checkpoint-0.pth'))
        for k, v in checkpoint['model'].items():
            if k in model_dict.keys():
                model_dict[k] = v
    
        model.load_state_dict(model_dict)
        print("load success")
    return model


def run_with_multi_images(img2, tgt2, img1, model, device, out_dir, idx):
    ######## img2为参考图构成的tensors，拼接在一起 ； tgt2 为reference mask 构成的tensors，拼接在一起
    x = torch.tensor(img1)
    x = x.unsqueeze(dim=0)

    ######## ref_mask多加一个维度 ########################
    support_size, h, w = tgt2.shape
    ref_masks = torch.zeros((support_size, 10, h, w))

    for i_s in range(support_size):
        tgt_1_list = torch.unique(tgt2[i_s])

        for i_d, choi in enumerate(tgt_1_list):
            if choi != 0:
                ref_masks[i_s, i_d, :, :] = torch.tensor(tgt2[i_s] == choi, dtype=torch.uint8)
    
    y_pos = model(x.float().to(device), ref_masks.float().to(device), img2.float().to(device), ref_masks.float().to(device), mode='test')
    y_pos = y_pos.sigmoid()

    save_mask = torch.zeros((h, w), dtype=torch.uint8)
    for id, choi in enumerate(tgt_1_list):
        if choi !=0:
            print((y_pos[0][id]>0.5).sum())
            save_mask[y_pos[0][id]>0.5] = torch.tensor(choi, dtype=torch.uint8)
            mask = (y_pos[0][id]>0.5)
            mask[y_pos[0][id]>0.5] = torch.tensor(choi, dtype=torch.uint8)
            mask = mask.cpu().numpy().astype(np.uint8)

            save_colored_mask(mask, os.path.join(out_dir,'{}_{}_.png'.format(idx,choi)))
    save_colored_mask(np.array(save_mask), os.path.join(out_dir, 'final_{}.png'.format(idx)))


if __name__ == '__main__':
    dataset_dir = "datasets/"
    args = get_args_parser()
    args = ddp_utils.init_distributed_mode(args)
    device = torch.device("cuda")

    models_painter = prepare_model()
    print('Model loaded.')

    device = torch.device("cuda")
    models_painter.to(device)

    # load the shared prompt image pair
    gt_path = "/mnt/XN-Net/data/npy/gts/"
    img_path = "/mnt/XN-Net/data/npy/imgs/"

    out_dir = "./eval_dir_new/"
    gt_dir = out_dir + '/gts/'
    img_dir = out_dir + '/imgs/'

    if not os.path.exists(gt_dir):
        os.makedirs(gt_dir)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    f_val_list = open("/mnt/paintercoco/data/medical_dataset/split/val.txt", 'r').readlines()
    f_train_list = open("/mnt/paintercoco/data/medical_dataset/split/train.txt", 'r').readlines()

    gt_path_files_val = [
                os.path.join(gt_path, file.strip())
                for file in f_val_list
                if os.path.isfile(os.path.join(img_path, file.strip()))
    ]

    gt_path_files_train = [
                os.path.join(gt_path, file.strip())
                for file in f_train_list
                if os.path.isfile(os.path.join(img_path, file.strip()))
    ]

    dataset_dict_val = defaultdict(list)
    dataset_dict_train = defaultdict(list)

    models_painter.eval()

    for gt_file in gt_path_files_val:
        if 'acdc' in gt_file:
            dataset_dict_val['acdc'].append(gt_file)
        elif 'amos' in gt_file:
            dataset_dict_val['amos'].append(gt_file)
        elif 'CHAOS' in gt_file:
            dataset_dict_val['CHAOS'].append(gt_file)
        elif 'MMWHS' in gt_file:
            dataset_dict_val['MMWHS'].append(gt_file)
        elif 'protaste' in gt_file:
            dataset_dict_val['protaste'].append(gt_file)
        elif 'Task05' in gt_file:
            dataset_dict_val['Task05'].append(gt_file)
        elif 'cardic' in gt_file:
            dataset_dict_val['cardic'].append(gt_file)
        elif 'caridc' in gt_file:
            dataset_dict_val['caridc'].append(gt_file)
        elif 'MSDHEART' in gt_file:
            dataset_dict_val['MSDHEART'].append(gt_file)
    
    for gt_file in gt_path_files_train:
        if 'acdc' in gt_file:
            dataset_dict_train['acdc'].append(gt_file)
        elif 'amos' in gt_file:
            dataset_dict_train['amos'].append(gt_file)
        elif 'CHAOS' in gt_file:
            dataset_dict_train['CHAOS'].append(gt_file)
        elif 'MMWHS' in gt_file:
            dataset_dict_train['MMWHS'].append(gt_file)
        elif 'protaste' in gt_file:
            dataset_dict_train['protaste'].append(gt_file)
        elif 'Task05' in gt_file:
            dataset_dict_train['Task05'].append(gt_file)
        elif 'cardic' in gt_file:
            dataset_dict_train['cardic'].append(gt_file)
        elif 'caridc' in gt_file:
            dataset_dict_train['caridc'].append(gt_file)
        elif 'MSDHEART' in gt_file:
            dataset_dict_train['MSDHEART'].append(gt_file)

    
    ########## 在训练集中抽取一些图像作为support_set #########################
    support_imgs = []
    support_labels = []

    img2_dataset = args.prompt.split("_")[1]
    support_set = random.sample(dataset_dict_train[img2_dataset], args.support_size)
    for idx, gt_file in enumerate(dataset_dict_train[img2_dataset]):
        img2_1024 = np.load(
                gt_file.replace("gts", "imgs"), "r", allow_pickle=True
            )
        if len(img2_1024.shape) == 3:
            x, y, _ = img2_1024.shape
            img2_224 = zoom(img2_1024, (448 / x, 448 / y, 1), order=0)
            img2_224 = np.transpose(img2_224, (2, 0, 1))  ### (3, H, W)
            img2_224 = torch.tensor(img2_224)
        else:
            x, y = img2_1024.shape
            img2_224 = zoom(img2_1024, (448 / x, 448 / y), order=0)
            img2_224 = torch.tensor(img2_224).unsqueeze(0).repeat(3, 1, 1)  #### （ 3，448，448）
        
        support_imgs.append(img2_224)

        gt2_file = np.load(
                    gt_file, "r", allow_pickle=True
            )
        
        gt2_file = Image.fromarray(np.uint8(gt2_file)).resize((448, 448), Image.NEAREST)
        save_colored_mask(np.array(gt2_file), 'gt2.png')
        gt2_file = torch.tensor(np.array(gt2_file))  ####（448，448）

        support_labels.append(gt2_file)
    
    support_img_tensor = torch.stack(support_imgs, dim=0)  ### (support_size, 3, 448, 448)
    support_labels_tensor = torch.stack(support_labels, dim=0)  ### (support_size, 448, 448)

    for idx, gt_file in enumerate(dataset_dict_val[img2_dataset]):
        if gt_file.replace('gts', 'imgs') != args.prompt:
            img1_1024 = np.load(
                gt_file.replace('gts', 'imgs'), "r", allow_pickle=True
            )
            if len(img1_1024.shape) == 3:
                x, y, _ = img1_1024.shape
                img1_224 = zoom(img1_1024, (448 / x, 448 / y, 1), order=0)
                img1_224 = np.transpose(img1_224, (2, 0, 1))  ### (3, H, W)
                img1_224 = torch.tensor(img1_224)
            else:
                x, y = img1_1024.shape
                img1_224 = zoom(img1_1024, (448 / x, 448 / y), order=0)
                img1_224 = torch.tensor(img1_224).unsqueeze(0).repeat(3, 1, 1)
            
            gt1_file = np.load(
                    gt_file.replace("imgs", "gts"), "r", allow_pickle=True
            )
            gt1_file = Image.fromarray(np.uint8(gt1_file)).resize((448, 448), Image.NEAREST)
            save_colored_mask(np.array(gt1_file), os.path.join(gt_dir, 'gt{}.png'.format(idx)))
            gt1_file = torch.tensor(np.array(gt1_file))

            torch.manual_seed(2)
            run_with_multi_images(support_img_tensor, support_labels_tensor, img1_224, models_painter, device, img_dir, idx)