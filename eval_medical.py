import sys
import os
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
sys.path.append('.')
from util import ddp_utils
from xdecoder.BaseModel import BaseModel
from xdecoder import build_model
from util.distributed import init_distributed
from util.arguments import load_opt_from_config_files
from collections import defaultdict
from scipy.ndimage.interpolation import zoom
import imgviz

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


def save_colored_mask(mask, save_path):
    lbl_pil = Image.fromarray(mask.astype(np.uint8), mode='P')
    color_map = imgviz.label_colormap()
    lbl_pil.putpalette(color_map.flatten())
    lbl_pil.save(save_path)


def get_args_parser():
    parser = argparse.ArgumentParser('COCO panoptic segmentation', add_help=False)
    parser.add_argument('--ckpt_path', type=str, help='path to ckpt', default='/dataset/zhongqiaoyong/MedicalImages/mri/npy/imgs/Mri_amos_amos_0578-016.npy')
    parser.add_argument('--model', type=str, help='dir to ckpt',
                        default='painter_vit_large_patch16_input896x448_win_dec64_8glb_sl1')
    parser.add_argument('--prompt', type=str, help='prompt image in train set',
                        default='/data1/stare_imgs/stare/im0255.npy')
    parser.add_argument('--input_size', type=int, default=448)
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--conf_files',
                        default="/data1/paintercoco/configs/seem/seem_dino_lang.yaml",
                        metavar="FILE",
                        help='path to config file', )
    return parser.parse_args()


def prepare_model():
    opt = load_opt_from_config_files(args.conf_files)
    opt = init_distributed(opt)
    model = BaseModel(opt, build_model(opt)).cuda()

    if os.path.exists(os.path.join('/data1/output_dir_simple_1/', 'checkpoint-0.pth')):
        model_dict = model.state_dict()
        checkpoint = torch.load(os.path.join('/data1/output_dir_simple_1/', 'checkpoint-0.pth'))
        for k, v in checkpoint['model'].items():
            if k in model_dict.keys():
                model_dict[k] = v

        model.load_state_dict(model_dict)
        print("load success")

    return model


def run_one_image(img2, tgt2, img1, model, device, out_dir, idx):
    x = torch.tensor(img1)
    x = x.unsqueeze(dim=0)

    img2 = torch.tensor(img2)
    img2 = img2.unsqueeze(dim=0)

    tgt_1_list = torch.unique(tgt2)
    tgt = torch.tensor(tgt2)

    h, w = tgt.shape[-2:]
    ref_mask = torch.zeros((10, h, w))

    for id, choi in enumerate(tgt_1_list[:10]):
        if choi != 0:
            ref_mask[id, :, :] = torch.tensor(tgt == choi, dtype=torch.uint8)

    ref_mask = ref_mask.unsqueeze(dim=0)
    y_pos = model(x.float().to(device), tgt.float().to(device), img2.float().to(device), ref_mask.float().to(device),
                  mode='test')
    y_pos = y_pos.sigmoid()
    save_mask = torch.zeros((h, w), dtype=torch.uint8)

    for id, choi in enumerate(tgt_1_list[:10]):
        if choi != 0:
            # a = np.percentile(y_pos[0][id].detach().cpu().numpy(), 90)
            print((y_pos[0][id] > 0.5).sum())
            # plt.figure(figsize=(10, 10))
            # for index in range(1):
            #     ax = plt.subplot(5, 5, index+1,)
            #     plt.imshow(y_pos[0][id].cpu().detach().numpy())
            #     plt.savefig("feature.jpg",dpi=300)
            save_mask[y_pos[0][id] > 0.5] = torch.tensor(choi, dtype=torch.uint8)
            mask = (y_pos[0][id] > 0.5)
            mask[y_pos[0][id] > 0.5] = torch.tensor(choi, dtype=torch.uint8)
            mask = mask.cpu().numpy().astype(np.uint8)
            save_colored_mask(mask, os.path.join(out_dir, '{}_{}_.png'.format(idx, choi)))
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

    out_dir = "./eval_dir_test/"
    gt_dir = out_dir + '/gts/'
    img_dir = out_dir + '/imgs/'

    if not os.path.exists(gt_dir):
        os.makedirs(gt_dir)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    f_val_list = open("/data1/paintercoco/data/medical_dataset/split/val_mri_ct.txt", 'r').readlines()
    gt_path_files = [
        os.path.join(file.strip().replace('imgs', 'gts'))
        for file in f_val_list
        if os.path.isfile(os.path.join(file.strip()))
    ]
    dataset_dict = defaultdict(list)
    models_painter.eval()

    for gt_file in gt_path_files:
        if 'AbdomenCT-1K' in gt_file:
            dataset_dict['AbdomenCT-1K'].append(gt_file)
        elif 'amos' in gt_file:
            dataset_dict['amos'].append(gt_file)
        elif 'CHAOS' in gt_file:
            dataset_dict['CHAOS'].append(gt_file)
        elif 'MMWHS' in gt_file:
            dataset_dict['MMWHS'].append(gt_file)
        elif 'protaste' in gt_file:
            dataset_dict['protaste'].append(gt_file)
        elif 'Task05' in gt_file:
            dataset_dict['Task05'].append(gt_file)
        elif 'cardic' in gt_file:
            dataset_dict['cardic'].append(gt_file)
        elif 'caridc' in gt_file:
            dataset_dict['caridc'].append(gt_file)
        elif 'MSDHEART' in gt_file:
            dataset_dict['MSDHEART'].append(gt_file)
        elif 'BTCV' in gt_file:
            dataset_dict['BTCV'].append(gt_file)
        elif 'MSD_Spleen' in gt_file:
            dataset_dict['MSD_Spleen'].append(gt_file)
        elif 'pental' in gt_file:
            dataset_dict['pental'].append(gt_file)
        elif 'chasedb' in gt_file:
            dataset_dict['chasedb'].append(gt_file)
        elif 'bus' in gt_file:
            dataset_dict['bus'].append(gt_file)
        elif 'bbc' in gt_file:
            dataset_dict['bbc'].append(gt_file)
        elif 'MSD_Spleen' in gt_file:
            dataset_dict['MSD_Spleen'].append(gt_file)
        elif 'drive' in gt_file:
            dataset_dict['drive'].append(gt_file)
        elif 'stare' in gt_file:
            dataset_dict['stare'].append(gt_file)
        elif 'acdc' in gt_file:
            dataset_dict['acdc'].append(gt_file)
        elif 'SPINE' in gt_file:
            dataset_dict['SPINE'].append(gt_file)
        elif 'monuseg' in gt_file:
            dataset_dict['monuseg'].append(gt_file)

    if ('AbdomenCT-1K' in args.prompt) or ('BTCV' in args.prompt) or ('SPINE' in args.prompt):
        img2_dataset = args.prompt.split("_")[1].split("/")[0]
    elif 'rgb' in args.prompt:
        img2_dataset = args.prompt.split("/")[-2]
    elif 'stare' in args.prompt:
        img2_dataset = args.prompt.split("/")[-2]
    else:
        img2_dataset = args.prompt.split("_")[1]

    img2_1024 = np.load(
        args.prompt, "r", allow_pickle=True
    )

    if ('rgb' not in args.prompt) and ('stare' not in args.prompt):
        if len(img2_1024.shape) == 3:
            x, y, _ = img2_1024.shape
            img2_224 = zoom(img2_1024, (448 / x, 448 / y, 1), order=0)
            img2_224 = np.transpose(img2_224, (2, 0, 1))  ### (3, H, W)
            img2_224 = torch.tensor(img2_224)
        else:
            x, y = img2_1024.shape
            img2_224 = zoom(img2_1024, (448 / x, 448 / y), order=0)
            img2_224 = torch.tensor(img2_224).unsqueeze(0).repeat(3, 1, 1)

        gt2_file = np.load(
            args.prompt.replace("imgs", "gts"), "r", allow_pickle=True
        )
        gt2_file = Image.fromarray(np.uint8(gt2_file)).resize((448, 448), Image.NEAREST)
        # save_colored_mask(np.array(gt2_file), 'gt2.png')
        gt2_file = torch.tensor(np.array(gt2_file))

    else:
        img2_1024_pre = (img2_1024 - np.min(img2_1024)) / (np.max(img2_1024) - np.min(img2_1024))
        if len(img2_1024_pre.shape) == 3:
            x, y, _ = img2_1024_pre.shape
            img2_224 = zoom(img2_1024_pre, (448 / x, 448 / y, 1), order=0)
            img2_224 = np.transpose(img2_224, (2, 0, 1))  ### (3, H, W)
            img2_224 = torch.tensor(img2_224)
        else:
            x, y = img2_1024_pre.shape
            img2_224 = zoom(img2_1024, (448 / x, 448 / y), order=0)
            img2_224 = torch.tensor(img2_224).unsqueeze(0).repeat(3, 1, 1)

        if 'drive' in args.prompt:
            gt_filename = args.prompt.replace("_training", "_manual1")
            gt_filename = gt_filename.replace("imgs", "gts")
            gt2_file = np.load(
                gt_filename, "r", allow_pickle=True
            )
        else:
            gt2_file = np.load(
                args.prompt.replace("imgs", "gts"), "r", allow_pickle=True
            )

        gt2_file = Image.fromarray(np.uint8(gt2_file)).resize((448, 448), Image.NEAREST)
        gt2_file = torch.tensor(np.array(gt2_file))

    for idx, gt_file in enumerate(dataset_dict[img2_dataset]):
        if gt_file.replace('gts', 'imgs') != args.prompt:
            img1_1024 = np.load(
                gt_file.replace('gts', 'imgs'), "r", allow_pickle=True
            )
            img_save = Image.fromarray(np.uint8(img1_1024 * 255))
            img_save.save("img2.png")

            if 'rgb' in args.prompt or 'stare' in args.prompt:
                img1_1024 = (img1_1024 - np.min(img1_1024)) / (np.max(img1_1024) - np.min(img1_1024))

            if len(img1_1024.shape) == 3:
                x, y, _ = img1_1024.shape
                img1_224 = zoom(img1_1024, (448 / x, 448 / y, 1), order=0)
                img1_224 = np.transpose(img1_224, (2, 0, 1))  ### (3, H, W)
                img1_224 = torch.tensor(img1_224)
            else:
                x, y = img1_1024.shape
                img1_224 = zoom(img1_1024, (448 / x, 448 / y), order=0)
                img1_224 = torch.tensor(img1_224).unsqueeze(0).repeat(3, 1, 1)

            if 'drive' in gt_file:
                gt_filename = gt_file.replace("_training", "_manual1")
                gt_filename = gt_filename.replace("imgs", "gts")
                gt1_file = np.load(
                    gt_filename, "r", allow_pickle=True
                )
            else:
                gt1_file = np.load(
                    gt_file.replace("imgs", "gts"), "r", allow_pickle=True
                )

            gt1_file = Image.fromarray(np.uint8(gt1_file)).resize((448, 448), Image.NEAREST)
            save_colored_mask(np.array(gt1_file), os.path.join(gt_dir, 'gt{}.png'.format(idx)))
            gt1_file = torch.tensor(np.array(gt1_file))

            torch.manual_seed(2)
            run_one_image(img2_224, gt2_file, img1_224, models_painter, device, img_dir, idx)