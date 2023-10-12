import os
import sys
sys.path.append("/mnt/XN-Net/demo_code/utils/")
print(sys.path)
# from .ACDC.transform1 import random_rot_flip, random_rotate
import h5py
import numpy as np
from scipy.ndimage.interpolation import zoom
import torch
from torch.utils.data import Dataset
import random
from detectron2.structures import BitMasks, Instances
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage import transform
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import glob
from PIL import Image, ImageOps, ImageFilter
from constants import MEDICAL_CLASSES
from utils.prompt_engineering import prompt_engineering
from collections import defaultdict
import imgviz
# from visual_sampler.sampler import build_shape_sampler

def save_colored_mask(mask, save_path):
    lbl_pil = Image.fromarray(mask.astype(np.uint8), mode="P")
    colormap = imgviz.label_colormap()
    lbl_pil.putpalette(colormap.flatten())
    lbl_pil.save(save_path)

CLASSES_NAME_IDS = {
    'acdc':{0:'background', 1:'right ventricle', 2:'myocardium', 3:'left ventricle'
           }, 
    'amos': {0:'background', 1:'spleen', 2:'right kidney', 3:'left kidney', 4:'gall bladder', 5:'esophagus', 
             6:'liver', 7:'stomach', 8:'arota', 9:'postcava', 10:'pancreas', 11:'right adrenal gland', 
             12:'left adrenal gland', 13:'duodenum', 14:'bladder', 15:'prostate'},
    'CHAOS': {0:'background',1:'liver', 2:'right kidney',3:'left kidney', 4:'spleen'
             }, 
    'MMWHS': {0:'background', 500:'left ventricle', 600:'right ventricle', 205:'myocardium', 420:'left atrium', 550:'right atrium', 820:'ascending aorta', 850:'pulmonary artery'}, 
    'protaste': {0:'background', 1:'prostate'}, 
    'Task05': {0:'background', 1:'prostate'},
    'cardic': {0:'background', 1:'left ventricle', 2:'right ventricle', 3:'myocardium'},
    'caridc' : {0:'background', 1:'left ventricle', 2:'right ventricle', 3:'myocardium'},
    'MSDHEART': {0: 'background', 1: 'left atrium'}
    }

join = os.path.join
# set seeds
torch.manual_seed(2023)
torch.cuda.empty_cache()

class NpyDataset(Dataset):
    def __init__(self, data_root, mode):
        self.data_root = data_root
        self.gt_path = join(data_root, "gts")
        self.img_path = join(data_root, "imgs")
        self.mode = mode
        if mode == 'train':
            f_train_list = open("/mnt/XN-Net/demo_code/dataset/uniform_dataset/train.txt", 'r').readlines()
            self.gt_path_files = [
            join(self.gt_path, file.strip())
            for file in f_train_list
            if os.path.isfile(join(self.img_path, file.strip()))
        ]
        else:
            f_val_list = open("/mnt/XN-Net/demo_code/dataset/uniform_dataset/val.txt", 'r').readlines()
            self.gt_path_files = [
            join(self.gt_path, file.strip())
            for file in f_val_list
            if os.path.isfile(join(self.img_path, file.strip()))
        ]
            
        dataset_dict = defaultdict(list)
        for gt_file in self.gt_path_files:
            if 'acdc' in gt_file:
                dataset_dict['acdc'].append(gt_file)
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
        
        self.dataset_dict = dataset_dict
        # self.shape_sampler = build_shape_sampler(cfg)
        # self.gt_path_files = sorted(
        #     glob.glob(join(self.gt_path, "**/*.npy"), recursive=True)
        # )
        # self.gt_path_files = [
        #     file
        #     for file in self.gt_path_files
        #     if os.path.isfile(join(self.img_path, os.path.basename(file)))
        # ]
        
        print(f"number of images: {len(self.gt_path_files)}")

    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        if self.mode == 'train':
        # load npy image (1024, 1024, 3), [0,1]
            class_names = [k for k in CLASSES_NAME_IDS.keys()]
            name = random.sample(class_names, 1)
            path = random.sample(self.dataset_dict[name[0]], 1)
            img_name = os.path.basename(path[0])
            # img_name = os.path.basename(self.gt_path_files[index])
        else:

            img_name = os.path.basename(self.gt_path_files[index])
        dataset_name = img_name.split("_")[1]
        print(dataset_name)
        img_1024 = np.load(
            join(self.img_path, img_name), "r", allow_pickle=True
        )  # (1024, 1024, 3)
        # print(img_1024)

        img_save = Image.fromarray(np.uint8(img_1024 * 255))
        img_save.save("1.jpg")

        if len(img_1024.shape) == 3:
            x, y, _ = img_1024.shape
            img_224 = zoom(img_1024, (224 / x, 224 / y, 1), order=0)
            img_224 = np.transpose(img_224, (2, 0, 1))  ### (3, H, W)
            img_224 = torch.tensor(img_224)
        else:
            x, y = img_1024.shape
            img_224 = zoom(img_1024, (224 / x, 224 / y), order=0)
            img_224 = torch.tensor(img_224).unsqueeze(0).repeat(3, 1, 1)


        assert (
            np.max(img_1024) <= 1.0 and np.min(img_1024) >= 0.0
        ), "image should be normalized to [0, 1]"
        
        if self.mode == 'val':
            gt = np.load(
                self.gt_path_files[index], "r", allow_pickle=True
            )
        else:
            gt = np.load(
                path[0], "r", allow_pickle=True
            )
            # gt = np.load(
            #     self.gt_path_files[index], "r", allow_pickle=True
            # )

        gt_save = Image.fromarray(np.uint8(gt))
        save_colored_mask(np.uint8(gt==1), "2.png")
        save_colored_mask(np.uint8(gt==2), "3.png")
        save_colored_mask(np.uint8(gt==3), "4.png")


        gt_save.save("2.jpg")

        new_gt_224 = np.zeros_like(gt, dtype=np.uint8)
        # print(np.unique(gt))

        for it in np.unique(gt):
            cls = CLASSES_NAME_IDS[dataset_name][it]
            true_label = MEDICAL_CLASSES.index(cls)
            new_gt_224[gt == it] = true_label

        new_gt_224 = Image.fromarray(np.uint8(new_gt_224)).resize((224, 224), Image.NEAREST)
        new_gt_224 = np.array(new_gt_224)
        # save_colored_mask(new_gt_224, "3.png")
        # assert len(np.unique(gt_224)) <= len(CLASSES_NAME_IDS[dataset_name])
        ###### 重新设置gt值 ##############################

        # assert img_name == os.path.basename(self.gt_path_files[index]), (
        #     "img gt name error" + self.gt_path_files[index] + self.npy_files[index]
        # )

        # label_ids = np.unique(gt)[1:]
        label_ids = np.unique(new_gt_224)
        dataset_dict = {}
        dataset_dict['spatial_query'] = {}
        dataset_dict["image"] = img_224
        image_shape = img_224.shape
        
        instances = Instances(image_shape)
        instances.gt_classes = torch.tensor(label_ids, dtype=torch.int64)[:-1]
        
        seg_masks = []
        for cls in label_ids:
            if cls != 22:
                segment = new_gt_224 == cls
                seg_masks.append(segment)  ##### 每个类别的0/1 mask

        if len(seg_masks) == 0:
            instances.gt_masks = torch.zeros((0, new_gt_224.shape[0], new_gt_224.shape[1]))
        else:
            masks = BitMasks(
                torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in seg_masks])
            )
            instances.gt_masks = masks.tensor

        dataset_dict["instances"] = instances

        texts_grd = np.array([MEDICAL_CLASSES[idx] for idx in instances.gt_classes])
        hash_grd = np.array([hash(txt) for txt in texts_grd])
        unique_hash_grd = np.unique(hash_grd)
        np.random.shuffle(unique_hash_grd)

        grounding_len = 20
        mode = 'class'
        masks_grd = instances.gt_masks

        max_len = min(grounding_len, len(unique_hash_grd))
        indices = np.random.permutation(max_len)     

        selected_unique_hash_grd = unique_hash_grd[indices]
        selected_mask = np.in1d(hash_grd, selected_unique_hash_grd)
        texts_grd = texts_grd[selected_mask]
        hash_grd = hash_grd[selected_mask]
        masks_grd = masks_grd[selected_mask]
        
        texts_grd = [prompt_engineering(text, topk=10000, suffix='.') \
                                        for text in texts_grd]
        
        groundings = {'masks': masks_grd, 'texts': texts_grd, 'mode': mode, 'hash': hash_grd}
        dataset_dict["groundings"] = groundings

        true_mask = instances.gt_masks
        random_mask = torch.randint(low=0, high=2, size=instances.gt_masks.shape).bool()
        dataset_dict['spatial_query']['rand_shape'] = torch.logical_and(random_mask, true_mask)
        # print(dataset_dict['spatial_query']['rand_shape'].sum())
        # dataset_dict['spatial_query']['rand_shape'] = torch.randint(low=0, high=2, size=instances.gt_masks.shape).bool()
        dataset_dict['spatial_query']['gt_masks'] = instances.gt_masks

        # spatial_query_utils = self.shape_sampler(instances)
        # dataset_dict['spatial_query'] = spatial_query_utils

        return dataset_dict

    def __len__(self):
        return len(self.gt_path_files)

def collate_func(batch_dic):
    result = []
    for i in range(len(batch_dic)):
        result.append(batch_dic[i])
    return result

# if __name__ == '__main__':
#     tr_dataset = NpyDataset("data/npy/")
#     tr_dataloader = DataLoader(tr_dataset, batch_size=1, shuffle=True)

#     for data in tr_dataloader:
#         print(data)

