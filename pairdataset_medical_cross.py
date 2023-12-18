import imagecorruptions
import math
import os.path
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import torch
from torchvision.datasets.vision import VisionDataset, StandardTransform
from data.trainsam_transform import *
import data.pair_transforms as pair_transforms
from collections import defaultdict
from scipy.ndimage.interpolation import zoom
import imgviz
import albumentations as A
import cv2 as cv


class PairStandardTransform(StandardTransform):
    def __init__(self, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None) -> None:
        super().__init__(transform=transform, target_transform=target_transform)

    def __call__(self, input: Any, target: Any, interpolation1: Any, interpolation2: Any) -> Tuple[Any, Any]:
        if self.transform is not None:
            input, target = self.transform(input, target, interpolation1, interpolation2)
        return input, target


def save_colored_mask(mask, save_path):
    lbl_pil = Image.fromarray(mask.astype(np.uint8), mode='P')
    color_map = imgviz.label_colormap()
    lbl_pil.putpalette(color_map.flatten())
    lbl_pil.save(save_path)


join = os.path.join

CLASSES_NAME_IDS = {
    'amos': {0: 'background', 1: 'spleen', 2: 'right kidney', 3: 'left kidney', 4: 'gall bladder', 5: 'esophagus',
             6: 'liver', 7: 'stomach', 8: 'arota', 9: 'postcava', 10: 'pancreas', 11: 'right adrenal gland',
             12: 'left adrenal gland', 13: 'duodenum', 14: 'bladder', 15: 'prostate'},
    'CHAOS': {0: 'background', 1: 'liver', 2: 'right kidney', 3: 'left kidney', 4: 'spleen'
              },
    'MMWHS': {0: 'background', 500: 'left ventricle', 600: 'right ventricle', 205: 'myocardium', 420:'left atrium', 550: 'right atrium', 820: 'ascending aorta', 850: 'pulmonary artery'},
    'protaste': {0: 'background', 1: 'prostate'},
    'Task05': {0: 'background', 1: 'prostate'},
    'cardic': {0: 'background', 1: 'left ventricle', 2: 'right ventricle', 3: 'myocardium'},
    'caridc': {0: 'background', 1: 'left ventricle', 2: 'right ventricle', 3: 'myocardium'},
    'MSDHEART': {0: 'background', 1: 'left atrium'},
    'BTCV': {0: 'background', 1: 'spleen', 2: 'right kidney', 3: 'left kidney', 4: 'gall bladder',
             5: 'esophagus', 6: 'liver', 7: 'stomach', 8: 'aorta', 9: 'inferior vena cava',
             10: 'portal vein and splenic vein', 11: 'pancreas', 12: 'right adrenal gland', 13: 'left adrenal gland'},
    'AbdomenCT-1K': {0: 'background', 1: 'liver', 2: 'kidney', 3: 'spleen', 4: 'pancreas'},
    'MSD_Spleen': {0: 'background', 1: 'spleen'},
    'bbc': {0: 'background', 1: 'Mouse embryos'},
    'bus': {0: 'background', 1: 'Breast tumor'},
    'chasedb': {0: 'background', 1: 'Blood vessels'},
    'drive': {0: 'background', 1: 'Blood vessels'},
    'ircid': {0: 'background', 1: 'Blood vessels'},
    'SegTHOR': {0: 'background', 1: 'esophagus', 2: 'heart', 3: 'trachea', 4: 'aorta'},
    'LiTS': {0: 'background', 1: 'liver', 2: 'liver tumor'},
    'KiPA': {0: 'background', 1: 'renal artery', 2: 'kidney', 3: 'renal_vein', 4: 'kidney tumor'},
    'COVID_lung': {0: 'background', 1: 'left lung', 2: 'right lung'},
    'MSD_Colon': {0: 'background', 1: 'colon'},
    'camus': {0: 'background', 1: 'left ventricle', 2: 'myocardium', 3: 'left atrium'},
    'Cdemris': {0: 'background', 1: 'left atrium'},
    'chest_xray': {0: 'background', 1: 'chest'},
    'consep': {0: 'background', 1: 'other', 2: 'inflammatory', 3: 'epithelial', 4: 'spindle-shaped'},
    'eopha': {0: 'background', 1: 'EX/MA'},
    'HMCQU': {0: 'background', 1: 'left ventricle wall'},
    'isles': {0: 'background', 1: 'stroke_lesion'},
    'MRI_I2CVB': {0: 'background', 1: 'prostate'},
    'oasis': {0: 'background', 1 : 'brain'},
    'ROSE': {0: 'background', 1: 'Blood vessels'},
    'Thyroid Dataset': {0: 'background', 1: 'thyroid_nodules'}
}
rgb_ids = {'Cdemris', 'chest_xray', 'consep', 'eopha', 'HMCQU', 'ROSE', 'Thyroid Dataset'}

abnormal_dict = {
    'MMWHS': {0: 0, 500: 1, 600: 2, 205: 3, 420: 4, 550: 5, 820: 6, 850: 7}
}


class PairMedicalDataset(VisionDataset):

    def __init__(
            self,
            data_root, mode,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,

    ) -> None:

        super().__init__(data_root, transforms, transform, target_transform)
        self.data_root = data_root
        self.mode = mode
        if mode == 'train':
            f_train_list = open("/data1/paintercoco/data/medical_dataset/split/train_mri_ct.txt", 'r').readlines()
            self.gt_path_files = [
                join(file.strip().replace('imgs', 'gts'))
                for file in f_train_list
                if os.path.isfile(join(file.strip()))
            ]

        elif mode == 'val':
            f_val_list = open("/data1/paintercoco/data/medical_dataset/split/val_mri_ct.txt", 'r').readlines()
            self.gt_path_files = [
                join(file.strip().replace('imgs', 'gts'))
                for file in f_val_list
                if os.path.isfile(join(file.strip()))
            ]
        else:
            f_test_list = open("/data1/paintercoco/data/medical_dataset/split/test.txt", 'r').readlines()
            self.gt_path_files = [
                join(file.strip().replace('imgs', 'gts'))
                for file in f_test_list
                if os.path.isfile(join(file.strip()))
            ]

        dataset_dict = defaultdict(list)
        for gt_file in self.gt_path_files:
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
            elif 'chasedb' in gt_file:
                dataset_dict['chasedb'].append(gt_file)
            elif 'bus' in gt_file:
                dataset_dict['bus'].append(gt_file)
            elif 'bbc' in gt_file:
                dataset_dict['bbc'].append(gt_file)
            elif 'drive' in gt_file:
                dataset_dict['drive'].append(gt_file)
            elif 'ircid' in gt_file:
                dataset_dict['ircid'].append(gt_file)
            elif 'SegTHOR' in gt_file:
                dataset_dict['SegTHOR'].append(gt_file)
            elif 'MSD_Colon' in gt_file:
                dataset_dict['MSD_Colon'].append(gt_file)
            elif 'LiTS' in gt_file:
                dataset_dict['LiTS'].append(gt_file)
            elif 'KiPA' in gt_file:
                dataset_dict['KiPA'].append(gt_file)
            elif 'COVID_lung' in gt_file:
                dataset_dict['COVID_lung'].append(gt_file)
            elif 'camus' in gt_file:
                dataset_dict['camus'].append(gt_file)
            elif 'Cdemris' in gt_file:
                dataset_dict['Cdemris'].append(gt_file)
            elif 'chest_xray' in gt_file:
                dataset_dict['chest_xray'].append(gt_file)
            elif 'consep' in gt_file:
                dataset_dict['consep'].append(gt_file)
            elif 'eopha' in gt_file:
                dataset_dict['eopha'].append(gt_file)
            elif 'HMCQU' in gt_file:
                dataset_dict['HMCQU'].append(gt_file)
            elif 'isles' in gt_file:
                dataset_dict['isles'].append(gt_file)
            elif 'MRI_I2CVB' in gt_file:
                dataset_dict['MRI_I2CVB'].append(gt_file)
            elif 'oasis' in gt_file:
                dataset_dict['oasis'].append(gt_file)
            elif 'ROSE' in gt_file:
                dataset_dict['ROSE'].append(gt_file)
            elif 'Thyroid Dataset' in gt_file:
                dataset_dict['Thyroid Dataset'].append(gt_file)

        self.dataset_dict = dataset_dict
        self.use_two_pairs = True
        self.transforms2 = PairStandardTransform(transform, target_transform) if transform is not None else None

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        if self.mode == 'train':
            class_names = [k for k in CLASSES_NAME_IDS.keys()]
            name = random.choices(class_names, k=1)[0]
            path = random.sample(self.dataset_dict[name], 1)
            img_name = path[0]
        else:
            img_name = self.gt_path_files[index]

        img_1024 = np.load(
            join(img_name.replace('gts', 'imgs')), "r", allow_pickle=True
        )

        if ('rgb' in img_name) or any(substring in img_name for substring in rgb_ids):
            if len(img_1024.shape) == 3:
                img_1024 = Image.fromarray(img_1024)
                img_1024 = img_1024.resize((448, 448), resample=Image.BILINEAR)
                img_1024 = img_1024.convert('L')
                img_1024 = np.array(img_1024)
                img_1024 = (img_1024 - np.min(img_1024)) / (np.max(img_1024) - np.min(img_1024))

                img = img_1024.astype(np.float32)
                img_448 = torch.tensor(img).unsqueeze(0).repeat(3, 1, 1)

            else:
                x, y = img_1024.shape
                img_1024 = (img_1024 - np.min(img_1024)) / (np.max(img_1024) - np.min(img_1024))
                img_448 = zoom(img_1024, (448 / x, 448 / y), order=0)
                img_448 = torch.tensor(img_448).unsqueeze(0).repeat(3, 1, 1)
        else:
            if len(img_1024.shape) == 3:
                x, y, _ = img_1024.shape
                img_448 = zoom(img_1024, (448 / x, 448 / y, 1), order=0)
                img_448 = np.transpose(img_448, (2, 0, 1))  ### (3, H, W)
                img_448 = torch.tensor(img_448)
            else:
                x, y = img_1024.shape
                img_448 = zoom(img_1024, (448 / x, 448 / y), order=0)
                img_448 = torch.tensor(img_448).unsqueeze(0).repeat(3, 1, 1)


        assert (
                np.max(img_1024) <= 1.0 and np.min(img_1024) >= 0.0
        ), "image should be normalized to [0, 1]"

        if self.mode == 'val' or self.mode == 'test':
            if 'drive' in self.gt_path_files[index]:
                gt_filename = self.gt_path_files[index].replace("_training", "_manual1")
                gt = np.load(
                    gt_filename, "r", allow_pickle=True
                )
            elif 'ircid' in self.gt_path_files[index]:
                gt_filename = self.gt_path_files[index].split(".")[0] + "_OD.npy"
                gt = np.load(
                    gt_filename, "r", allow_pickle=True
                )
            elif 'Cdemris' in self.gt_path_files[index]:
                gt_filename = self.gt_path_files[index].replace('de_', 'la_seg_')
                gt = np.load(
                    gt_filename, "r", allow_pickle=True
                )
            elif 'chest_xray' in self.gt_path_files[index]:
                gt_filename = self.gt_path_files[index][:-4] + '_mask.npy'
                gt = np.load(
                    gt_filename, "r", allow_pickle=True
                )
            else:
                gt = np.load(
                    self.gt_path_files[index], "r", allow_pickle=True
                )
        else:
            if 'drive' in img_name:
                gt_filename = img_name.replace("_training", "_manual1")
                gt = np.load(
                    gt_filename, "r", allow_pickle=True
                )
            elif 'ircid' in img_name:
                gt_filename = img_name.split(".")[0] + "_OD.npy"
                gt = np.load(
                    gt_filename, "r", allow_pickle=True
                )
            elif 'Cdemris' in img_name:
                gt_filename = img_name.replace('de_', 'la_seg_')
                gt = np.load(
                    gt_filename, "r", allow_pickle=True
                )
            elif 'chest_xray' in img_name:
                gt_filename = img_name[:-4] + '_mask.npy'
                gt = np.load(
                    gt_filename, "r", allow_pickle=True
                )
            else:
                gt = np.load(
                    img_name, "r", allow_pickle=True
                )

        new_gt_448 = np.zeros_like(gt, dtype=np.uint8)
        for it in np.unique(gt):
            new_gt_448[gt == it] = it

        new_gt_448 = Image.fromarray(np.uint8(new_gt_448)).resize((448, 448), Image.NEAREST)
        new_gt_448 = np.array(new_gt_448)
        label_ids = (np.unique(new_gt_448)).tolist()

        h, w = new_gt_448.shape
        pos_target = torch.zeros((10, h, w), dtype=torch.uint8)
        ref_in = torch.zeros((10, h, w), dtype=torch.uint8)
        area_list = torch.zeros(10, dtype=torch.float32)

        if len(label_ids) > 10:
            pos_ids = random.sample(label_ids, 10)
        else:
            pos_ids = random.sample(label_ids, len(label_ids))

        for id, choi in enumerate(pos_ids):
            if choi != 0:
                final_mask = new_gt_448 == choi
                final_mask = final_mask.astype(np.uint8)
                ref_in[id, final_mask] = 1
                pos_target[id, final_mask] = 1
                area_list[id] = final_mask.sum()

        target = pos_target
        target2 = ref_in

        interpolation1 = 'bicubic'
        interpolation2 = 'nearest'

        ########## 对于train sample和example进行足够的数据增强，使得两者差异足够大 ##########################
        if self.mode == 'train':
            train_transform = A.Compose([
                A.RandomResizedCrop(448, 448, scale=(0.9999, 1.0), interpolation=3),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.GaussNoise(var_limit=[0, 0.05], mean=0, p=0.4),
                A.GaussianBlur(sigma_limit=[0.1, 1.1], p=0.4),
                A.ElasticTransform(alpha=2.5, p=0.8),
                A.RandomBrightnessContrast(p=0.4, brightness_limit=[-0.1, 0.1], contrast_limit=[0.5, 1.5]),
                A.RandomRotate90(p=0.5),
                A.GridDistortion(num_steps=5, distort_limit=0.3, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.5)
            ], is_check_shapes=False)

            trans = train_transform(image=np.array(img_448.permute(1, 2, 0), dtype=np.float32), mask=np.array(target.permute(1, 2, 0)))
            image, target = trans['image'], trans['mask']

            image = torch.Tensor(image).permute(2, 1, 0)
            target = torch.Tensor(target).permute(2, 1, 0)
        else:
            train_transform = pair_transforms.Compose([
                pair_transforms.RandomResizedCrop(448, scale=(0.9999, 1.0), interpolation=3)])
            image, target = train_transform(img_448, target, interpolation1, interpolation2)

        if self.use_two_pairs:
            img2_1024 = np.load(
                join(img_name.replace('gts', 'imgs')), "r", allow_pickle=True
            )

            if 'rgb' in img_name or any(substring in img_name for substring in rgb_ids):
                if len(img2_1024.shape) == 3:
                    img2_1024 = Image.fromarray(img2_1024)
                    img2_1024 = img2_1024.resize((448, 448), resample=Image.BILINEAR)
                    img2_1024 = img2_1024.convert('L')
                    img2_1024 = np.array(img2_1024)
                    img2_1024 = (img2_1024 - np.min(img2_1024)) / (np.max(img2_1024) - np.min(img2_1024))

                    img = img2_1024.astype(np.float32)
                    img2_448 = torch.tensor(img).unsqueeze(0).repeat(3, 1, 1)
                else:
                    x, y = img2_1024.shape
                    img2_1024 = (img2_1024 - np.min(img2_1024)) / (np.max(img2_1024) - np.min(img2_1024))
                    img2_448 = zoom(img2_1024, (448 / x, 448 / y), order=0)
                    img2_448 = torch.tensor(img2_448).unsqueeze(0).repeat(3, 1, 1)
            else:
                if len(img2_1024.shape) == 3:
                    x, y, _ = img2_1024.shape
                    img2_448 = zoom(img2_1024, (448 / x, 448 / y, 1), order=0)
                    img2_448 = np.transpose(img2_448, (2, 0, 1))  ### (3, H, W)
                    img2_448 = torch.tensor(img2_448)
                else:
                    x, y = img2_1024.shape
                    img2_448 = zoom(img2_1024, (448 / x, 448 / y), order=0)
                    img2_448 = torch.tensor(img2_448).unsqueeze(0).repeat(3, 1, 1)

            image2 = img2_448
        return image, torch.tensor(target), image2, target2.to(torch.float32), area_list, img_name

    def __len__(self) -> int:
        return len(self.gt_path_files)
