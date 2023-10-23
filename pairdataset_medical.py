import os.path
import json
from typing import Any, Callable, List, Optional, Tuple
from torchvision.datasets.vision import VisionDataset, StandardTransform
from data.trainsam_transform import *
import data.pair_transforms as pair_transforms
from torchvision import transforms
import cv2 as cv
from collections import defaultdict
from scipy.ndimage.interpolation import zoom
import imgviz
import albumentations as A

def save_colored_mask(mask, save_path):
    lbl_pil = Image.fromarray(mask.astype(np.uint8), mode='P')
    color_map = imgviz.label_colormap()
    lbl_pil.putpalette(color_map.flatten())
    lbl_pil.save(save_path)

join = os.path.join

CLASSES_NAME_IDS = {
    'acdc':{0:'background', 1:'right ventricle', 2:'myocardium', 3:'left ventricle'
           },
    'amos': {0:'background', 1:'spleen', 2:'right kidney', 3:'left kidney', 4:'gall bladder', 5:'esophagus',
             6:'liver', 7:'stomach', 8:'arota', 9:'postcava', 10:'pancreas', 11:'right adrenal gland',
             12:'left adrenal gland', 13:'duodenum', 14: 'bladder', 15:'prostate'},
    'CHAOS': {0:'background',1:'liver', 2:'right kidney',3: 'left kidney', 4:'spleen'
             },
    # 'MMWHS': {0: 'background', 500: 'left ventricle', 600: 'right ventricle', 205:'myocardium', 420:'left atrium', 550:'right atrium', 820:'ascending aorta', 850:'pulmonary artery'},
    # 'protaste': {0: 'background', 1: 'prostate'},
    # 'Task05': {0: 'background', 1: 'prostate'},
    'cardic': {0: 'background', 1: 'left ventricle', 2: 'right ventricle', 3: 'myocardium'},
    'caridc' : {0: 'background', 1: 'left ventricle', 2: 'right ventricle', 3: 'myocardium'},
    'MSDHEART': {0: 'background', 1: 'left atrium'},
    'btcv':{0:'background', 1:'spleen', 2:'right kidney', 3:'left kidney', 4:'gall bladder',
        5:'esophagus', 6:'liver', 7:'stomach', 8:'aorta', 9:'inferior vena cava', 10:'portal vein and splenic vein',
        11:'pancreas', 12:'right adrenal gland', 13:'left adrenal gland'},
    'abdomenct-1k':{0:'background', 1:'liver', 2:'kidney', 3:'spleen', 4:'pancreas'},
    'chaos_ct':{0:'background', 1:'liver'},
    'bbc003': {0:'background', 1:'Mouse embryos'},
    'BUS': {0:'background', 1:'Breast tumor'},
    'DRIVE':{0:'background', 1:'Blood vessels'},
    'CHASEDB':{0:'background', 1:'Blood vessels'},
    'WBC': {0:'background', 1:'cytoplasm', 2:'nucleus'}
    }

abnormal_dict = {
    'MMWHS':{0:0, 500:1, 600:2, 205:3, 420:4, 550:5, 820:6, 850:7}
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
        self.gt_path = join(data_root, "gts")
        self.img_path = join(data_root, "imgs")
        self.mode = mode
        if mode == 'train':
            f_train_list = open("/mnt/paintercoco/data/medical_dataset/split/train.txt", 'r').readlines()
            self.gt_path_files = [
                join(self.gt_path, file.strip())
                for file in f_train_list
                if os.path.isfile(join(self.img_path, file.strip()))
            ]
        elif mode == 'val':
            f_val_list = open("/mnt/paintercoco/data/medical_dataset/split/val.txt", 'r').readlines()
            self.gt_path_files = [
                join(self.gt_path, file.strip())
                for file in f_val_list
                if os.path.isfile(join(self.img_path, file.strip()))
            ]
        else:
            f_test_list = open("/mnt/paintercoco/data/medical_dataset/split/test.txt", 'r').readlines()
            self.gt_path_files = [
                join(self.gt_path, file.strip())
                for file in f_test_list
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
        self.use_two_pairs = True
        self.transforms2 = PairStandardTransform(transform, target_transform) if transform is not None else None

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        if self.mode == 'train':
            class_names = [k for k in CLASSES_NAME_IDS.keys()]
            name = random.sample(class_names, 1)
            path = random.sample(self.dataset_dict[name[0]], 1)
            img_name = os.path.basename(path[0])
        else:
            img_name = os.path.basename(self.gt_path_files[index])

        dataset_name = img_name.split("_")[1]
        img_1024 = np.load(
            join(self.img_path, img_name), "r", allow_pickle=True
        )

        if len(img_1024.shape) == 3:
            x, y, _ = img_1024.shape
            img_224 = zoom(img_1024, (448 / x, 448 / y, 1), order=0)
            img_224 = np.transpose(img_224, (2, 0, 1))  ### (3, H, W)
            img_224 = torch.tensor(img_224)
        else:
            x, y = img_1024.shape
            img_224 = zoom(img_1024, (448 / x, 448 / y), order=0)
            img_224 = torch.tensor(img_224).unsqueeze(0).repeat(3, 1, 1)

        assert (
                np.max(img_1024) <= 1.0 and np.min(img_1024) >= 0.0
        ), "image should be normalized to [0, 1]"

        if self.mode == 'val' or self.mode == 'test':
            gt = np.load(
                self.gt_path_files[index], "r", allow_pickle=True
            )
        else:
            gt = np.load(
                path[0], "r", allow_pickle=True
            )


        ################# 统一不同数据集的类别为0, 1, 2, 3.... ##############################
        new_gt_224 = np.zeros_like(gt, dtype=np.uint8)
        for it in np.unique(gt):
            if dataset_name in abnormal_dict.keys():
                true_label = abnormal_dict[dataset_name][it]
                new_gt_224[gt == it] = true_label
            else:
                true_label = it
                new_gt_224[gt == it] = it

        new_gt_224 = Image.fromarray(np.uint8(new_gt_224)).resize((448, 448), Image.NEAREST)
        new_gt_224 = np.array(new_gt_224)
        label_ids = (np.unique(new_gt_224)).tolist()

        h, w = new_gt_224.shape
        pos_target = torch.zeros((5, h, w), dtype=torch.uint8)
        ref_in = torch.zeros((5, h, w), dtype=torch.uint8)

        ################# 随机选择10个类别作为提示并使得prompt和target不一致 #####################
        if len(label_ids) > 5:
            pos_ids = random.sample(label_ids, 5)
        else:
            pos_ids = random.sample(label_ids, len(label_ids))
        

        for id, choi in enumerate(pos_ids):
            if choi != 0:
                final_mask = new_gt_224 == choi
                final_mask = final_mask.astype(np.uint8)
                ####### 部分提示整体操作 ###########################################################
                # num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(final_mask, connectivity=8)
                # choi_id = random.randint(1, num_labels - 1)
                # mask = labels == choi_id
                ####### 腐蚀过的提示完整的部分 #####################################################
                # kernel2=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))#圆形结构元素
                # ellipes=cv2.erode(final_mask,kernel2,iterations=8)#圆形腐蚀
                # save_colored_mask(ellipes, "fushi.png")
                # save_colored_mask(final_mask, "yuanlai.png")
                ref_in[id, final_mask] = 1
                pos_target[id, final_mask] = 1

        target = pos_target
        target2 = ref_in

        interpolation1 = 'bicubic'
        interpolation2 = 'nearest'
        cur_transforms = self.transforms2
        
        ########## 对于train sample和example进行足够的数据增强，使得两者差异足够大 ##########################
        if self.mode == 'train':
            train_transform = pair_transforms.Compose([
                pair_transforms.RandomResizedCrop(448, scale=(0.9999, 1.0), interpolation=3),
                pair_transforms.RandomHorizontalFlip()])
            image, target = train_transform(img_224, target, interpolation1, interpolation2)
            image = transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8)(image)
        else:
            train_transform = pair_transforms.Compose([
                pair_transforms.RandomResizedCrop(448, scale=(0.9999, 1.0), interpolation=3)])
            image, target = train_transform(img_224, target, interpolation1, interpolation2)

        if self.use_two_pairs:
            img2_1024 = np.load(
            join(self.img_path, img_name), "r", allow_pickle=True
            )
            if len(img2_1024.shape) == 3:
                x, y, _ = img2_1024.shape
                img2_224 = zoom(img2_1024, (448 / x, 448 / y, 1), order=0)
                img2_224 = np.transpose(img2_224, (2, 0, 1))  ### (3, H, W)
                img2_224 = torch.tensor(img2_224)
            else:
                x, y = img2_1024.shape
                img2_224 = zoom(img2_1024, (448 / x, 448 / y), order=0)
                img2_224 = torch.tensor(img2_224).unsqueeze(0).repeat(3, 1, 1)

            image2, target2 = cur_transforms(img2_224, target2, interpolation1, interpolation2)

            # image2_1 = Image.fromarray(np.uint8(image2.permute(1, 2, 0) * 255))
            # image2_1.save("image2.png")

        # image = normalize(image)
        # image2 = normalize(image2)
        # for i in range(ref_in.shape[0]):
        #     img = Image.fromarray(np.uint8(target2[i]) * 255)
        #     img.save("ref{}.jpg".format(i))
        #     img_t = Image.fromarray(np.uint8(target[i] * 255))
        #     img_t.save("tar{}.jpg".format(i))

        return image, torch.tensor(target), image2, target2.to(torch.float32)


    def __len__(self) -> int:
        return len(self.gt_path_files)


class PairStandardTransform(StandardTransform):
    def __init__(self, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None) -> None:
        super().__init__(transform=transform, target_transform=target_transform)

    def __call__(self, input: Any, target: Any, interpolation1: Any, interpolation2: Any) -> Tuple[Any, Any]:
        if self.transform is not None:
            input, target = self.transform(input, target, interpolation1, interpolation2)
        return input, target
