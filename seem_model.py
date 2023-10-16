import random
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from kornia.contrib import distance_transform

from .registry import register_model
from ..utils import configurable
from ..utils import get_iou
from ..backbone import build_backbone, Backbone
from ..body import build_xdecoder_head
from ..modules import sem_seg_postprocess, bbox_postprocess
from ..language import build_language_encoder
from ..language.loss import vl_similarity
from util.util import AverageMeter, count_params, init_log, DiceLoss
from nltk.stem.lancaster import LancasterStemmer
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.data import MetadataCatalog
from util.constants import COCO_PANOPTIC_CLASSES
from util.criterion import SetCriterion
import imgviz
from PIL import Image
st = LancasterStemmer()

def save_colored_mask(mask, save_path):
    lbl_pil = Image.fromarray(mask.astype(np.uint8), mode='P')
    color_map = imgviz.label_colormap()
    lbl_pil.putpalette(color_map.flatten())
    lbl_pil.save(save_path)
    
class SEEM_Model(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
            self,
            *,
            backbone: Backbone,
            sem_seg_head: nn.Module,
            criterion: nn.Module,
            losses: dict,
            num_queries: int,
            object_mask_threshold: float,
            overlap_threshold: float,
            metadata,
            task_switch: dict,
            phrase_prob: float,
            size_divisibility: int,
            sem_seg_postprocess_before_inference: bool,
            pixel_mean: Tuple[float],
            pixel_std: Tuple[float],
            # inference
            semantic_on: bool,
            panoptic_on: bool,
            instance_on: bool,
            test_topk_per_image: int,
            train_dataset_name: str,
            interactive_mode: str,
            interactive_iter: str,
            dilation_kernel: torch.Tensor,
    ):
        super().__init__()
        self.backbone = backbone

        self.sem_seg_head = sem_seg_head
        self.losses = losses
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        metadata = MetadataCatalog.get('coco_2017_train_panoptic')

        self.metadata = metadata
        if size_divisibility < 0:
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on

        self.task_switch = task_switch
        self.phrase_prob = phrase_prob

        self.test_topk_per_image = test_topk_per_image
        self.train_class_names = COCO_PANOPTIC_CLASSES + ["background"]
        self.interactive_mode = interactive_mode
        self.interactive_iter = interactive_iter

        self.criterion1 = nn.BCEWithLogitsLoss().cuda()
        self.criterion2 = DiceLoss(n_classes=1).cuda()

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

        self.register_buffer("dilation_kernel", dilation_kernel)

    @classmethod
    def from_config(cls, cfg):
        enc_cfg = cfg['MODEL']['ENCODER']
        dec_cfg = cfg['MODEL']['DECODER']

        openimage_switch = {'grounding': dec_cfg['OPENIMAGE']['GROUNDING'].get('ENABLED', False),
                            'mask': dec_cfg['OPENIMAGE'].get('ENABLED', False)}

        task_switch = {'bbox': dec_cfg.get('DETECTION', False),
                       'mask': dec_cfg.get('MASK', True),
                       'spatial': dec_cfg['SPATIAL'].get('ENABLED', False),
                       'grounding': dec_cfg['GROUNDING'].get('ENABLED', False),
                       'openimage': openimage_switch,
                       'visual': dec_cfg['VISUAL'].get('ENABLED', False),
                       'audio': dec_cfg['AUDIO'].get('ENABLED', False)}

        extra = {'task_switch': task_switch}
        backbone = build_backbone(cfg)
        lang_encoder = build_language_encoder(cfg)
        sem_seg_head = build_xdecoder_head(cfg, backbone.output_shape(), lang_encoder, extra=extra)

        # Training Settings.
        loss_weights = {}
        matcher = None
        losses = {}
        weight_dict = {}
        grd_weight = {}
        top_x_layers = {}
        criterion = None
        train_dataset_name = None
        phrase_prob = None
        # Loss parameters:
        deep_supervision = None
        no_object_weight = None

        interactive_mode = 'best'
        interactive_iter = 10
        dilation = 3
        dilation_kernel = torch.ones((1, 1, dilation, dilation), device=torch.cuda.current_device())

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "losses": losses,
            "num_queries": dec_cfg['NUM_OBJECT_QUERIES'],
            "object_mask_threshold": dec_cfg['TEST']['OBJECT_MASK_THRESHOLD'],
            "overlap_threshold": dec_cfg['TEST']['OVERLAP_THRESHOLD'],
            "metadata": MetadataCatalog.get('coco_2017_train_panoptic'),
            "size_divisibility": dec_cfg['SIZE_DIVISIBILITY'],
            "sem_seg_postprocess_before_inference": (
                    dec_cfg['TEST']['SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE']
                    or dec_cfg['TEST']['PANOPTIC_ON']
                    or dec_cfg['TEST']['INSTANCE_ON']
            ),
            "pixel_mean": cfg['INPUT']['PIXEL_MEAN'],
            "pixel_std": cfg['INPUT']['PIXEL_STD'],
            "task_switch": task_switch,
            "phrase_prob": phrase_prob,
            # inference
            "semantic_on": dec_cfg['TEST']['SEMANTIC_ON'],
            "instance_on": dec_cfg['TEST']['INSTANCE_ON'],
            "panoptic_on": dec_cfg['TEST']['PANOPTIC_ON'],
            "test_topk_per_image": cfg['MODEL']['DECODER']['TEST']['DETECTIONS_PER_IMAGE'],
            "train_dataset_name": train_dataset_name,
            "interactive_mode": interactive_mode,
            "interactive_iter": interactive_iter,
            "dilation_kernel": dilation_kernel,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs, targets, reference, ref_mask, id_indicator=None, mode='train'):
        if mode == 'train' or mode == ' val':
            losses_seg = self.forward_seg(batched_inputs, targets, reference, ref_mask, id_indicator, mode=mode)
            return losses_seg
        else:
            pred = self.forward_seg(batched_inputs, targets, reference, ref_mask, mode=mode)
            return pred

    def forward_seg(self, batched_inputs, targets, reference_img, ref_mask, id_indicator=None, mode='train'):
        images = [x.to(self.device) for x in batched_inputs]
        images = ImageList.from_tensors(images, self.size_divisibility)

        ref_images = [x.to(self.device) for x in reference_img]
        ref_images = ImageList.from_tensors(ref_images, self.size_divisibility)

        ref_masks = [x.to(self.device) for x in ref_mask]
        ref_masks = ImageList.from_tensors(ref_masks, self.size_divisibility)

        features = self.backbone.get_intermediate_layers(images.tensor.float(), 3)
        ref_features = self.backbone.get_intermediate_layers(ref_images.tensor.float(), 3)

        features = [v for k, v in features.items()]
        ref_features = [v for k, v in ref_features.items()]

        ref_information = (ref_features, ref_masks)
        query_information = features

        outputs = self.sem_seg_head.predictor(ref_information, query_information,
                                                          task='spatial')

        pred = F.interpolate(outputs["predictions_mask"], (targets.shape[-2], targets.shape[-1]), align_corners=True, mode='bilinear')
        # pred = torch.clamp(pred, min=1e-7, max=1-1e-7)
        
        targets = targets.cuda()
        targets = targets.float()

        # pred_1 = torch.argmax(pred, dim=1)
        # save_colored_mask(pred_1[0].cpu().numpy(), 'pred_1.png')

        # print(pred.sigmoid())

        # back_mask = (targets.sum(dim=1) == 0)
        # targets = torch.argmax(targets, dim=1)
        # # targets[back_mask] = 10
        # save_colored_mask(targets[0].cpu().numpy(), 'target.png')

        # print((targets[0]==2).sum())
        # print((targets[0] == 3).sum())
        # print((targets[0] == 10).sum())
        # save_colored_mask(targets[0].cpu().numpy(), 'target.png')

        if self.task_switch['spatial'] and mode != 'test':
            if id_indicator == None:
                losses = torch.tensor(0.0).cuda()
                for id in range(5):
                    losses += (self.criterion2(pred[:, id, :, :].unsqueeze(1).sigmoid(), targets[:, id, :, :].unsqueeze(1)) * 0.7 \
                            + self.criterion1(pred[:, id, :, :].unsqueeze(1), targets[:, id, :, :].unsqueeze(1)) * 0.3)

                del outputs
                return losses
        else:
            return pred


@register_model
def get_segmentation_model(cfg, **kwargs):
    return SEEM_Model(cfg)

