# --------------------------------------------------------
# Images Speak in Images: A Generalist Painter for In-Context Visual Learning (https://arxiv.org/abs/2212.02499)
# Github source: https://github.com/baaivision/Painter
# Copyright (c) 2022 Beijing Academy of Artificial Intelligence (BAAI)
# Licensed under The MIT License [see LICENSE for details]
# By Xinlong Wang, Wen Wang
# Based on MAE, BEiT, detectron2, Mask2Former, bts, mmcv, mmdetetection, mmpose, MIRNet, MPRNet, and Uformer codebases
# --------------------------------------------------------'

import math
import sys
from typing import Iterable

import torch

import util.misc as misc
import util.lr_sched as lr_sched

import numpy as np
import wandb

def get_loss_scale_for_deepspeed(optimizer):
    # optimizer = model.optimizer
    loss_scale = None
    if hasattr(optimizer, 'loss_scale'):
        loss_scale = optimizer.loss_scale
    elif hasattr(optimizer, 'cur_scale'):
        loss_scale = optimizer.cur_scale
    return loss_scale, optimizer._global_grad_norm
    # return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    global_rank=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    wandb_images = []

    for data_iter_step, (samples, targets, references, ref_masks) in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)  #####  2, 3, 448, 448
        targets = targets.to(device, non_blocking=True)  #####  2, 3, 896, 448
        references = references.to(device, non_blocking=True)
        ref_masks = ref_masks.to(device, non_blocking=True)

        loss = model(samples, targets, references, ref_masks)
        # loss_1 = sum(loss.values())

        loss_value = loss.item()

        if loss_scaler is None:
            loss /= accum_iter
            loss.backward()

            if (data_iter_step + 1) % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad()
        else:
            loss /= accum_iter
            grad_norm = loss_scaler(loss, optimizer, clip_grad=args.clip_grad,
                                    parameters=model.parameters(),
                                    update_grad=(data_iter_step + 1) % accum_iter == 0)
            if (data_iter_step + 1) % accum_iter == 0:
                optimizer.zero_grad()
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

        if data_iter_step % 100 == 0:
            misc.save_model(
                args=args, model=model, model_without_ddp=model, optimizer=optimizer,
                loss_scaler=None, epoch=epoch)

    if global_rank == 0 and args.log_wandb and len(wandb_images) > 0:
        wandb.log({"Training examples": wandb_images})

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_pt(data_loader, model, device, epoch=None, global_rank=None, args=None):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    # switch to evaluation mode
    model.eval()
    wandb_images = []
    for batch in metric_logger.log_every(data_loader, 10, header):
        samples = batch[0]
        targets = batch[1]
        references = batch[2]
        ref_masks = batch[3]

        # with torch.cuda.amp.autocast():
        loss = model(samples, targets, references, ref_masks)

        metric_logger.update(loss=loss.item())

    metric_logger.synchronize_between_processes()
    print('Val loss {losses.global_avg:.3f}'.format(losses=metric_logger.loss))

    out = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    if global_rank == 0 and args.log_wandb:
        wandb.log({**{f'test_{k}': v for k, v in out.items()},'epoch': epoch})
        if len(wandb_images) > 0:
            wandb.log({"Testing examples": wandb_images[::2][:20]})
    return out
