# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found at https://github.com/facebookresearch/ijepa/blob/main/LICENSE
#
# Modifications and additional code:
# Copyright (c) 2026 Forschungszentrum Jülich GmbH
# Licensed under the Apache License, Version 2.0.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

import torch
import os
import wandb
import torch.distributed as dist

def gpu_timer(closure, log_timings=True):
    """ Helper to time gpu-time to execute closure() """
    log_timings = log_timings and torch.cuda.is_available()

    elapsed_time = -1.
    if log_timings:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

    result = closure()

    if log_timings:
        end.record()
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end)

    return result, elapsed_time

def get_param_norm_to_update_ratio(named_params, old_named_params):
    """ Computes the ratio of parameter norm to update norm for each parameter group.
        Returns a dictionary with parameter names as keys and their respective ratios.
    """
    ratios = {}
    for (name, param), (_, old_param) in zip(named_params, old_named_params):
        if param.grad is not None:
            param_norm = param.data.norm().item()
            update_norm = (param.data - old_param.data).norm().item()
            if update_norm > 0:
                ratios[name] = param_norm / update_norm
            else:
                ratios[name] = float('inf')  # Avoid division by zero
    return ratios

class Logger(object):
    
    def __init__(self, train_metrics, val_metrics, use_wandb=False, entity='', project='', directory='logs/', wandb_name='', imgs_per_epoch=0):

        rank = dist.get_rank() if dist.is_initialized() else 0
        print(f'WandbLogger: is_primary={rank==0}, rank={rank}')
        self.use_wandb = use_wandb
        if rank != 0:
            self.use_wandb = False  # disable wandb logging for non-primary ranks
        if self.use_wandb:
            run = wandb.init(
                entity=entity,
                project=project,
                dir=directory,
                mode='offline',
                id=wandb_name,
                settings=wandb.Settings(
                    x_label=f"rank_{dist.get_rank()}" if dist.is_initialized() else "rank_0",
                    mode="shared",
                    x_primary=rank==0,
                    # x_stats_gpu_device_ids=[os.environ.get('LOCAL_RANK', '0')],
                )
            )

        train_csv_file = os.path.join(directory, f'train_r{rank}.csv')
        train_metrics = ['epoch', 'itr'] + train_metrics
        self.train_csv_logger = CSVLogger(train_csv_file, *[(f'%.5f', name) for name in train_metrics])

        val_csv_file = os.path.join(directory, f'val_r{rank}.csv')
        val_metrics = ['epoch'] + val_metrics
        self.val_csv_logger = CSVLogger(val_csv_file, *[(f'{name}: %.5f', name) for name in val_metrics])

        self.global_step = 0
        self.imgs_per_epoch = imgs_per_epoch

    def log_config(self, config):
        if self.use_wandb:
            wandb.config.update(config)

    def log_train(self, epoch, itr, **kwargs):
        if self.global_step % self.imgs_per_epoch != itr:
            self.global_step += 1
            assert self.global_step % self.imgs_per_epoch == itr, \
                f'global_step {self.global_step} not matching imgs_per_epoch {self.imgs_per_epoch} and itr {itr}'

        log_dict = {"epoch": epoch, "itr": itr}
        log_dict.update(kwargs)
        if self.train_csv_logger.is_something_to_log(kwargs.keys()):
            self.train_csv_logger.log(*[log_dict[name.split(':')[0]] for name in self.train_csv_logger.names if name in log_dict])
        if self.use_wandb:
            wandb.log(kwargs, step=self.global_step)
            wandb.log({"epoch": epoch, "itr": itr, **kwargs}, step=self.global_step)

    def log_val(self, epoch, **kwargs):
        log_dict = {"epoch": epoch}
        log_dict.update(kwargs)
        if self.val_csv_logger.is_something_to_log(kwargs.keys()):
            self.val_csv_logger.log(*[log_dict[name.split(':')[0]] for name in self.val_csv_logger.names])
        if self.use_wandb:
            wandb.log(kwargs, step=self.global_step)
            wandb.log({"epoch": epoch, **kwargs}, step=self.global_step)


class CSVLogger(object):

    def __init__(self, fname, *argv):
        self.fname = fname
        self.types = []
        self.names = []
        # -- print headers
        with open(self.fname, '+a') as f:
            for i, v in enumerate(argv, 1):
                self.types.append(v[0])
                self.names.append(v[1])
                if i < len(argv):
                    print(v[1], end=',', file=f)
                else:
                    print(v[1], end='\n', file=f)

    def is_something_to_log(self, names):
        """ Check if there is something to log in the given names """
        for name in names:
            if name in self.names:
                return True
        return False

    def log(self, *argv):
        with open(self.fname, '+a') as f:
            for i, tv in enumerate(zip(self.types, argv), 1):
                end = ',' if i < len(argv) else '\n'
                print(tv[0] % tv[1], end=end, file=f)


class AverageMeter(object):
    """computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.max = float('-inf')
        self.min = float('inf')
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        try:
            self.max = max(val, self.max)
            self.min = min(val, self.min)
        except Exception:
            pass
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def grad_logger(named_params):
    stats = AverageMeter()
    stats.first_layer = None
    stats.last_layer = None
    for n, p in named_params:
        if (p.grad is not None) and not (n.endswith('.bias') or len(p.shape) == 1):
            grad_norm = float(torch.norm(p.grad.data))
            stats.update(grad_norm)
            if 'qkv' in n:
                stats.last_layer = grad_norm
                if stats.first_layer is None:
                    stats.first_layer = grad_norm
    if stats.first_layer is None or stats.last_layer is None:
        stats.first_layer = stats.last_layer = 0.
    return stats
