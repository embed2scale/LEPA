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

import datetime
import os

import torch
import torch.distributed as dist

from logging import getLogger

logger = getLogger()


def init_distributed():

    try:
        local_rank = int(os.environ["LOCAL_RANK"])
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    except KeyError:
        local_rank = 0
        rank = 0
        world_size = 1
        
    print(f'Using SLURM vars: rank={rank}, world_size={world_size}, local_rank={local_rank}')

    if world_size > 1:
        print(f'Initializing distributed training: rank={rank}, world_size={world_size}, local_rank={local_rank}')
        dist.init_process_group("cpu:gloo,cuda:nccl", timeout=datetime.timedelta(seconds=300))
        torch.cuda.device(local_rank)
        torch.cuda.empty_cache()
        local_device = local_rank
    else:
        if not torch.cuda.is_available():
            print(f'Initializing distributed training (CPU only): rank={rank}, world_size={world_size}')
            local_device = 'cpu'
        else:
            print(f'Initializing distributed training (only one GPU): rank={rank}, world_size={world_size}')
            local_device = 'cuda'


    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

    return world_size, rank, local_device

class AllGather(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            x = x.contiguous()
            outputs = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
            dist.all_gather(outputs, x)
            return torch.cat(outputs, 0)
        return x

    @staticmethod
    def backward(ctx, grads):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            s = (grads.shape[0] // dist.get_world_size()) * dist.get_rank()
            e = (grads.shape[0] // dist.get_world_size()) * (dist.get_rank() + 1)
            grads = grads.contiguous()
            dist.all_reduce(grads)
            return grads[s:e]
        return grads


class AllReduceSum(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            x = x.contiguous()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads


class AllReduce(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            x = x.contiguous() / dist.get_world_size()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads
