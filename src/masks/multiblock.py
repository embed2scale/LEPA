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

import math

from multiprocessing import Value

from logging import getLogger

import torch

_GLOBAL_SEED = 0
logger = getLogger()


class MaskCollator(object):

    def __init__(
        self,
        input_size=(224, 224),
        patch_size=16,
        enc_mask_scale=(0.2, 0.8),
        pred_mask_scale=(0.2, 0.8),
        aspect_ratio=(0.3, 3.0),
        nenc=1,
        npred=2,
        min_keep=4,
        allow_overlap=False,
        image_key='sample',
        target_key='target',
    ):
        super(MaskCollator, self).__init__()
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.patch_size = patch_size
        self.height, self.width = input_size[0] // patch_size, input_size[1] // patch_size
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.aspect_ratio = aspect_ratio
        self.nenc = nenc
        self.npred = npred
        self.min_keep = min_keep  # minimum number of patches to keep
        self.allow_overlap = allow_overlap  # whether to allow overlap b/w enc and pred masks
        self._itr_counter = Value('i', -1)  # collator is shared across worker processes
        self.img_key = image_key
        self.target_key = target_key
        
    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _sample_block_size(self, generator, scale, aspect_ratio_scale):
        _rand = torch.rand(1, generator=generator).item()
        # -- Sample block scale
        min_s, max_s = scale
        mask_scale = min_s + _rand * (max_s - min_s)
        max_keep = int(self.height * self.width * mask_scale)
        # -- Sample block aspect-ratio
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)
        # -- Compute block height and width (given scale and aspect-ratio)
        h = int(round(math.sqrt(max_keep * aspect_ratio)))
        w = int(round(math.sqrt(max_keep / aspect_ratio)))
        while h >= self.height:
            h -= 1
        while w >= self.width:
            w -= 1

        return (h, w)

    def _sample_block_mask(self, b_size, acceptable_regions=None):
        h, w = b_size

        def constrain_mask(mask, tries=0):
            """ Helper to restrict given mask to a set of acceptable regions """
            N = max(int(len(acceptable_regions)-tries), 0)
            for k in range(N):
                mask *= acceptable_regions[k]
        # --
        # -- Loop to sample masks until we find a valid one
        tries = 0
        timeout = og_timeout = 20
        valid_mask = False
        while not valid_mask:
            # -- Sample block top-left corner
            top = torch.randint(0, self.height - h, (1,))
            left = torch.randint(0, self.width - w, (1,))
            mask = torch.zeros((self.height, self.width), dtype=torch.int32)
            mask[top:top+h, left:left+w] = 1
            # -- Constrain mask to a set of acceptable regions
            if acceptable_regions is not None:
                constrain_mask(mask, tries)
            mask = torch.nonzero(mask.flatten())
            # -- If mask too small try again
            valid_mask = len(mask) > self.min_keep
            if not valid_mask:
                timeout -= 1
                if timeout == 0:
                    tries += 1
                    timeout = og_timeout
                    logger.warning(f'Mask generator says: "Valid mask not found, decreasing acceptable-regions [{tries}]"')
        mask = mask.squeeze()
        # --
        mask_complement = torch.ones((self.height, self.width), dtype=torch.int32)
        mask_complement[top:top+h, left:left+w] = 0
        # --
        return mask, mask_complement

    def __call__(self, batch):
        '''
        Create encoder and predictor masks when collating imgs into a batch
        # 1. sample enc block (size + location) using seed
        # 2. sample pred block (size) using seed
        # 3. sample several enc block locations for each image (w/o seed)
        # 4. sample several pred block locations for each image (w/o seed)
        # 5. return enc mask and pred mask
        '''
        if isinstance(batch, dict):
            B = len(batch[self.img_key])
            collated_batch = batch
        else:
            B = len(batch)
            collated_batch = torch.utils.data.default_collate(batch)
            collated_batch = {'sample': collated_batch[0],
                              'target': collated_batch[0],
                              'class': collated_batch[1],
                              'augmentation_params': {}}


        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        p_size = self._sample_block_size(
            generator=g,
            scale=self.pred_mask_scale,
            aspect_ratio_scale=self.aspect_ratio)
        e_size = self._sample_block_size(
            generator=g,
            scale=self.enc_mask_scale,
            aspect_ratio_scale=(1., 1.))

        collated_masks_pred, collated_masks_enc = [], []
        min_keep_pred = self.height * self.width
        min_keep_enc = self.height * self.width
        for _ in range(B):

            masks_p, masks_C = [], []
            for _ in range(self.npred):
                mask, mask_C = self._sample_block_mask(p_size)
                masks_p.append(mask)
                masks_C.append(mask_C)
                min_keep_pred = min(min_keep_pred, len(mask))
            collated_masks_pred.append(masks_p)

            acceptable_regions = masks_C
            try:
                if self.allow_overlap:
                    acceptable_regions= None
            except Exception as e:
                logger.warning(f'Encountered exception in mask-generator {e}')

            masks_e = []
            for _ in range(self.nenc):
                mask, _ = self._sample_block_mask(e_size, acceptable_regions=acceptable_regions)
                masks_e.append(mask)
                min_keep_enc = min(min_keep_enc, len(mask))
            collated_masks_enc.append(masks_e)

        collated_masks_pred = [[cm[:min_keep_pred] for cm in cm_list] for cm_list in collated_masks_pred]
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)
        # --
        collated_masks_enc = [[cm[:min_keep_enc] for cm in cm_list] for cm_list in collated_masks_enc]
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)

        return collated_batch, collated_masks_enc, collated_masks_pred

    def convert_large_mask_to_small(self, large_masks, large_patch_per_row, small_patch_per_row):
        # large_mask: list of 1D tensors with indices for the large mask eg for an image with size 256x256
        # large_patch_per_row: number of patches per row in the large mask
        # small_patch_per_row: number of patches per row in the small mask
        B = large_masks[0].size(0)
        num_discarded_per_row = large_patch_per_row - small_patch_per_row
        if num_discarded_per_row == 0:
            return large_masks  # No need to convert if sizes are the same
        masks_small = []
        min_keep = small_patch_per_row * small_patch_per_row
        for mask_indices in large_masks:
            batch = []
            for i in range(B):
                mask_tensor = torch.zeros(large_patch_per_row * large_patch_per_row, dtype=torch.int32)
                mask_tensor[mask_indices[i]] = 1
                mask_tensor = mask_tensor.view(large_patch_per_row, large_patch_per_row)
                
                start = num_discarded_per_row // 2
                end = -start if num_discarded_per_row % 2 == 0 else -(start + 1)
                mask_tensor = mask_tensor[start:end, start:end]
                mask_small_indices = torch.nonzero(mask_tensor.flatten()).squeeze()

                min_keep = min(min_keep, len(mask_small_indices))
                batch.append(mask_small_indices)
            masks_small.append(batch)
        masks_small = [torch.stack([ms[:min_keep] for ms in ms_list],dim=0) for ms_list in masks_small]
        # Convert to tensor
        return masks_small


class RandomMaskCollator(object):

    def __init__(
        self,
        input_size=(224, 224),
        patch_size=16,
        enc_mask_scale=(0.2, 0.8),
        pred_mask_scale=(0.2, 0.8),
        nenc=1,
        npred=2,
        min_context_tokens=4,
        allow_overlap=False,
        image_key='sample',
        target_key='target',
        **kwargs
    ):
        super(RandomMaskCollator, self).__init__()
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.patch_size = patch_size
        self.height, self.width = input_size[0] // patch_size, input_size[1] // patch_size
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.nenc = nenc
        self.npred = npred
        self.min_keep_context = min_context_tokens  # minimum number of patches to keep in the context
        self.allow_overlap = allow_overlap  # whether to allow overlap b/w enc and pred masks
        self._itr_counter = Value('i', -1)  # collator is shared across worker processes
        self.img_key = image_key
        self.target_key = target_key

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v
    
    def _generate_pred_mask(self, g, n_pred_patches, min_keep_pred):

        masks_p, masks_C = [], []
        for _ in range(self.npred):
            perm = torch.randperm(n=self.height * self.width, generator=g)
            mask = perm[:n_pred_patches]
            masks_p.append(mask)
            masks_C.append(perm[:n_pred_patches])
            min_keep_pred = min(min_keep_pred, len(mask))

        if self.allow_overlap:
            acceptable_regions = torch.arange(self.height * self.width, dtype=torch.int64)
        else:
            # join the complement of the pred masks into one list of acceptable regions
            acceptable_regions = torch.ones(self.height * self.width, dtype=torch.bool)
            for mask in masks_C:
                acceptable_regions[mask] = False
            acceptable_regions = torch.nonzero(acceptable_regions).squeeze()

        return masks_p, masks_C, acceptable_regions, min_keep_pred


    def __call__(self, batch):
        '''
        Create encoder and predictor masks when collating imgs into a batch
        # 1. randomly sample enc patch locations
        # 2. randomly sample pred patch locations
        # 3. return enc mask and pred mask
        '''
        if isinstance(batch, dict):
            B = len(batch[self.img_key])
            collated_batch = batch
        else:
            B = len(batch)
            collated_batch = torch.utils.data.default_collate(batch)
            collated_batch = {'sample': collated_batch[0],
                              'target': collated_batch[1],
                              'augmentation_params': {}}


        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)

        collated_masks_pred, collated_masks_enc = [], []
        min_keep_pred = self.height * self.width
        min_keep_enc = self.height * self.width
        for _ in range(B):

            # Generate prediction masks
            n_pred_patches = torch.randint(
                int(self.pred_mask_scale[0] * self.height * self.width),
                int(self.pred_mask_scale[1] * self.height * self.width),
                (1,),
                generator=g
            ).item() + 1

            masks_e, masks_p = [], []
            masks_C = []
            acceptable_regions = torch.arange(1)
            tries = 0
            while len(acceptable_regions) < self.min_keep_context:
                # Regenerate prediction masks if not enough acceptable regions
                n_pred_patches -= 1
                masks_p, masks_C, acceptable_regions, min_keep_pred = self._generate_pred_mask(g, n_pred_patches=n_pred_patches, min_keep_pred=min_keep_pred)
                tries += 1
                if tries > 100:
                    raise ValueError(f"Could not generate enough acceptable regions after {tries} tries. "
                                     f"Consider adjusting the pred_mask_scale or min_keep_context parameters.")
            # print(f"Generated {len(acceptable_regions)} acceptable regions after {tries} tries for seed {seed}")
            
            collated_masks_pred.append(masks_p)
            n_enc_patches = torch.randint(
                int(self.enc_mask_scale[0] * self.height * self.width),
                int(self.enc_mask_scale[1] * self.height * self.width),
                (1,),
                generator=g
            ).item() + 1
            for _ in range(self.nenc):
                perm = torch.randperm(len(acceptable_regions), generator=g)
                mask = acceptable_regions[perm[:n_enc_patches]]
                masks_e.append(mask)
                min_keep_enc = min(min_keep_enc, len(mask))

            collated_masks_enc.append(masks_e)

        collated_masks_pred = [[cm[:min_keep_pred] for cm in cm_list] for cm_list in collated_masks_pred]
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)
        # --
        collated_masks_enc = [[cm[:min_keep_enc] for cm in cm_list] for cm_list in collated_masks_enc]
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)

        return collated_batch, collated_masks_enc, collated_masks_pred




    def convert_large_mask_to_small(self, large_masks, large_patch_per_row, small_patch_per_row):
        # large_mask: list of 1D tensors with indices for the large mask eg for an image with size 256x256
        # large_patch_per_row: number of patches per row in the large mask
        # small_patch_per_row: number of patches per row in the small mask
        B = large_masks[0].size(0)
        num_discarded_per_row = large_patch_per_row - small_patch_per_row
        if num_discarded_per_row == 0:
            return large_masks  # No need to convert if sizes are the same
        masks_small = []
        min_keep = small_patch_per_row * small_patch_per_row
        for mask_indices in large_masks:
            batch = []
            for i in range(B):
                mask_tensor = torch.zeros(large_patch_per_row * large_patch_per_row, dtype=torch.int32)
                mask_tensor[mask_indices[i]] = 1
                mask_tensor = mask_tensor.view(large_patch_per_row, large_patch_per_row)
                if num_discarded_per_row > 0:
                    start = num_discarded_per_row // 2
                    end = -start if num_discarded_per_row % 2 == 0 else -(start + 1)
                    mask_tensor = mask_tensor[start:end, start:end]
                mask_small_indices = torch.nonzero(mask_tensor.flatten()).squeeze()
                min_keep = min(min_keep, len(mask_small_indices))
                batch.append(mask_small_indices)
            masks_small.append(batch)
        masks_small = [torch.stack([ms[:min_keep] for ms in ms_list],dim=0) for ms_list in masks_small]
        # Convert to tensor
        return masks_small
