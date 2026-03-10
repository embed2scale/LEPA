import os

import torch

def get_dataloader(
    root_path,
    eval_root_path,
    mask_collator,
    batch_size,
    chunk_size,
    input_size,
    crop_size,
    bands,
    scaling,
    mean,
    std,
    world_size,
    rank,
    num_workers,
    pin_mem,
    **kwargs
):
    """
    Selects and creates the appropriate dataloader and eval loader based on the dataset type.
    Returns: unsupervised_loader, unsupervised_sampler, eval_loader, eval_sampler, ipe, ipve
    """
    if 'HLS' in root_path:
        from src.datasets.hls_interpol_dataset import build_hls_dataloader
        unsupervised_loader, unsupervised_sampler = build_hls_dataloader(
            data_dir=root_path,
            data_aug=False,
            collator=mask_collator,
            batch_size=batch_size,
            chunk_size=chunk_size,
            input_shape=[1, input_size, input_size],
            output_size=[1, crop_size, crop_size] if isinstance(crop_size,int) else crop_size,
            bands=bands,
            scaling=scaling,
            mean=mean,
            std=std,
            world_size=world_size,
            rank=rank,
            num_workers=num_workers,
            pin_memory=pin_mem,
            shuffle=True,
        )
        ipe = len(unsupervised_loader)
        eval_loader, eval_sampler = build_hls_dataloader(
            data_dir=eval_root_path,
            data_aug=False,
            collator=mask_collator,
            batch_size=batch_size*2,
            chunk_size=chunk_size,
            input_shape=[1, input_size, input_size],
            output_size=[1, crop_size, crop_size] if isinstance(crop_size,int) else crop_size,
            bands=bands,
            scaling=scaling,
            mean=mean,
            std=std,
            world_size=world_size,
            rank=rank,
            num_workers=num_workers,
            pin_memory=pin_mem,
            shuffle=False,
        )
        ipve = len(eval_loader)
    elif 'imagenet1k' in root_path.lower():
        from src.datasets.imagenet1k import make_imagenet1k
        from src.transforms import make_transforms
        transforms = make_transforms(
            crop_size=input_size,
            crop_scale=kwargs.get('crop_scale', (0.3, 1.0)),
        )
        dataset, unsupervised_loader, unsupervised_sampler = make_imagenet1k(
            transform=transforms,
            batch_size=batch_size,
            collator=mask_collator,
            pin_mem=pin_mem,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
            root_path=root_path,
            image_folder='',
            training=True,
            copy_data=False,
            drop_last=True
        )
        ipe = len(unsupervised_loader)
        eval_dataset, eval_loader, eval_sampler = make_imagenet1k(
            transform=transforms,
            batch_size=batch_size,
            collator=mask_collator,
            pin_mem=pin_mem,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
            root_path=eval_root_path,
            image_folder='',
            training=False,
            copy_data=False,
            drop_last=False
        )
        ipve = len(eval_loader)
    elif 'terramesh' in root_path.lower():
        from src.datasets.terramesh import get_terramesh_dataloader
        unsupervised_loader, unsupervised_sampler = get_terramesh_dataloader(
            root_path=root_path,
            modalities="S2L2A",
            split='train',
            batch_size=batch_size,
            input_size=input_size,
            shuffle=True,
            num_workers=num_workers,
            collator=mask_collator,
            )
        ipe = 9058304 // (batch_size * world_size)
        # ipe = 9060352 // (batch_size * world_size)
        eval_loader, eval_sampler = get_terramesh_dataloader(
            root_path=eval_root_path,
            modalities="S2L2A",
            split='val',
            batch_size=int(batch_size),
            input_size=input_size,
            shuffle=False,
            num_workers=num_workers,
            collator=mask_collator,
            )
        ipve = 89087 // (batch_size * world_size)
    else:
        raise ValueError(f'Unknown dataset: {root_path}')
    

    class DummySampler(torch.utils.data.Sampler):
        """
        Dummy sampler that does not sample anything.
        """
        def __iter__(self):
            return iter([])

        def __len__(self):
            return 0
        
        def set_epoch(self, epoch: int):
            """
            Set the epoch for the sampler.
            This is a no-op for the dummy sampler.
            """
            pass
    if unsupervised_sampler is None:
        unsupervised_sampler = DummySampler()
    if eval_sampler is None:
        eval_sampler = DummySampler()
    return unsupervised_loader, unsupervised_sampler, eval_loader, eval_sampler, ipe, ipve
