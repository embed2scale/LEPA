import glob
import os
from collections.abc import Sequence

import numpy as np
import torch
import torchvision.transforms as T
import xarray as xr
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision.transforms.functional import resized_crop
from src.transforms import random_resize_and_rotate

HLS_MIN = 0.0
HLS_MAX = 10000.0

SCALING_TYPES = ['none', 'standard', 'norm_clip', 'log1p']

class HLSInterpolDataset(Dataset):

    def __init__(self, data_stores: list[xr.Dataset], bands: list[str], mean: list[float] | None = None,
                 std: list[float] | None = None, in_shape: list[int] | None = None, out_shape: list[int] | None = None,
                 scaling: str = 'standard', data_augmentation=False, chunk_size: int = 1, max_images: int | None = None):
        """ HLSDataset class to load preprocessed zarr files.

        Args:
            data_stores: list of loaded xr.Dataset for each zarr store.
            bands: list containing the band names to load.
            mean: list containing the mean values for each band (same length as *bands*); ignored when scaling = 'none'.
            std: list containing the std values for each band (same length as *bands*); ignored when scaling = 'none'.
            out_shape: (list) final shape of the data (limit to this size). If 2D shape is given,
                assumes size 1 in temporal dimension.
            scaling: (str) one of 'standard', 'norm_clip' or 'none'.
            data_augmentation: whether to apply data augmentation transforms (RandomHorizontalFlip + RandomVerticalFlip).
            chunk_size: (int) size of the slice to get each batch.
            max_images: (int, optional) maximum number of images (chunks) to include in the dataset.
        """
        self.data_stores = data_stores
        self.samples_per_store = [store.sizes['sample'] for store in data_stores]
        # Only complete chunks will be loaded! Last incomplete chunks in each store will be skipped
        self.chunks_per_store = [s // chunk_size for s in self.samples_per_store]

        # compute the last index that each store contains, exclusive.
        # to find index i, find the first position in this list with value > i
        self.end_index = np.cumsum(self.chunks_per_store).tolist()

        self.total_chunks = sum(self.chunks_per_store)
        self.num_examples = self.total_chunks * chunk_size
        # make sure we have the right number of samples
        assert self.end_index[-1] == self.total_chunks
        self.data_shape = [data_stores[0].sizes['time'], data_stores[0].sizes['y'], data_stores[0].sizes['x']]
        if in_shape is None:
            in_shape = self.data_shape
        if isinstance(in_shape, int):
            in_shape = [1, in_shape, in_shape]
        assert type(in_shape) in [tuple, list] and len(in_shape) in [2, 3], f"Incorrect shape format {in_shape}"
        if len(in_shape) == 2:   # Handle 2D shape
            in_shape = [1] + in_shape

        if out_shape is None:
            out_shape = self.data_shape
        assert type(out_shape) in [tuple, list] and len(out_shape) in [2, 3], f"Incorrect shape format {out_shape}"
        if len(out_shape) == 2:   # Handle 2D shape
            out_shape = [1] + out_shape

        if not all([in_shape[i] <= self.data_shape[i] for i in range(3)]):
            raise ValueError(f"shape {in_shape} too large for data {self.data_shape}.")
        if not all([out_shape[i] <= in_shape[i] for i in range(3)]):
            raise ValueError(f"shape {out_shape} too large for input shape {in_shape}.")

        if not set(bands).issubset(set(data_stores[0].band.values.tolist())):
            raise KeyError(f"Not all requested bands {bands} are present in the dataset.")

        if scaling not in SCALING_TYPES:
            raise ValueError(f"Invalid scaling {scaling}. Please select one of {SCALING_TYPES}.")
        if scaling not in ['none', 'log1p'] and (mean is None or std is None):
            raise ValueError(f"Please provide valid mean and std values for {scaling} type.")

        self.out_shape = out_shape
        self.in_shape = in_shape
        self.bands = bands
        self.data_augmentation = data_augmentation
        self.mean = mean
        self.std = std
        self.scaling = scaling
        self.chunk_size = chunk_size
        self.transform = self._get_transforms()
        # Select frames if required number < number of frames in self.data
        if self.out_shape[0] < self.data_shape[0]:
            self.frame_selection = IdxSelection(size=self.out_shape[0], random=self.data_augmentation)
        else:
            self.frame_selection = None

        # Limit the number of images/chunks if max_images is set
        self._max_chunks = None
        if max_images is not None:
            max_chunks = max_images // self.chunk_size
            if max_chunks < 1:
                raise ValueError("max_images is less than chunk_size, no data would be loaded.")
            self._max_chunks = min(max_chunks, self.total_chunks)
        else:
            self._max_chunks = self.total_chunks

    def get_chunk(self, index):
        # find store that holds this index (do it linearly, there are not that many stores)
        store_index = 0
        while store_index < len(self.end_index) and index >= self.end_index[store_index]:
            store_index += 1

        if store_index >= len(self.end_index):
            raise Exception(f"Index {store_index} not found in stores. Max value is {self.end_index[-1] - 1}")

        store = self.data_stores[store_index]

        # adjust the index so that it is adapted to this store
        if store_index > 0:
            offset = index - self.end_index[store_index - 1]
        else:
            offset = index

        return store.isel(sample=slice(offset * self.chunk_size, offset * self.chunk_size + self.chunk_size))

    @property
    def num_chunks(self):
        return self._max_chunks

    def _get_transforms(self):
        """ Returns the transforms to apply to the data. This step does NOT crop to the correct size"""

        transforms = []

        if self.data_augmentation:
            transforms.extend([
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip()])
            transforms.append(T.RandomCrop(self.in_shape[-2:]))  # Crop to input shape
        else:
            transforms.append(T.CenterCrop(self.in_shape[-2:]))

        if self.scaling == 'standard':
            transforms.append(Normalize(mean=self.mean, std=self.std))
        elif self.scaling == 'norm_clip':
            transforms.append(NormalizeClip(mean=self.mean, std=self.std))
        elif self.scaling == 'log1p':
            transforms.append(Log1pScaling(scale=HLS_MAX))


        return T.Compose(transforms)

    def _get_samples(self, chunk: xr.Dataset) -> np.ndarray:
        def get_slice():
            return chunk['bands'].sel(band=self.bands)

        values = np.concatenate([get_slice()], axis=0, dtype=np.float32)   # channels first

        return values

    def _get_date_time(self, chunk: xr.Dataset) -> np.ndarray:
        dates = chunk["time_"]

        return self._parse_dates(dates)

    def _get_latlon(self, chunk: xr.Dataset) -> np.ndarray:
        latlon = chunk[["center_lat", "center_lon"]]

        return np.stack([latlon["center_lat"], latlon["center_lon"]], axis=-1, dtype=np.float32)

    def preprocess(self, sample: torch.Tensor) -> torch.Tensor:
        # Clip values to acceptable range
        sample = torch.clip(sample, min=HLS_MIN, max=HLS_MAX)
        sample = self.transform(sample).contiguous()

        if len(sample.shape) != 5:
            raise ValueError(f"Expected sample shape (B, C, T, H, W), got {sample.shape}.")
        if sample.shape[2] != 1:
            raise ValueError(f"Expected sample shape with T=1, got {sample.shape[2]}.")
        sample = sample.squeeze(2)

        return sample


    def __len__(self):
        return self._max_chunks

    def __getitem__(self, index):
        if index >= self._max_chunks:
            raise IndexError(f"Index {index} out of bounds for dataset with {self._max_chunks} chunks.")
        chunk = self.get_chunk(index)
        sample = torch.from_numpy(self._get_samples(chunk))   # (N, C, T, H, W)
        date_time = torch.from_numpy(self._get_date_time(chunk))  # (N, T, 2)
        latlon = torch.from_numpy(self._get_latlon(chunk))  # (N, 2)

        if self.frame_selection is not None:
            idx = self.frame_selection(in_size=sample.shape[-3])
            sample = sample[..., idx, :, :].contiguous()
            date_time = date_time[..., idx, :].contiguous()

        sample = self.preprocess(sample)

        target, augmentation_params = random_resize_and_rotate(sample, self.out_shape)

        return {
            "sample": sample,
            "temporal_coords": date_time,
            "location_coords": latlon,
            "target": target,
            "augmentation_params": augmentation_params,
        }

    def _parse_dates(self, dates: xr.DataArray):
        years = dates.dt.year
        days = dates.dt.dayofyear - 1

        return np.stack([years, days], axis=-1, dtype=np.float32)

    def __repr__(self):
        return f"{self.__class__.__name__}(examples: {self.num_examples}, " \
               f"chunks: {len(self)}, shape: sample - {self.out_shape}; temporal_coords - 2)"


class RandomResizedCrop(T.RandomResizedCrop):
    """ RandomResizeCrop on H, W for data with shape (B, ..., H, W). """

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        shape = img.size()
        img = img.reshape(shape[0], -1, shape[-2], shape[-1])   # flatten channels + T if present
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        img = resized_crop(img, i, j, h, w, self.size, self.interpolation, antialias=self.antialias)
        img = img.reshape(shape)

        return img


class IdxSelection(torch.nn.Module):
    """ Selects indices from a given list size (subsample).

        If random=True, random indices are selected, but they are ordered after.
        If random=False, center indices are selected.

    Args:
        size: (int) number of indices to select.
        random: (bool) If True, the selection is random, otherwise, the center indices are selected.
    """

    def __init__(self, size: int, random: bool = False):
        super().__init__()
        self.size = size
        self.random = random

    def forward(self, in_size: int) -> slice | list[int]:
        """
        Args:
            in_size (int): current size to subsample.

        Returns:
            slice or list with selected indices.
        """

        if in_size == self.size:
            return slice(0, in_size)

        if in_size < self.size:
            raise ValueError(f"Required size {self.size} is larger than input dim {in_size}")

        if self.random:
            idx, _ = torch.randperm(in_size)[:self.size].sort()
            idx = idx.tolist()
        else:
            start = int(round((in_size - self.size) / 2.0))
            stop = start + self.size
            idx = slice(start, stop)

        return idx

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, random={self.random})"


class Normalize(torch.nn.Module):
    """ Normalize on channels for data with shape (B, C, ..., H, W).
        Based on the original torchvision transform Normalize.

        Given mean: (mean[1], ..., mean[n]) and std: (std[1], ..., std[n]) for N channels, normalize each channel as:
            output[channel] = (input[channel] - mean[channel]) / std[channel]
    """
    def __init__(self, mean: Sequence[float], std: Sequence[float], inplace: bool = False):
        super().__init__()
        self.mean = list(mean)
        self._check_std(std)
        self.std = list(std)
        self.inplace = inplace

    @staticmethod
    def _check_std(std):
        if isinstance(std, (tuple, list)):
            div_zero = not all(std)
        elif isinstance(std, (int, float)):
            div_zero = std == 0
        else:
            div_zero = False
        if div_zero:
            raise ValueError("std evaluated to zero, leading to division by zero.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_floating_point():
            raise TypeError(f"Input tensor should be a float tensor. Got {x.dtype}.")

        dtype = x.dtype
        device = x.device
        mean = torch.as_tensor(self.mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.std, dtype=dtype, device=device)
        shape = [-1] + [1] * (x.ndim - 2)

        mean = mean.view(*shape)
        std = std.view(*shape)

        if self.inplace:
            x = x.sub_(mean)
        else:
            x = x.sub(mean)

        return x.div_(std)

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + f"(mean={[round(m, 4) for m in self.mean]}"
        format_string += f", std={[round(s, 4) for s in self.std]}"
        return format_string


class NormalizeClip(torch.nn.Module):
    """ Based on https://github.com/sustainlab-group/SatMAE/blob/main/util/datasets.py#L349 """
    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        super().__init__()
        self.min = [mean[i] - 2 * std[i] for i in range(len(mean))]
        self.max = [mean[i] + 2 * std[i] for i in range(len(mean))]

    def forward(self, x: torch.Tensor):
        dtype = x.dtype
        device = x.device
        min_value = torch.as_tensor(self.min, dtype=dtype, device=device)
        max_value = torch.as_tensor(self.max, dtype=dtype, device=device)

        shape = [-1] + [1] * (x.ndim - 2)
        min_value = min_value.view(*shape)
        max_value = max_value.view(*shape)

        x = (x - min_value) / (max_value - min_value)
        x = torch.clip(x, 0.0, 1.0)
        return x

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + f"(min={[round(m, 4) for m in self.min]}"
        format_string += f", max={[round(s, 4) for s in self.max]}"
        return format_string


class Log1pScaling(torch.nn.Module):
    def __init__(self, scale: float = 1.0):
        super().__init__()
        if scale <= 0:
            raise ValueError("scale should be > 0.")
        self.scale = scale

    def forward(self, x: torch.Tensor):
        x = torch.div(x, self.scale)
        x = torch.log1p(x)
        return x

    def __repr__(self) -> str:
        return self.__class__.__name__ + f"(scale={self.scale})"


def collate_fn(data: list[dict], batch_size: int):

    keys = list(data[0].keys())
    big_batch = {k: torch.concat([sample[k] for sample in data], dim=0) for k in keys}
    random_idx = torch.randperm(big_batch[keys[0]].shape[0])[:batch_size]

    batch = {k: v[random_idx] for k, v in big_batch.items()}

    return batch


class AttributePreservingSubset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        if hasattr(dataset, 'total_chunks'):
            self.num_examples = len(indices) * (sum(dataset.samples_per_store) // dataset.total_chunks)
        else:
            self.num_examples = len(indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

    def __getattr__(self, name):
        return getattr(self.dataset, name)

class collate_fn_mask(torch.nn.Module):
    """ Collate function for HLS dataloader that returns a batch of masks. """

    def __init__(self, batch_size: int, collator=None):
        super().__init__()
        self.batch_size = batch_size
        self.collator = collator

    def forward(self, data: list[dict]) -> dict:
        """ Collate function for HLS dataloader that returns a batch of masks. """
        keys = list(data[0].keys())
        big_batch = {k: torch.concat([sample[k] for sample in data], dim=0) for k in keys if k != 'augmentation_params'}
        random_idx = torch.randperm(big_batch[keys[0]].shape[0])[:self.batch_size]

        batch = {k: v[random_idx] for k, v in big_batch.items()}
        if 'augmentation_params' in keys:
            batch['augmentation_params'] = {k: torch.concat([sample['augmentation_params'][k] for sample in data], dim=0)[random_idx]
                                            for k in data[0]['augmentation_params'].keys()}
            
        if self.collator is not None:
            return self.collator(batch)
        return batch

def build_hls_dataloader(
        batch_size,
        data_dir: str,
        data_aug: bool,
        collator=None,
        chunk_size=16, 
        input_shape=[1, 256, 256],
        output_size=[1, 224, 224],
        bands=["B02", "B03", "B04", "B05", "B06", "B07"],
        scaling="standard",
        mean=[0]*6, 
        std=[1]*6,
        max_images=None,
        world_size=1,
        rank=0,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2,
        shuffle: bool = True,
        subset_indices: list[int] | None = None):
    """ Wrapper to build dataloader for models that use HLSDataset. """

    file_list = sorted(glob.glob(os.path.join(data_dir, 'data*.zarr')))
    data = [xr.open_zarr(file_path, mask_and_scale=False) for file_path in file_list]

    assert batch_size % chunk_size == 0  # batch size must be divisible by chunk size!

    dataset = HLSInterpolDataset(data_stores=data, in_shape=input_shape, out_shape=output_size, bands=bands,
                         scaling=scaling, mean=mean, std=std,
                         data_augmentation=data_aug, chunk_size=chunk_size, max_images=max_images)

    if subset_indices:
        dataset = AttributePreservingSubset(dataset, subset_indices)

    sampler = DistributedSampler(dataset, rank=rank, num_replicas=world_size, shuffle=shuffle)

    collate_function = collate_fn_mask(batch_size=batch_size, collator=collator)

    dataloader = DataLoader(dataset,
                            batch_size=batch_size // chunk_size,
                            sampler=sampler,
                            num_workers=num_workers,
                            pin_memory=pin_memory,
                            prefetch_factor=prefetch_factor,
                            drop_last=True,
                            collate_fn=collate_function)
    return dataloader, sampler
