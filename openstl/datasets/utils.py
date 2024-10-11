import random
from functools import partial
from itertools import repeat
from typing import Callable
from timm.data.distributed_sampler import OrderedDistributedSampler, RepeatAugSampler

import torch.utils.data
import numpy as np
import math
import torch
from torch.utils.data.sampler import RandomSampler


def worker_init(worker_id, worker_seeding='all'):
    worker_info = torch.utils.data.get_worker_info()
    assert worker_info.id == worker_id
    if isinstance(worker_seeding, Callable):
        seed = worker_seeding(worker_info)
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed % (2 ** 32 - 1))
    else:
        assert worker_seeding in ('all', 'part')
        # random / torch seed already called in dataloader iter class w/ worker_info.seed
        # to reproduce some old results (same seed + hparam combo), partial seeding
        # is required (skip numpy re-seed)
        if worker_seeding == 'all':
            np.random.seed(worker_info.seed % (2 ** 32 - 1))


def fast_collate_for_prediction(batch):
    """ A fast collation function optimized for float32 images (np array or torch)
        and float32 targets (video prediction labels) in video prediction tasks"""
    assert isinstance(batch[0], tuple)
    batch_size = len(batch)
    if isinstance(batch[0][0], tuple):
        # This branch 'deinterleaves' and flattens tuples of input tensors into
        # one tensor ordered by position such that all tuple of position n will end up
        # in a torch.split(tensor, batch_size) in nth position
        inner_tuple_size = len(batch[0][0])
        flattened_batch_size = batch_size * inner_tuple_size
        targets = torch.zeros(flattened_batch_size, dtype=torch.float32)
        tensor = torch.zeros((flattened_batch_size, *batch[0][0][0].shape), dtype=torch.float32)
        for i in range(batch_size):
            # all input tensor tuples must be same length
            assert len(batch[i][0]) == inner_tuple_size
            for j in range(inner_tuple_size):
                targets[i + j * batch_size] = batch[i][1]
                tensor[i + j * batch_size] += torch.from_numpy(batch[i][0][j])
        return tensor, targets
    elif isinstance(batch[0][0], np.ndarray):
        targets = torch.tensor([b[1] for b in batch], dtype=torch.float32)
        assert len(targets) == batch_size
        tensor = torch.zeros((batch_size, *batch[0][0].shape), dtype=torch.float32)
        for i in range(batch_size):
            tensor[i] += torch.from_numpy(batch[i][0])
        return tensor, targets
    elif isinstance(batch[0][0], torch.Tensor):
        targets = torch.zeros((batch_size, *batch[1][0].shape), dtype=torch.float32)
        assert len(targets) == batch_size
        tensor = torch.zeros((batch_size, *batch[0][0].shape), dtype=torch.float32)
        for i in range(batch_size):
            tensor[i].copy_(batch[i][0])
        return tensor, targets
    else:
        assert False


def expand_to_chs(x, n):
    if not isinstance(x, (tuple, list)):
        x = tuple(repeat(x, n))
    elif len(x) == 1:
        x = x * n
    else:
        assert len(x) == n, 'normalization stats must match image channels'
    return x


class PrefetchLoader:

    def __init__(self,
                 loader,
                 mean=None,
                 std=None,
                 channels=3,
                 fp16=False):

        self.fp16 = fp16
        self.loader = loader
        if mean is not None and std is not None:
            mean = expand_to_chs(mean, channels)
            std = expand_to_chs(std, channels)
            normalization_shape = (1, channels, 1, 1)

            self.mean = torch.tensor([x * 255 for x in mean]).cuda().view(normalization_shape)
            self.std = torch.tensor([x * 255 for x in std]).cuda().view(normalization_shape)        
            if fp16:
                self.mean = self.mean.half()
                self.std = self.std.half()
        else:
            self.mean, self.std = None, None

    def __iter__(self):
        stream = torch.cuda.Stream()
        first = True

        for next_input, next_target in self.loader:
            with torch.cuda.stream(stream):
                next_input = next_input.cuda(non_blocking=True)
                next_target = next_target.cuda(non_blocking=True)
                if self.fp16:
                    if self.mean is not None:
                        next_input = next_input.half().sub_(self.mean).div_(self.std)
                        next_target = next_target.half().sub_(self.mean).div_(self.std)
                    else:
                        next_input = next_input.half()
                        next_target = next_target.half()
                else:
                    if self.mean is not None:
                        next_input = next_input.float().sub_(self.mean).div_(self.std)
                        next_target = next_target.float().sub_(self.mean).div_(self.std)
                    else:
                        next_input = next_input.float()
                        next_target = next_target.float()

            if not first:
                yield input, target
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            input = next_input
            target = next_target

        yield input, target

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset


def create_loader(dataset,
                  batch_size,
                  shuffle=True,
                  sampler=None,
                  is_training=False,
                  mean=None,
                  std=None,
                  num_workers=1,
                  num_aug_repeats=0,
                  input_channels=1,
                  use_prefetcher=False,
                  distributed=False,
                  pin_memory=False,
                  drop_last=False,
                  fp16=False,
                  collate_fn=None,
                  persistent_workers=True,
                  worker_seeding='all'):
    if sampler is None:
        if distributed and not isinstance(dataset, torch.utils.data.IterableDataset):
            if is_training:
                if num_aug_repeats:
                    sampler = RepeatAugSampler(dataset, num_repeats=num_aug_repeats)
                else:
                    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            else:
                # This will add extra duplicate entries to result in equal num
                # of samples per-process, will slightly alter validation results
                sampler = OrderedDistributedSampler(dataset)
    # else:
    #     assert num_aug_repeats==0, "RepeatAugment is not supported in non-distributed or IterableDataset"

    if collate_fn is None:
        collate_fn = torch.utils.data.dataloader.default_collate
    loader_class = torch.utils.data.DataLoader

    loader_args = dict(
        batch_size=batch_size,
        shuffle=shuffle and (not isinstance(dataset, torch.utils.data.IterableDataset)) and sampler is None and is_training,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
        worker_init_fn=partial(worker_init, worker_seeding=worker_seeding),
        persistent_workers=persistent_workers
    )
    try:
        loader = loader_class(dataset, **loader_args)
    except TypeError:
        loader_args.pop('persistent_workers')  # only in Pytorch 1.7+
        loader = loader_class(dataset, **loader_args)

    if use_prefetcher:
        loader = PrefetchLoader(
            loader,
            mean=mean,
            std=std,
            channels=input_channels,
            fp16=fp16,
        )

    return loader


def reshape_patch(img_tensor, patch_size):
    assert 4 == img_tensor.ndim
    seq_length = np.shape(img_tensor)[0]
    img_height = np.shape(img_tensor)[1]
    img_width = np.shape(img_tensor)[2]
    num_channels = np.shape(img_tensor)[3]
    a = np.reshape(img_tensor, [seq_length,
                                img_height // patch_size, patch_size,
                                img_width // patch_size, patch_size,
                                num_channels])
    b = np.transpose(a, [0, 1, 3, 2, 4, 5])
    patch_tensor = np.reshape(b, [seq_length,
                                  img_height // patch_size,
                                  img_width // patch_size,
                                  patch_size * patch_size * num_channels])
    return patch_tensor


def reshape_patch_back(patch_tensor, patch_size):
    # B L H W C
    assert 5 == patch_tensor.ndim
    batch_size = np.shape(patch_tensor)[0]
    seq_length = np.shape(patch_tensor)[1]
    patch_height = np.shape(patch_tensor)[2]
    patch_width = np.shape(patch_tensor)[3]
    channels = np.shape(patch_tensor)[4]
    img_channels = channels // (patch_size * patch_size)
    a = np.reshape(patch_tensor, [batch_size, seq_length,
                                  patch_height, patch_width,
                                  patch_size, patch_size,
                                  img_channels])
    b = np.transpose(a, [0, 1, 2, 4, 3, 5, 6])
    img_tensor = np.reshape(b, [batch_size, seq_length,
                                patch_height * patch_size,
                                patch_width * patch_size,
                                img_channels])
    return img_tensor


def reshape_patch_back_tensor(patch_tensor, patch_size):
    # B L H W C
    assert 5 == patch_tensor.ndim
    patch_narray = patch_tensor.detach().cpu().numpy()
    batch_size = np.shape(patch_narray)[0]
    seq_length = np.shape(patch_narray)[1]
    patch_height = np.shape(patch_narray)[2]
    patch_width = np.shape(patch_narray)[3]
    channels = np.shape(patch_narray)[4]
    img_channels = channels // (patch_size * patch_size)
    a = torch.reshape(patch_tensor, [batch_size, seq_length,
                                     patch_height, patch_width,
                                     patch_size, patch_size,
                                     img_channels])
    b = a.permute([0, 1, 2, 4, 3, 5, 6])
    img_tensor = torch.reshape(b, [batch_size, seq_length,
                                   patch_height * patch_size,
                                   patch_width * patch_size,
                                   img_channels])
    return img_tensor.permute(0, 1, 4, 2, 3)


def random_split_dataset(dataset, split_ratio, seed=42):
    from torch.utils.data import random_split
    total_ratio = sum(split_ratio)
    split_ratio = [x / total_ratio for x in split_ratio]
    dataset_len = len(dataset)
    dataset_split_len = [int(x * dataset_len) for x in split_ratio]
    dataset_split_len[-1] = int(dataset_len - sum(dataset_split_len) + dataset_split_len[-1])
    return random_split(dataset=dataset, lengths=dataset_split_len, generator=torch.Generator().manual_seed(seed))


class BatchSchedulerSampler(torch.utils.data.sampler.Sampler):
    """
    iterate over tasks and provide a random batch per task in each mini-batch
    """
    def __init__(self, dataset, batch_size, rank, gpus, shuffle=False, epoch=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.number_of_datasets = len(dataset.datasets)
        self.largest_dataset_size = max([len(cur_dataset) for cur_dataset in dataset.datasets])

        self.number_selected_samples = int(self.batch_size * math.ceil(self.largest_dataset_size / self.batch_size) * len(self.dataset.datasets) / gpus)
        self.number_of_total_size = self.number_selected_samples*gpus
        print('number_selected_samples', self.number_selected_samples)
        print('total sample epoch', self.number_of_datasets * self.largest_dataset_size)
        self.gpus = gpus
        self.rank = rank
        self.shuffle = shuffle
        self.epoch = epoch

    def __len__(self):
        return self.number_selected_samples

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            # indices = torch.randperm(len(self.dataset), generator=g)
        else:
            g = torch.Generator()
            g.manual_seed(0)

        samplers_list = []
        sampler_iterators = []
        for dataset_idx in range(self.number_of_datasets):
            cur_dataset = self.dataset.datasets[dataset_idx]
            sampler = RandomSampler(cur_dataset, generator=g)
            samplers_list.append(sampler)
            cur_sampler_iterator = iter(list(sampler.__iter__())[self.rank:self.number_selected_samples:self.gpus])
            sampler_iterators.append(cur_sampler_iterator)

        push_index_val = [0] + self.dataset.cumulative_sizes[:-1]
        step = self.batch_size * self.number_of_datasets
        samples_to_grab = self.batch_size
        
        # for this case we want to get all samples in dataset, this force us to resample from the smaller datasets
        epoch_samples = self.largest_dataset_size * self.number_of_datasets

        final_samples_list = []  # this is a list of indexes from the combined dataset
        for _ in range(0, epoch_samples, step):
            for i in range(self.number_of_datasets):
                cur_batch_sampler = sampler_iterators[i]
                cur_samples = []
                for _ in range(samples_to_grab):
                    try:
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                    except StopIteration:
                        # got to the end of iterator - restart the iterator and continue to get samples
                        # until reaching "epoch_samples"
                        sampler_iterators[i] = iter(list(samplers_list[i].__iter__())[self.rank:self.number_of_total_size:self.gpus])
                        cur_batch_sampler = sampler_iterators[i]
                        cur_sample_org = cur_batch_sampler.__next__()
                        cur_sample = cur_sample_org + push_index_val[i]
                        cur_samples.append(cur_sample)
                final_samples_list.extend(cur_samples)
        
        return iter(final_samples_list[:self.number_selected_samples])
    
    def set_epoch(self, epoch):
        self.epoch = epoch
    