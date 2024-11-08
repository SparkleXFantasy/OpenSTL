import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py
from openstl.datasets.utils import create_loader
import torch.nn.functional as F
from PIL import Image

try:
    import tensorflow as tf
except ImportError:
    tf = None

from openstl.datasets.utils import create_loader
from .pyxis import Reader

class CityscapesDataset(Dataset):
    """Taxibj <https://arxiv.org/abs/1610.00081>`_ Dataset"""

    def __init__(self, data_path, use_augment=False, input_length=2, output_length=5, data_name='city'):
        super(CityscapesDataset, self).__init__()
        self.use_augment = use_augment
        self.mean = 0
        self.std = 1
        self.input_length = input_length
        self.output_length = output_length
        self.data = Reader(data_path, lock=False)
        self.num_trunks = 3
        self.data_name = data_name
        
        clip_len = self.input_length+self.output_length

        frame_interval = 1
        total_frames = clip_len * frame_interval
        data_list = []

        stride = 5


    def _augment_seq(self, imgs, crop_scale=0.95):
        """Augmentations for video"""
        _, _, h, w = imgs.shape  # original shape, e.g., [10, 3, 64, 64]
        imgs = F.interpolate(imgs, scale_factor=1 / crop_scale, mode='bilinear')
        _, _, ih, iw = imgs.shape
        # Random Crop
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        imgs = imgs[:, :, x:x+h, y:y+w]
        # Random Flip
        if random.randint(0, 1):
            imgs = torch.flip(imgs, dims=(3, ))  # horizontal flip
        return imgs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # print(torch.tensor(self.data[index, ::]).float().shape)
        data = self.data[index]
        inputs, labels = torch.tensor(data['input']), torch.tensor(data['target'])

        if self.use_augment:
            # len_data = data.shape[0]  # 4
            seqs = self._augment_seq(torch.cat([inputs, labels], dim=0),crop_scale=0.95)
            inputs = seqs[:self.input_length,...]
            labels = seqs[self.input_length:self.input_length+self.output_length,...]
        return inputs, labels

def load_data(batch_size, val_batch_size, data_root, num_workers=4,
              pre_seq_length=None, aft_seq_length=None, in_shape=None,
              distributed=False, use_augment=False, use_prefetcher=False, drop_last=False):

    # train_data = os.path.join(data_root, 'citescape/data_city_train')
    # test_data = os.path.join(data_root, 'citescape/data_city_test')
    train_data = '/home/bingxing2/ailab/group/ai4multi/data_full/citescape/data_city_train'
    test_data = '/home/bingxing2/ailab/group/ai4multi/data_full/citescape/data_city_test'


    train_set = CityscapesDataset(data_path=train_data, use_augment=True)
    test_set = CityscapesDataset(data_path=test_data, use_augment=False)

    dataloader_train = create_loader(train_set,
                                     batch_size=batch_size,
                                     shuffle=True, is_training=True,
                                     pin_memory=True, drop_last=True,
                                     num_workers=num_workers,
                                     distributed=distributed, use_prefetcher=use_prefetcher)
    dataloader_vali = create_loader(test_set,
                                    batch_size=val_batch_size,
                                    shuffle=False, is_training=False,
                                    pin_memory=True, drop_last=drop_last,
                                    num_workers=num_workers,
                                    distributed=distributed, use_prefetcher=use_prefetcher)
    dataloader_test = create_loader(test_set,
                                    batch_size=val_batch_size,
                                    shuffle=False, is_training=False,
                                    pin_memory=True, drop_last=drop_last,
                                    num_workers=num_workers,
                                    distributed=distributed, use_prefetcher=use_prefetcher)

    return dataloader_train, dataloader_vali, dataloader_test


def load_dataset(batch_size, val_batch_size, data_root, num_workers=4,
              pre_seq_length=None, aft_seq_length=None, in_shape=None,
              distributed=False, use_augment=False, use_prefetcher=False, drop_last=False):


    train_data = '/home/bingxing2/ailab/group/ai4multi/data_full/citescape/data_city_train'
    test_data = '/home/bingxing2/ailab/group/ai4multi/data_full/citescape/data_city_test'
    train_set = CityscapesDataset(data_path=train_data, use_augment=True)
    test_set = CityscapesDataset(data_path=test_data, use_augment=False)

 
    return train_set, test_set, test_set


if __name__ == '__main__':
    dataloader_train, _, dataloader_test = \
        load_data(batch_size=16,
                  val_batch_size=4,
                  data_root='/home/bingxing2/ailab/group/ai4multi/data_full/citescape',
                  num_workers=4,
                  pre_seq_length=4, aft_seq_length=4)

    print(len(dataloader_train), len(dataloader_test))
    for item in dataloader_train:
        print(item[0].shape, item[1].shape)
        break
    for item in dataloader_test:
        print(item[0].shape, item[1].shape)
        break
