from typing import Iterable, Union
from torch.utils.data import ConcatDataset, Dataset
import bisect

import torch
from torch.utils.data import ConcatDataset

class ConCatDatasetWithIndex(torch.utils.data.ConcatDataset):
    def __init__(self, datasets):
        super().__init__(datasets)

    def __getitem__(self, idx):
        # 如果 idx 是列表，那么需要批量获取多个样本
        if isinstance(idx, list):
            return [self.__getitem__(i) for i in idx]

        # 如果 idx 是一个 tuple，表示已经包含了 dataset_idx 和 sample_idx
        if isinstance(idx, tuple):
            dataset_idx, sample_idx = idx
        else:
            # 如果不是 tuple，则使用原始的索引逻辑，将 idx 转换为 dataset_idx 和 sample_idx
            if idx < 0:
                if -idx > len(self):
                    raise ValueError("absolute value of index should not exceed dataset length")
                idx = len(self) + idx
            dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
            if dataset_idx == 0:
                sample_idx = idx
            else:
                sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        # 获取样本数据
        data = self.datasets[dataset_idx][sample_idx]
        
        # 将样本数据的形状写入文本文件
        with open("tensor_shapes_log.txt", "a") as f:
            if isinstance(data, tuple) and len(data) == 2:
                tensor1, tensor2 = data
                f.write(f"Dataset Index: {dataset_idx}, Sample Index: {sample_idx}\n")
                f.write(f"Tensor 1 Shape: {tensor1.shape}\n")
                f.write(f"Tensor 2 Shape: {tensor2.shape}\n")
            else:
                f.write(f"Dataset Index: {dataset_idx}, Sample Index: {sample_idx} - Data is not in expected format.\n")

        # 返回 dataset_idx 和样本数据
        return dataset_idx, data





