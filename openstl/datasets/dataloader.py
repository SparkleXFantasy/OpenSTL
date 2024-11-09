# Copyright (c) CAIRI AI Lab. All rights reserved
from torch.utils.data import ConcatDataset
from .utils import create_loader
from .dataset_multi import ConCatDatasetWithIndex
from .utils import ImprovedBatchSchedulerSampler
import torch
from torch.utils.data import DataLoader, default_collate
import numpy as np

batch_counter = 0


def load_data(dataname, batch_size, val_batch_size, num_workers, data_root, dist=False, **kwargs):
    cfg_dataloader = dict(
        pre_seq_length=kwargs.get('pre_seq_length', 10),
        aft_seq_length=kwargs.get('aft_seq_length', 10),
        in_shape=kwargs.get('in_shape', None),
        distributed=dist,
        use_augment=kwargs.get('use_augment', False),
        use_prefetcher=kwargs.get('use_prefetcher', False),
        drop_last=kwargs.get('drop_last', False),
    )

    if dataname == 'bair':
        from .dataloader_bair import load_data
        return load_data(batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader)
    elif dataname == 'city':
        from .dataloader_city import load_data
        return load_data(batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader)
    elif dataname == 'human':
        from .dataloader_human import load_data
        return load_data(batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader)
    elif dataname == 'kitticaltech':
        from .dataloader_kitticaltech import load_data
        return load_data(batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader)
    elif 'kth' in dataname:  # 'kth', 'kth20', 'kth40'
        from .dataloader_kth import load_data
        return load_data(batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader)
    elif dataname in ['mmnist', 'mfmnist', 'mmnist_cifar']:  # 'mmnist', 'mfmnist', 'mmnist_cifar'
        from .dataloader_moving_mnist import load_data
        cfg_dataloader['data_name'] = kwargs.get('data_name', 'mnist')
        return load_data(batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader)
    elif 'noisymmnist' in dataname:  # 'mmnist - perceptual', 'mmnist - missing', 'mmnist - dynamic' 
        from .dataloader_noisy_moving_mnist import load_data
        cfg_dataloader['noise_type'] = kwargs.get('noise_type', 'perceptual')
        return load_data(batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader)
    elif 'kinetics' in dataname:  # 'kinetics400', 'kinetics600'
        from .dataloader_kinetics import load_data
        cfg_dataloader['data_name'] = kwargs.get('data_name', 'kinetics400')
        return load_data(batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader)
    elif dataname == 'taxibj':
        from .dataloader_taxibj import load_data
        return load_data(batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader)
    elif 'weather' in dataname:  # 'weather', 'weather_t2m', etc.
        from .dataloader_weather import load_data
        data_split_pool = ['5_625', '2_885', '1_40625']
        data_split = '5_625'
        for k in data_split_pool:
            if dataname.find(k) != -1:
                data_split = k
        return load_data(batch_size, val_batch_size, data_root, num_workers,
                         distributed=dist, data_split=data_split, **kwargs)
    elif 'sevir' in dataname:  #'sevir_vis', 'sevir_ir069', 'sevir_ir107', 'sevir_vil'
        from .dataloader_sevir import load_data
        cfg_dataloader['data_name'] = kwargs.get('data_name', 'sevir')
        return load_data(batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader)
    else:
        raise ValueError(f'Dataname {dataname} is unsupported')
    
    


def save_batches_to_file(dataloader, filename='batch_details.txt'):
    with open(filename, 'w') as f:
        for batch_idx, batch in enumerate(dataloader):
            f.write(f"Batch {batch_idx}:\n")
            for i, sample in enumerate(batch):
                if isinstance(sample, torch.Tensor):
                    f.write(f"  Sample {i} shape: {sample.shape}\n")
                else:
                    f.write(f"  Sample {i} type: {type(sample)}\n")
            f.write("\n")
            if batch_idx >= 1:  
                break


def custom_collate_fn(batch):

    global batch_counter

    # 打开文件用于保存输出信息
    with open(f'batch_structure_log.txt', 'a') as file:
        # 记录当前 batch 的索引
        file.write(f"Batch {batch_counter}:\n")

        batch_pre_list = []
        batch_aft_list = []
        dataset_idx = None

        # 处理 batch 中每个元素
        for idx, item in enumerate(batch):
            
            if isinstance(item, list) and len(item) == 4:
                for sub_idx, sub_item in enumerate(item):
                    
                    if not isinstance(sub_item, tuple) or len(sub_item) != 2:
                        file.write(f"Sub-element {sub_idx} in Sample {idx} is not valid. Type: {type(sub_item)}, Value: {sub_item}\n")
                    else:
                        dataset_idx, data = sub_item
                        if not isinstance(dataset_idx, int):
                            file.write(f"Sub-element {sub_idx} in Sample {idx} has invalid `dataset_idx` type: {type(dataset_idx)}, value: {dataset_idx}\n")
                        
                        if isinstance(data, tuple) and len(data) == 2:
                            pre, aft = data
                            if isinstance(pre, torch.Tensor) and isinstance(aft, torch.Tensor):
                                file.write(f"  Sample {idx}, Sub-sample {sub_idx} contains two tensors:\n")
                                file.write(f"    Pre tensor shape: {pre.shape}\n")
                                file.write(f"    Aft tensor shape: {aft.shape}\n")
                                
                                batch_pre_list.append(pre)
                                batch_aft_list.append(aft)
                            else:
                                file.write(f"  Sample {idx}, Sub-sample {sub_idx} contains non-tensor elements.\n")
                        else:
                            file.write(f"  Sample {idx}, Sub-sample {sub_idx} data is not a tuple of two tensors. Type: {type(data)}, Value: {data}\n")
            else:
                file.write(f"  Sample {idx} is not a valid list of 4 elements. Type: {type(item)}, Length: {len(item)}, Value: {item}\n")

        
        file.write("=== End of Batch Information ===\n\n")

        
        if batch_pre_list and batch_aft_list:
            combined_pre = torch.stack(batch_pre_list, dim=0)
            combined_aft = torch.stack(batch_aft_list, dim=0)

            # 记录合并后的 batch 数据
            file.write(f"  Combined Pre Batch data shape: {combined_pre.shape}\n")
            file.write(f"  Combined Aft Batch data shape: {combined_aft.shape}\n\n")

           
            batch_data = (combined_pre, combined_aft)

    # 增加批次索引计数器
    batch_counter += 1

    # 返回 dataset_idx 和 batch_data
    return dataset_idx, batch_data



def load_concat_data_with_index(datanames, configs, batch_size, val_batch_size, num_workers, data_root, dist=False, **kwargs):
    concat_datasets_train = []
    concat_datasets_val = []
    concat_datasets_test = []
    

    for idx, dataname in enumerate(datanames):
        cfg_dataloader = dict(
            pre_seq_length=configs[idx].get('pre_seq_length', 4),
            aft_seq_length=configs[idx].get('aft_seq_length', 4),
            in_shape=configs[idx].get('in_shape', None),
            distributed=dist,
            use_augment=configs[idx].get('use_augment', False),
            use_prefetcher=configs[idx].get('use_prefetcher', False),
            drop_last=configs[idx].get('drop_last', False),
        )
        
        if dataname == 'bair':
            from .dataloader_bair import load_dataset
            dataset_train, dataset_val, dataset_test = load_dataset(batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader)
        elif dataname == 'human':
            from .dataloader_human import load_dataset
            dataset_train, dataset_val, dataset_test = load_dataset(batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader)
        elif dataname == 'taxibj':
            from .dataloader_taxibj import load_dataset
            dataset_train, dataset_val, dataset_test = load_dataset(batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader)
        elif dataname == 'city':
            from .dataloader_city import load_dataset
            dataset_train, dataset_val, dataset_test = load_dataset(batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader)
        elif dataname == 'sevir' : #'sevir_vis', 'sevir_ir069', 'sevir_ir107', 'sevir_vil'
            from .dataloader_sevir import load_dataset
            cfg_dataloader['data_name'] = kwargs.get('data_name', 'vil')
            dataset_train, dataset_val, dataset_test = load_dataset(batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader)
        else:
            raise ValueError(f'Dataname {dataname} is unsupported')

      
       
        sample = next(iter(dataset_train))

        if isinstance(sample, (tuple, list)):
            for i, element in enumerate(sample):
                if isinstance(element, torch.Tensor):
                    print(f"Element {i} is a tensor with shape: {element.shape}")
                    print(f"First few values in tensor element {i}: {element.flatten()[:5]}")
                elif isinstance(element, list):
                    print(f"Element {i} is a list with length: {len(element)}")
                    if len(element) > 0 and isinstance(element[0], torch.Tensor):
                        print(f"First tensor element shape in list element {i}: {element[0].shape}")
                elif isinstance(element, dict):
                    print(f"Element {i} is a dict with keys: {element.keys()}")
                    for key, value in element.items():
                        if isinstance(value, torch.Tensor):
                            print(f"Key '{key}' has tensor value with shape: {value.shape}")
                else:
                    print(f"Element {i} is of type: {type(element)}")
        else:
            print(f"Sample is of unrecognized type: {type(sample)}")

        concat_datasets_train.append(dataset_train)
        concat_datasets_val.append(dataset_val)
        concat_datasets_test.append(dataset_test)


        # Print dataset details
        #print(f"Dataset '{dataname}' loaded with Train: {len(dataset_train)}, Val: {len(dataset_val)}, Test: {len(dataset_test)} samples")

    # Creating ConCatDatasetWithIndex for combined training dataset
    dataset_train = ConCatDatasetWithIndex(concat_datasets_train)
    sampler_train = ImprovedBatchSchedulerSampler(
        dataset=dataset_train,
        batch_size=4,
        shuffle=True
    )
    sampler_train.set_epoch(0)

    # Creating DataLoader for training dataset
    dataloader_train = create_loader(
        dataset_train,
        batch_size=1,
        shuffle=False,
        sampler=sampler_train,
        is_training=True,
        pin_memory=True,
        drop_last=True,
        num_workers=0,
        distributed=False,
        use_prefetcher=False,
        persistent_workers=False,
        collate_fn=custom_collate_fn
        
    )
    
 

    with open("debug_output.txt", "w") as debug_file:

        for batch_idx, batch in enumerate(dataloader_train):
            debug_file.write(f"[DEBUG] Training Batch {batch_idx}: Batch size: {len(batch)}\n")
            
         
            debug_file.write(f"[DEBUG] Batch {batch_idx} data type: {type(batch)}\n")

          
            if isinstance(batch, (list, tuple)):
                for element_idx, element in enumerate(batch):
                    debug_file.write(f"[DEBUG] Element {element_idx} type: {type(element)}\n")

                    if isinstance(element, torch.Tensor):
                        debug_file.write(f"[DEBUG] Element {element_idx} is a tensor with shape: {element.shape}\n")
                        debug_file.write(f"[DEBUG] First few values in tensor element {element_idx}: {element.flatten()[:16]}\n")

                    elif isinstance(element, list):
                        debug_file.write(f"[DEBUG] Element {element_idx} is a list with length: {len(element)}\n")
                     
                        debug_file.write(f"[DEBUG] First 3 elements of list element {element_idx}: {element[:3]}\n")

                    elif isinstance(element, dict):
                        debug_file.write(f"[DEBUG] Element {element_idx} is a dict with keys: {element.keys()}\n")
                        
                        for key, value in list(element.items())[:3]:
                            debug_file.write(f"[DEBUG] Key: {key}, Value Type: {type(value)}\n")
                            if isinstance(value, torch.Tensor):
                                debug_file.write(f"[DEBUG] Value Tensor Shape: {value.shape}\n")

                    else:
                        debug_file.write(f"[DEBUG] Element {element_idx} is of unrecognized type: {type(element)}\n")

            
            elif isinstance(batch, torch.Tensor):
                debug_file.write(f"[DEBUG] Batch {batch_idx} is a tensor with shape: {batch.shape}\n")
                debug_file.write(f"[DEBUG] First few values in tensor: {batch.flatten()[:5]}\n")

      
            else:
                debug_file.write(f"[DEBUG] Batch {batch_idx} contains elements of unrecognized type: {type(batch)}\n")

            if batch_idx >= 2:  
                break



      

    # Similar DataLoader creation for validation and testing datasets
    dataset_val = ConCatDatasetWithIndex(concat_datasets_val)
    sampler_val = ImprovedBatchSchedulerSampler(
        dataset=dataset_val,
        batch_size=4,
        shuffle=False
    )
    sampler_val.set_epoch(0)

    dataloader_vali = create_loader(
        dataset_val,
        batch_size=1,
        shuffle=False,
        sampler=sampler_val,
        is_training=False,
        pin_memory=True,
        drop_last=False,
        num_workers=0,
        distributed=False,
        use_prefetcher=False,
        persistent_workers=False,
        collate_fn=custom_collate_fn
    )

    dataset_test = ConCatDatasetWithIndex(concat_datasets_test)
    sampler_test = ImprovedBatchSchedulerSampler(
        dataset=dataset_test,
        batch_size=4,
        shuffle=False
    )
    sampler_test.set_epoch(0)

    dataloader_test = create_loader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        sampler=sampler_test,
        is_training=False,
        pin_memory=True,
        drop_last=False,
        num_workers=0,
        distributed=False,
        use_prefetcher=False,
        persistent_workers=False,
        collate_fn=custom_collate_fn
    )

    return dataloader_train, dataloader_vali, dataloader_test

