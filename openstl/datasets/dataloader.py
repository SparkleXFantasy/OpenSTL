# Copyright (c) CAIRI AI Lab. All rights reserved
from torch.utils.data import ConcatDataset
from .utils import create_loader
from .dataset_multi import ConCatDatasetWithIndex
from .utils import BatchSchedulerSampler

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
        data_split_pool = ['5_625', '2_8125', '1_40625']
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
    
    
def load_concat_data(datanames, batch_size, val_batch_size, num_workers, data_root, dist=False, **kwargs):
    cfg_dataloader = dict(
        pre_seq_length=kwargs.get('pre_seq_length', 10),
        aft_seq_length=kwargs.get('aft_seq_length', 10),
        in_shape=kwargs.get('in_shape', None),
        distributed=dist,
        use_augment=kwargs.get('use_augment', False),
        use_prefetcher=kwargs.get('use_prefetcher', False),
        drop_last=kwargs.get('drop_last', False),
    )
    concat_datasets_train = []
    concat_datasets_val = []
    concat_datasets_test = []
    print(f'data cfg: {cfg_dataloader}')
    for dataname in datanames:
        if dataname == 'bair':
            from .dataloader_bair import load_dataset
            dataset_train, dataset_val, dataset_test = load_dataset(batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader)
            concat_datasets_train.append(dataset_train)
            concat_datasets_val.append(dataset_val)
            concat_datasets_test.append(dataset_test)
        elif dataname == 'human':
            from .dataloader_human import load_dataset
            dataset_train, dataset_val, dataset_test = load_dataset(batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader)
            concat_datasets_train.append(dataset_train)
            concat_datasets_val.append(dataset_val)
            concat_datasets_test.append(dataset_test)
        # elif dataname == 'kitticaltech':
        #     from .dataloader_kitticaltech import load_data
        #     return load_data(batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader)
        # elif 'kth' in dataname:  # 'kth', 'kth20', 'kth40'
        #     from .dataloader_kth import load_data
        #     return load_data(batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader)
        # elif dataname in ['mmnist', 'mfmnist', 'mmnist_cifar']:  # 'mmnist', 'mfmnist', 'mmnist_cifar'
        #     from .dataloader_moving_mnist import load_data
        #     cfg_dataloader['data_name'] = kwargs.get('data_name', 'mnist')
        #     return load_data(batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader)
        # elif 'noisymmnist' in dataname:  # 'mmnist - perceptual', 'mmnist - missing', 'mmnist - dynamic' 
        #     from .dataloader_noisy_moving_mnist import load_data
        #     cfg_dataloader['noise_type'] = kwargs.get('noise_type', 'perceptual')
        #     return load_data(batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader)
        # elif 'kinetics' in dataname:  # 'kinetics400', 'kinetics600'
        #     from .dataloader_kinetics import load_data
        #     cfg_dataloader['data_name'] = kwargs.get('data_name', 'kinetics400')
        #     return load_data(batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader)
        elif dataname == 'taxibj':
            from .dataloader_taxibj import load_dataset
            dataset_train, dataset_val, dataset_test = load_dataset(batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader)
            concat_datasets_train.append(dataset_train)
            concat_datasets_val.append(dataset_val)
            concat_datasets_test.append(dataset_test)
        # elif 'weather' in dataname:  # 'weather', 'weather_t2m', etc.
        #     from .dataloader_weather import load_data
        #     data_split_pool = ['5_625', '2_8125', '1_40625']
        #     data_split = '5_625'
        #     for k in data_split_pool:
        #         if dataname.find(k) != -1:
        #             data_split = k
        #     return load_data(batch_size, val_batch_size, data_root, num_workers,
        #                      distributed=dist, data_split=data_split, **kwargs)
        # elif 'sevir' in dataname:  #'sevir_vis', 'sevir_ir069', 'sevir_ir107', 'sevir_vil'
        #     from .dataloader_sevir import load_data
        #     cfg_dataloader['data_name'] = kwargs.get('data_name', 'sevir')
        #     return load_data(batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader)
        else:
            raise ValueError(f'Dataname {dataname} is unsupported')
    
    dataloader_train = create_loader(ConcatDataset(concat_datasets_train),
                                     batch_size=batch_size,
                                     shuffle=True, 
                                     is_training=True,
                                     pin_memory=True, drop_last=True,
                                     num_workers=num_workers,
                                     distributed=False, use_prefetcher=False)
    dataloader_vali = create_loader(ConcatDataset(concat_datasets_val),
                                    batch_size=val_batch_size,
                                    shuffle=False, is_training=False,
                                    pin_memory=True, drop_last=False,
                                    num_workers=num_workers,
                                    distributed=False, use_prefetcher=False)
    dataloader_test = create_loader(ConcatDataset(concat_datasets_test),
                                    batch_size=val_batch_size,
                                    shuffle=False, is_training=False,
                                    pin_memory=True, drop_last=False,
                                    num_workers=num_workers,
                                    distributed=False, use_prefetcher=False)
    return dataloader_train, dataloader_vali, dataloader_test


def load_concat_data_with_index(datanames, configs, batch_size, val_batch_size, num_workers, data_root, dist=False, **kwargs):


    concat_datasets_train = []
    concat_datasets_val = []
    concat_datasets_test = []
    for idx, dataname in enumerate(datanames):
        cfg_dataloader = dict(
            pre_seq_length=configs[idx].get('pre_seq_length', 10),
            aft_seq_length=configs[idx].get('aft_seq_length', 10),
            in_shape=configs[idx].get('in_shape', None),
            distributed=dist,
            use_augment=configs[idx].get('use_augment', False),
            use_prefetcher=configs[idx].get('use_prefetcher', False),
            drop_last=configs[idx].get('drop_last', False),
        )
        print(f'{dataname} cfg: {cfg_dataloader}')
        if dataname == 'bair':
            from .dataloader_bair import load_dataset
            dataset_train, dataset_val, dataset_test = load_dataset(batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader)
            concat_datasets_train.append(dataset_train)
            concat_datasets_val.append(dataset_val)
            concat_datasets_test.append(dataset_test)
        elif dataname == 'human':
            from .dataloader_human import load_dataset
            dataset_train, dataset_val, dataset_test = load_dataset(batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader)
            concat_datasets_train.append(dataset_train)
            concat_datasets_val.append(dataset_val)
            concat_datasets_test.append(dataset_test)
        # elif dataname == 'kitticaltech':
        #     from .dataloader_kitticaltech import load_data
        #     return load_data(batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader)
        # elif 'kth' in dataname:  # 'kth', 'kth20', 'kth40'
        #     from .dataloader_kth import load_data
        #     return load_data(batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader)
        # elif dataname in ['mmnist', 'mfmnist', 'mmnist_cifar']:  # 'mmnist', 'mfmnist', 'mmnist_cifar'
        #     from .dataloader_moving_mnist import load_data
        #     cfg_dataloader['data_name'] = kwargs.get('data_name', 'mnist')
        #     return load_data(batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader)
        # elif 'noisymmnist' in dataname:  # 'mmnist - perceptual', 'mmnist - missing', 'mmnist - dynamic' 
        #     from .dataloader_noisy_moving_mnist import load_data
        #     cfg_dataloader['noise_type'] = kwargs.get('noise_type', 'perceptual')
        #     return load_data(batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader)
        # elif 'kinetics' in dataname:  # 'kinetics400', 'kinetics600'
        #     from .dataloader_kinetics import load_data
        #     cfg_dataloader['data_name'] = kwargs.get('data_name', 'kinetics400')
        #     return load_data(batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader)
        elif dataname == 'taxibj':
            from .dataloader_taxibj import load_dataset
            dataset_train, dataset_val, dataset_test = load_dataset(batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader)
            concat_datasets_train.append(dataset_train)
            concat_datasets_val.append(dataset_val)
            concat_datasets_test.append(dataset_test)
        # elif 'weather' in dataname:  # 'weather', 'weather_t2m', etc.
        #     from .dataloader_weather import load_data
        #     data_split_pool = ['5_625', '2_8125', '1_40625']
        #     data_split = '5_625'
        #     for k in data_split_pool:
        #         if dataname.find(k) != -1:
        #             data_split = k
        #     return load_data(batch_size, val_batch_size, data_root, num_workers,
        #                      distributed=dist, data_split=data_split, **kwargs)
        # elif 'sevir' in dataname:  #'sevir_vis', 'sevir_ir069', 'sevir_ir107', 'sevir_vil'
        #     from .dataloader_sevir import load_data
        #     cfg_dataloader['data_name'] = kwargs.get('data_name', 'sevir')
        #     return load_data(batch_size, val_batch_size, data_root, num_workers, **cfg_dataloader)
        else:
            raise ValueError(f'Dataname {dataname} is unsupported')
    epoch = kwargs.get('epoch', 0)
    dataset_train = ConCatDatasetWithIndex(concat_datasets_train)
    dataloader_train = create_loader(dataset_train,
                                     batch_size=batch_size,
                                     shuffle=False, 
                                     sampler=BatchSchedulerSampler(dataset=dataset_train, batch_size=batch_size, rank=0, gpus=1, shuffle=True, epoch=epoch),
                                     is_training=True,
                                     pin_memory=True, drop_last=True,
                                     num_workers=num_workers,
                                     distributed=False, use_prefetcher=False)
    dataset_val = ConCatDatasetWithIndex(concat_datasets_val)
    dataloader_vali = create_loader(dataset_val,
                                    batch_size=val_batch_size,
                                    shuffle=False, 
                                    sampler=BatchSchedulerSampler(dataset=dataset_val, batch_size=val_batch_size, rank=0, gpus=1, shuffle=False, epoch=epoch),
                                    is_training=False,
                                    pin_memory=True, drop_last=False,
                                    num_workers=num_workers,
                                    distributed=False, use_prefetcher=False)
    dataset_test = ConCatDatasetWithIndex(concat_datasets_test)
    dataloader_test = create_loader(dataset_test,
                                    batch_size=val_batch_size,
                                    shuffle=False, 
                                    sampler=BatchSchedulerSampler(dataset=dataset_val, batch_size=val_batch_size, rank=0, gpus=1, shuffle=False, epoch=epoch),
                                    is_training=False,
                                    pin_memory=True, drop_last=False,
                                    num_workers=num_workers,
                                    distributed=False, use_prefetcher=False)
    return dataloader_train, dataloader_vali, dataloader_test