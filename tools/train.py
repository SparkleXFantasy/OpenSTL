# Copyright (c) CAIRI AI Lab. All rights reserved
from configparser import ConfigParser
import debugpy
import os.path as osp
import warnings
warnings.filterwarnings('ignore')
import copy
from openstl.api import BaseExperiment
from openstl.utils import (create_parser, default_parser, get_dist_info, load_config,
                           update_config)
import torch

# debugpy.listen(("0.0.0.0", 5678))
# print("Waiting for debugger to attach...")
# debugpy.wait_for_client()


if __name__ == '__main__':




    torch.set_float32_matmul_precision('medium')

    args = create_parser().parse_args()
    config = args.__dict__

    configs_paths = args.configs 
    all_configs = []
    for cfg_path in configs_paths:
        loaded_cfg = load_config(cfg_path)
        print(f'加载的 {cfg_path} 配置内容为 {loaded_cfg}')
        all_configs.append(loaded_cfg)

   
    configs = []

    
    for idx, loaded_cfg in enumerate(all_configs):
        
        temp_config = copy.deepcopy(config)
        

        
        for key, value in loaded_cfg.items():
            temp_config[key] = value  

       
        default_values = default_parser()
        for attribute in default_values.keys():
            if temp_config.get(attribute) is None:
                temp_config[attribute] = default_values[attribute]

        
        configs.append(copy.deepcopy(temp_config))

   
    args.configs = configs
    exp = BaseExperiment(args, config=config)
    rank, _ = get_dist_info()
    if args.ckpt_path:
        # 加载 checkpoint 并进行测试
        print('>' * 35 + ' testing with checkpoint ' + '<' * 35)
        ckpt = torch.load(args.ckpt_path)
        exp.method.load_state_dict(ckpt['state_dict'], strict=True)
        mse = exp.test()
    else:
        # 否则，正常训练并测试
        exp.train()
        if rank == 0:
            print('>' * 35 + ' testing  ' + '<' * 35)
            mse = exp.test()





    # cfg_path = osp.join('./configs_multi', args.dataname, f'{args.method}.py') \
    #     if args.config_file is None else args.config_file
    # if args.overwrite:
    #     config = update_config(config, load_config(cfg_path),
    #                            exclude_keys=['method'])
    # else:
    #     loaded_cfg = load_config(cfg_path)
    #     config = update_config(config, loaded_cfg,
    #                            exclude_keys=['method', 'val_batch_size',
    #                                          'drop_path', 'warmup_epoch'])
        
    #     default_values = default_parser()
    #     for attribute in default_values.keys():
    #         if config[attribute] is None:
    #             config[attribute] = default_values[attribute]

    # print('>'*35 + ' training ' + '<'*35)
    # exp = BaseExperiment(args)
    # rank, _ = get_dist_info()
    # exp.train()

    # if rank == 0:
    #     print('>'*35 + ' testing  ' + '<'*35)
    # mse = exp.test()