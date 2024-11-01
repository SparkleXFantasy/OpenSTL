import numpy as np
import torch.nn as nn
import os.path as osp
import lightning as l
from openstl.utils import print_log, check_dir
from openstl.core import get_optim_scheduler, timm_schedulers
from openstl.core import metric
import torch
from torch.cuda.amp import autocast

class Base_method(l.LightningModule):

    def __init__(self, **args):
        super().__init__()

        if 'weather' in args['dataname']:
            self.metric_list, self.spatial_norm = args['metrics'], True
            self.channel_names = args.data_name if 'mv' in args['data_name'] else None
        else:
            self.metric_list, self.spatial_norm, self.channel_names = args['metrics'], False, None

        self.save_hyperparameters()
        self.model = self._build_model(**args)
        self.criterion = nn.MSELoss()
        self.test_outputs = {}

    def _build_model(self):
        raise NotImplementedError
    
    def configure_optimizers(self):
        optimizer, scheduler, by_epoch = get_optim_scheduler(
            self.hparams, 
            self.hparams.epoch, 
            self.model, 
            self.hparams.steps_per_epoch
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler, 
                "interval": "epoch" if by_epoch else "step"
            },
        }
    
    def lr_scheduler_step(self, scheduler, metric):
        if any(isinstance(scheduler, sch) for sch in timm_schedulers):
            scheduler.step(epoch=self.current_epoch)
        else:
            if metric is None:
                scheduler.step()
            else:
                scheduler.step(metric)

    def forward(self, batch):
        NotImplementedError
    
    def training_step(self, batch, batch_idx):
        NotImplementedError

    def validation_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        pred_y = self(batch_x, batch_y)
        loss = self.criterion(pred_y, batch_y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=False)
        return loss
    
    def test_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        pred_y = self(batch_x, batch_y)
        outputs = {'inputs': batch_x.cpu().numpy(), 'preds': pred_y.cpu().numpy(), 'trues': batch_y.cpu().numpy()}
        self.test_outputs.append(outputs)
        return outputs

    def on_test_epoch_end(self):
        results_all = {}
        for k in self.test_outputs[0].keys():
            results_all[k] = np.concatenate([batch[k] for batch in self.test_outputs], axis=0)
        
        eval_res, eval_log = metric(results_all['preds'], results_all['trues'],
            self.hparams.test_mean, self.hparams.test_std, metrics=self.metric_list, 
            channel_names=self.channel_names, spatial_norm=self.spatial_norm,
            threshold=self.hparams.get('metric_threshold', None))
        
        results_all['metrics'] = np.array([eval_res['mae'], eval_res['mse']])

        if self.trainer.is_global_zero:
            print_log(eval_log)
            folder_path = check_dir(osp.join(self.hparams.save_dir, 'saved'))

            for np_data in ['metrics', 'inputs', 'trues', 'preds']:
                np.save(osp.join(folder_path, np_data + '.npy'), results_all[np_data])
        return results_all
    
    
class Base_multi_method(l.LightningModule):

    def __init__(self, enc_dec_configs, **args):
        super().__init__()
       


        self.save_hyperparameters({**args, 'enc_dec_configs': enc_dec_configs})

        if 'weather' in args['dataname']:
            self.metric_list, self.spatial_norm = args['metrics'], True
            self.channel_names = args.data_name if 'mv' in args['data_name'] else None
        else:
            self.metric_list, self.spatial_norm, self.channel_names = args['metrics'], False, None

        self.test_outputs = {} 
        self.model = self._build_model(enc_dec_configs, **args)
        self.criterion = nn.MSELoss()
   

    def _build_model(self):
        raise NotImplementedError
    
    def configure_optimizers(self):
        optimizer, scheduler, by_epoch = get_optim_scheduler(
            self.hparams, 
            self.hparams.epoch, 
            self.model, 
            self.hparams.steps_per_epoch
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler, 
                "interval": "epoch" if by_epoch else "step"
            },
        }
    
    def lr_scheduler_step(self, scheduler, metric):
        if any(isinstance(scheduler, sch) for sch in timm_schedulers):
            scheduler.step(epoch=self.current_epoch)
        else:
            if metric is None:
                scheduler.step()
            else:
                scheduler.step(metric)

    def forward(self, batch):
        NotImplementedError



    def training_step(self, batch, batch_idx):
        
        dataset_idx, batch_data = batch
        batch_x, batch_y = batch_data

    
        with autocast():
          
            pred_y = self(batch_x, dataset_idx)

            if dataset_idx == 2:  
                target_length = 12
            else:
                target_length = 4
            
            loss = self.criterion(pred_y[:, :target_length], batch_y)

        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss




    def validation_step(self, batch, batch_idx):
        
        dataset_idx, batch_data = batch
        batch_x, batch_y = batch_data

        pred_y = self(batch_x, dataset_idx)

        if dataset_idx == 2:  
            loss = self.criterion(pred_y[:, :12], batch_y) 
        else:
            
            loss = self.criterion(pred_y[:, :4], batch_y)  

        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss




    def test_step(self, batch, batch_idx):
        # 获取 dataset_idx 和 batch 数据
        dataset_idx, batch_data = batch
        batch_x, batch_y = batch_data
        #print(f"Dataset index: {dataset_idx}")

        # 根据 dataset_idx 使用相应的 encoder 和 decoder
        pred_y = self(batch_x, dataset_idx, batch_y)

        # 将预测结果和实际值保存在 test_outputs 中，分开按数据集保存
        if dataset_idx not in self.test_outputs:
            #print(f"test_outputs{self.test_outputs}")
            self.test_outputs[dataset_idx] = []

        # 存储预测值、输入数据和真实值
        outputs = {'inputs': batch_x.cpu().numpy(), 
                'preds': pred_y.cpu().numpy(), 
                'trues': batch_y.cpu().numpy()}
        
        self.test_outputs[dataset_idx].append(outputs)
        return outputs



    def on_test_epoch_end(self):
        results_all = {0: {}, 1: {}, 2: {}}  # 初始化结果字典
        metrics_results = {0: {}, 1: {}, 2: {}}

        # 计算每个数据集的 MAE、MSE、PSNR 和 SSIM

        for dataset_idx, outputs_list in self.test_outputs.items():
            print(f"test_outputs的item有哪些{self.test_outputs.items}")
            results_all[dataset_idx] = {}
            for k in outputs_list[0].keys():
                results_all[dataset_idx][k] = np.concatenate([batch[k] for batch in outputs_list], axis=0)

            # 根据不同的数据集索引调整 preds 和 trues
            preds = results_all[dataset_idx]['preds']
            trues = results_all[dataset_idx]['trues']
            
            if dataset_idx == 2:
                # 如果是第三个数据集（索引为 2），裁剪预测的帧数以匹配真实值
                preds = preds[:, :trues.shape[1], :, :, :]

            # 使用 metric 函数计算 MAE、MSE、PSNR 和 SSIM
            eval_res, eval_log = metric(
                preds, 
                trues,
                self.hparams.test_mean, 
                self.hparams.test_std,
                metrics=['mae', 'mse', 'psnr', 'ssim'],  # 只计算这四个指标
                channel_names=self.channel_names, 
                spatial_norm=self.spatial_norm,
                threshold=self.hparams.get('metric_threshold', None)
            )

            # 保存 MAE、MSE、PSNR 和 SSIM
            mae = eval_res['mae']
            mse = eval_res['mse']
            psnr = eval_res['psnr']
            ssim = eval_res['ssim']

            metrics_results[dataset_idx] = {'mae': mae, 'mse': mse, 'psnr': psnr, 'ssim': ssim}

            if self.trainer.is_global_zero:
                # 打印每个数据集的日志
                print(f"Dataset {dataset_idx}: MAE={mae}, MSE={mse}, PSNR={psnr}, SSIM={ssim}")
                folder_path = check_dir(osp.join(self.hparams.save_dir, 'saved'))

                # 保存每个数据集的预测和真实值
                for np_data in ['inputs', 'trues', 'preds']:
                    np.save(osp.join(folder_path, f'{np_data}_dataset{dataset_idx}.npy'), results_all[dataset_idx][np_data])


        if self.trainer.is_global_zero:
            # 保存整体的度量结果
            np.save(osp.join(folder_path, 'metrics_results.npy'), metrics_results)

        return metrics_results




















    
    # def test_step(self, batch, batch_idx):
    #     dataset_idx, batch_data = batch
    #     batch_x, batch_y = batch_data
    #     pred_y = self(batch_x, dataset_idx, batch_y)
    #     outputs = {'inputs': batch_x.cpu().numpy(), 'preds': pred_y.cpu().numpy(), 'trues': batch_y.cpu().numpy()}
    #     self.test_outputs.append(outputs)
    #     return outputs


    # def on_test_epoch_end(self):
    #     results_all = {}
    #     for k in self.test_outputs[0].keys():
    #         results_all[k] = np.concatenate([batch[k] for batch in self.test_outputs], axis=0)
        
    #     eval_res, eval_log = metric(results_all['preds'], results_all['trues'],
    #         self.hparams.test_mean, self.hparams.test_std, metrics=self.metric_list, 
    #         channel_names=self.channel_names, spatial_norm=self.spatial_norm,
    #         threshold=self.hparams.get('metric_threshold', None))
        
    #     results_all['metrics'] = np.array([eval_res['mae'], eval_res['mse']])

    #     if self.trainer.is_global_zero:
    #         print_log(eval_log)
    #         folder_path = check_dir(osp.join(self.hparams.save_dir, 'saved'))

    #         for np_data in ['metrics', 'inputs', 'trues', 'preds']:
    #             np.save(osp.join(folder_path, np_data + '.npy'), results_all[np_data])
    #     return results_all



    # def on_test_epoch_end(self):
    #     results_all = {}
    #     for k in self.test_outputs[0].keys():
    #         results_all[k] = np.concatenate([batch[k] for batch in self.test_outputs], axis=0)
        
    #     eval_res, eval_log = metric(results_all['preds'], results_all['trues'],
    #         self.hparams.test_mean, self.hparams.test_std, metrics=self.metric_list, 
    #         channel_names=self.channel_names, spatial_norm=self.spatial_norm,
    #         threshold=self.hparams.get('metric_threshold', None))
        
    #     results_all['metrics'] = np.array([eval_res['mae'], eval_res['mse']])

    #     if self.trainer.is_global_zero:
    #         print_log(eval_log)
    #         folder_path = check_dir(osp.join(self.hparams.save_dir, 'saved'))

    #         for np_data in ['metrics', 'inputs', 'trues', 'preds']:
    #             np.save(osp.join(folder_path, np_data + '.npy'), results_all[np_data])
    #     return results_all