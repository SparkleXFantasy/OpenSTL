import numpy as np
import torch.nn as nn
import os.path as osp
import lightning as l
from openstl.utils import print_log, check_dir
from openstl.core import get_optim_scheduler, timm_schedulers
from openstl.core import metric
import torch


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
        self.test_outputs = []

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
       


        # 只保存公共参数和 enc_dec_configs
        self.save_hyperparameters({**args, 'enc_dec_configs': enc_dec_configs})
        

        if 'weather' in args['dataname']:
            self.metric_list, self.spatial_norm = args['metrics'], True
            self.channel_names = args.data_name if 'mv' in args['data_name'] else None
        else:
            self.metric_list, self.spatial_norm, self.channel_names = args['metrics'], False, None

        
        self.model = self._build_model(enc_dec_configs, **args)
        self.criterion = nn.MSELoss()
        self.test_outputs = []

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
        # 解包 batch
        dataset_idx, batch_data = batch
        batch_x, batch_y = batch_data

        # 模型前向传播，包含 dataset_idx 以便于根据数据集调整行为
        pred_y = self(batch_x, dataset_idx)

        # 根据不同的 dataset_idx 处理目标和预测的长度
        if dataset_idx == 2:  # 如果是第三个数据集
            # 假设第三个数据集的 batch_y 长度为 12
            loss = self.criterion(pred_y[:, :12], batch_y)  # 使用 12 帧的预测和目标进行损失计算
        else:
            # 如果是第一个或第二个数据集，目标长度为 4
            loss = self.criterion(pred_y[:, :4], batch_y)  # 使用前 4 帧的预测和目标进行损失计算

        # 记录损失值
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss



    def validation_step(self, batch, batch_idx):
        # 解包 batch
        dataset_idx, batch_data = batch
        batch_x, batch_y = batch_data

        # 前向传播，包含 dataset_idx 以便于根据数据集调整行为
        pred_y = self(batch_x, dataset_idx)

        # 根据不同的 dataset_idx 处理目标和预测的长度
        if dataset_idx == 2:  # 如果是第三个数据集
            # 假设第三个数据集的目标长度为 12
            loss = self.criterion(pred_y[:, :12], batch_y)  # 使用 12 帧的预测和目标进行损失计算
        else:
            # 如果是第一个或第二个数据集，目标长度为 4
            loss = self.criterion(pred_y[:, :4], batch_y)  # 使用前 4 帧的预测和目标进行损失计算

        # 记录验证损失
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss




    # def validation_step(self, batch, batch_idx):
    #     print(f"Batch type: {type(batch)}, Batch length: {len(batch)}")

    #     dataset_idx, batch_data = batch
    #     batch_x, batch_y = batch_data
    #     pred_y = self(batch_x, dataset_idx, batch_y)
    #     loss = self.criterion(pred_y, batch_y)
    #     self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=False)
    #     return loss

    
    def test_step(self, batch, batch_idx):
        # 解包 batch
        dataset_idx, batch_data = batch
        batch_x, batch_y = batch_data

        # 前向传播，包含 dataset_idx 以便于根据数据集调整行为
        pred_y = self(batch_x, dataset_idx)

        # 根据不同的 dataset_idx 处理预测的长度
        if dataset_idx == 2:  # 如果是第三个数据集
            pred_y = pred_y[:, :12]  # 使用 12 帧的预测
        else:
            pred_y = pred_y[:, :4]  # 使用 4 帧的预测

        # 将输入、预测和真实值保存到输出中
        outputs = {
            'inputs': batch_x.cpu().numpy(),
            'preds': pred_y.cpu().numpy(),
            'trues': batch_y.cpu().numpy()
        }

        # 将结果追加到 test_outputs 列表中
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