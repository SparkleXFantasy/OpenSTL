import torch
from openstl.models import SimVP_Model, Multi_SimVP_Model
from .base_method import Base_method, Base_multi_method


class SimVP(Base_method):
    r"""SimVP

    Implementation of `SimVP: Simpler yet Better Video Prediction
    <https://arxiv.org/abs/2206.05099>`_.

    """

    def __init__(self, **args):
        super().__init__(**args)

    def _build_model(self, **args):
        return SimVP_Model(**args)

    def forward(self, batch_x, batch_y=None, **kwargs):
        pre_seq_length, aft_seq_length = self.hparams.pre_seq_length, self.hparams.aft_seq_length
        if aft_seq_length == pre_seq_length:
            pred_y = self.model(batch_x)
        elif aft_seq_length < pre_seq_length:
            pred_y = self.model(batch_x)
            pred_y = pred_y[:, :aft_seq_length]
        elif aft_seq_length > pre_seq_length:
            pred_y = []
            d = aft_seq_length // pre_seq_length
            m = aft_seq_length % pre_seq_length
            
            cur_seq = batch_x.clone()
            for _ in range(d):
                cur_seq = self.model(cur_seq)
                pred_y.append(cur_seq)

            if m != 0:
                cur_seq = self.model(cur_seq)
                pred_y.append(cur_seq[:, :m])
            
            pred_y = torch.cat(pred_y, dim=1)
        return pred_y
    
    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        pred_y = self(batch_x)
        loss = self.criterion(pred_y, batch_y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    
class MultiSimVP(Base_multi_method):
    r"""SimVP for multiple encoders and decoders

    Implementation of `SimVP: Simpler yet Better Video Prediction
    <https://arxiv.org/abs/2206.05099>`_.

    """
   
    def __init__(self, enc_dec_configs, **args):
        super().__init__(enc_dec_configs=enc_dec_configs, **args)
        print(f"MultiSimVP enc_dec_configs initialized: {self.hparams.enc_dec_configs}")

    def _build_model(self, enc_dec_configs, **kwargs):
        return Multi_SimVP_Model(enc_dec_configs, **kwargs)
    
    def forward(self, batch_x, data_cls_idx, batch_y=None, **kwargs):
        
        if isinstance(data_cls_idx, (list, torch.Tensor)):
            data_cls_idx = data_cls_idx[0].item()
        elif isinstance(data_cls_idx, int):
            
            data_cls_idx = data_cls_idx
        else:
            raise TypeError(f"Unexpected type for data_cls_idx: {type(data_cls_idx)}")


        
        selected_enc_dec_config = self.hparams.enc_dec_configs[data_cls_idx]
        pre_seq_length = selected_enc_dec_config['pre_seq_length']
        aft_seq_length = selected_enc_dec_config['aft_seq_length']


        if aft_seq_length == pre_seq_length:
            pred_y = self.model(batch_x, data_cls_idx)
        elif aft_seq_length < pre_seq_length:
            pred_y = self.model(batch_x, data_cls_idx)
            pred_y = pred_y[:, :aft_seq_length]
        elif aft_seq_length > pre_seq_length:
            pred_y = []
            d = aft_seq_length // pre_seq_length
            m = aft_seq_length % pre_seq_length

            cur_seq = batch_x.clone()
            for _ in range(d):
                cur_seq = self.model(cur_seq, data_cls_idx)
                pred_y.append(cur_seq)

            if m != 0:
                cur_seq = self.model(cur_seq, data_cls_idx)
                pred_y.append(cur_seq[:, :m])

            pred_y = torch.cat(pred_y, dim=1)

        return pred_y
    
# def training_step(self, batch, batch_idx):
#     print(f"[DEBUG] Training Step Called - Batch {batch_idx}")
    
    
#     dataset_idx, batch_data = batch

   
#     if dataset_idx == 0:
        
#         batch_x = batch_data[:, :8]  
#         batch_y = batch_data[:, 8:]  
#         print(f"[DEBUG] Dataset Index: {dataset_idx} - Splitting into X: 8 frames, Y: 8 frames")
        
#     elif dataset_idx == 1:
        
#         batch_x = batch_data[:, :8]  
#         batch_y = batch_data[:, 8:]  
#         print(f"[DEBUG] Dataset Index: {dataset_idx} - Splitting into X: 8 frames, Y: 8 frames")

#     elif dataset_idx == 2:
        
#         batch_x = batch_data[:, :4]  
#         batch_y = batch_data[:, 4:]  
#         print(f"[DEBUG] Dataset Index: {dataset_idx} - Splitting into X: 4 frames, Y: 12 frames")

#     else:
#         raise ValueError(f"Unexpected dataset_idx: {dataset_idx}")

   
#     print(f"[DEBUG] Dataset Index: {dataset_idx}, Batch X shape: {batch_x.shape}, Batch Y shape: {batch_y.shape}")

    
#     pred_y = self(batch_x, dataset_idx)
#     print(f"[DEBUG] Prediction Y shape: {pred_y.shape}")

    
#     loss = self.criterion(pred_y, batch_y)
#     print(f"[DEBUG] Loss for Batch {batch_idx}: {loss.item()}")

#     # 记录损失值
#     self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
#     print(f"[DEBUG] Log Called for Train Loss - Batch {batch_idx}")

#     return loss

