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

    def __init__(self, **args):
        super().__init__(**args)

    def _build_model(self, enc_dec_configs, **kwargs):
        return Multi_SimVP_Model(enc_dec_configs, **kwargs)
    
    def forward(self, batch_x, data_cls_idx, batch_y=None, **kwargs):
        data_cls_idx = data_cls_idx[0].item()    # data_cls_idx remains the same for one batch.
        pre_seq_length, aft_seq_length = self.hparams.enc_dec_configs[data_cls_idx]['pre_seq_length'], self.hparams.enc_dec_configs[data_cls_idx]['aft_seq_length']
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
    
    def training_step(self, batch, batch_idx):
        dataset_idx, batch_data = batch
        batch_x, batch_y = batch_data
        pred_y = self(batch_x, dataset_idx)
        loss = self.criterion(pred_y, batch_y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss