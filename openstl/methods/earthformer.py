import torch
from openstl.models import MultiEarthFormer_Model
from .base_method import Base_method, Base_multi_method


class MultiEarthFormer(Base_multi_method):
    r"""SimVP for multiple encoders and decoders

    Implementation of `SimVP: Simpler yet Better Video Prediction
    <https://arxiv.org/abs/2206.05099>`_.

    """
   
    def __init__(self, enc_dec_configs, **args):
        super().__init__(enc_dec_configs=enc_dec_configs, **args)
        print(f"MultiEarthFormer enc_dec_configs initialized: {self.hparams.enc_dec_configs}")

    def _build_model(self, enc_dec_configs, **kwargs):
        return MultiEarthFormer_Model(enc_dec_configs, **kwargs)
    
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