# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/01_layers.ipynb.

# %% auto 0
__all__ = ['LinBnDrop', 'XMLAttention']

# %% ../nbs/01_layers.ipynb 3
from fastai.imports import *
from fastai.torch_imports import *
from fastai.torch_core import *
from fastai.layers import *

# %% ../nbs/01_layers.ipynb 4
from fastai.text.models.awdlstm import EmbeddingDropout, RNNDropout

# %% ../nbs/01_layers.ipynb 11
class LinBnDrop(nn.Sequential):
    "Module grouping `BatchNorm1d`, `Dropout` and `Linear` layers"
    def __init__(self, n_in, n_out=None, bn=True, ln=True, p=0., act=None, lin_first=False):
        layers = [BatchNorm(n_out if ln and lin_first else n_in, ndim=1)] if bn else []
        if p != 0: layers.append(nn.Dropout(p))
        lin = [nn.Linear(n_in, n_out, bias=not bn)] if ln else []
        if ln and act is not None: lin.append(act)
        layers = lin+layers if lin_first else layers+lin
        super().__init__(*layers)

# %% ../nbs/01_layers.ipynb 18
class XMLAttention(Module):
    "Compute label specific attention weights for each token in a sequence"
    def __init__(self, n_lbs, emb_sz, embed_p):
         store_attr('n_lbs,emb_sz,embed_p')
         self.lbs_weight = nn.Embedding(n_lbs, emb_sz)
         self.lbs_weight_dp = EmbeddingDropout(self.lbs_weight, embed_p)
         self.lbs_weight.weight.data.normal_(0, 0.01)   
         self.input_dp = RNNDropout(0.02)

    def forward(self, x):
        lbs_emb = self.lbs_weight(torch.arange(self.n_lbs, device=x.device))
        # x_dp = self.input_dp(x)
        attn_wgts = F.softmax(x @ lbs_emb.transpose(0,1), dim=1)
        return attn_wgts.transpose(1,2) @ x
    
