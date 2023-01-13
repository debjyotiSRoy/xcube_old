# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/11_l2r.callbacks.ipynb.

# %% auto 0
__all__ = ['TrackResults', 'PrintResults']

# %% ../../nbs/11_l2r.callbacks.ipynb 2
from fastai.torch_imports import *
from fastai.torch_core import *
from fastai.callback.core import *
from fastcore.all import *
from ..imports import *
from ..metrics import *

# %% ../../nbs/11_l2r.callbacks.ipynb 6
class TrackResults(Callback):
    def before_fit(self):
            pass
        
    def before_epoch(self): self.losses, self.ndcgs, self.ndcgs_at_6, self.accs, self.logger = [], [], [], [], defaultdict(list)
    
    def after_epoch(self):
        _li = [self.losses, self.ndcgs, self.ndcgs_at_6, self.accs]
        _li = [torch.stack(o) if o else torch.Tensor() for o in _li] 
        [self.losses, self.ndcgs, self.ndcgs_at_6, self.accs] = _li
        # pdb.set_trace()
        log = [round(o.mean().item(), 4) if o.sum() else "NA" for o in _li]
        # if self.model.training: self.logger['trn'] = log 
        # else: self.logger['val'] = log
        print(self.epoch, self.model.training, *log)
        # return logger
    
    def after_batch(self):
        with torch.no_grad():
            loss = self.loss_func(self.preds, self.xb)
            self.losses.append(loss.mean())
            if self.model.training:
                if self.track_trn: self._compute_metrics()
            else: self._compute_metrics()
                        
    def _compute_metrics(self):
        *_, _ndcg, _ndcg_at_k = ndcg(self.preds, self.xb, k=6)
        self.ndcgs.append(_ndcg.mean())
        self.ndcgs_at_6.append(_ndcg_at_k.mean())
        acc = accuracy(self.xb, self.model).mean()
        self.accs.append(acc.mean())

# %% ../../nbs/11_l2r.callbacks.ipynb 7
class PrintResults(Callback):
    pass
