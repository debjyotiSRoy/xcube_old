# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/09_l2r.learner.ipynb.

# %% auto 0
__all__ = ['Learner', 'get_learner']

# %% ../../nbs/09_l2r.learner.ipynb 2
from fastai.torch_imports import *
from fastai.learner import *
from fastai.optimizer import *
from fastai.torch_core import *
from fastcore.all import *
from ..imports import *
from ..metrics import *
from .gradients import *

# %% ../../nbs/09_l2r.learner.ipynb 6
class Learner:
    def __init__(self, model, dls, grad_func, loss_func, lr, cbs, opt_func=SGD, path=None):
        store_attr()
        if cbs: 
            for cb in cbs: cb.learn = self
        self.path = Path(path) if path is not None else getattr(dls, 'path', Path('.'))
   
    # def one_batch(self, losses, ndcgs, ndcgs_at_6, accs, track_trn=True, logger=None, grad_logger=None, metric_logger=None, **kwargs): #cb
    def one_batch(self, logger=None, grad_logger=None, **kwargs):
        self('before_batch')
        self.preds = self.model(self.xb)
        if self.model.training: # training
            srtd_preds, lambda_i = self.grad_func(self.preds, self.xb)
            srtd_preds.backward(lambda_i)
            
            # free memory
            lambda_i = None
            import gc; gc.collect()
            torch.cuda.empty_cache()
            
            ## tracking gradients
            for name,param in self.model.named_parameters():
            # import pdb; pdb.set_trace()
            # from IPython import embed; embed()
                grad = param.grad.data.detach().clone()
                grad_logger[name].append(grad)
            
            # tracking loss
            if logger is not None:
                with torch.no_grad():
                    loss = self.loss_func(self.preds, self.xb)
                    logger.append(loss.mean())
                    # losses.append(loss.mean()) #cb
            
            ## stepping the params
            self.opt.step()
            ## zeroing the grad before next batch
            self.opt.zero_grad()
            
            # tracking metrics during training #cb
            # if track_trn:
            #     with torch.no_grad():
            #         *_, _ndcg, _ = ndcg(self.preds, self.xb)
            #         btch_ndcg_mean = _ndcg.mean()
            #         ndcgs.append(btch_ndcg_mean)
            #         btch_acc_mean = accuracy(self.xb, self.model).mean()
            #         accs.append(btch_acc_mean)
            
        # else: # validation #cb
        #     loss = self.loss_func(self.preds, self.xb)
        #     losses.append(loss.mean())
        #     *_, _ndcg, _ndcg_at_k = ndcg(self.preds, self.xb, k=6)
        #     ndcgs.append(_ndcg.mean())
        #     ndcgs_at_6.append(_ndcg_at_k.mean())
        #     acc = accuracy(self.xb, self.model)
        #     accs.append(acc.mean())
            
        self('after_batch')
        
        # return losses, ndcgs, ndcgs_at_6, accs    #cb
        
    def one_epoch(self, train, mb, metric_logger=None, **kwargs):
        # losses, ndcgs, ndcgs_at_6, accs = [], [], [], [] #cb
        self.model.training = train
        self('before_epoch')
        dl = self.dls.train if train else self.dls.valid
        for self.num, self.xb in enumerate(progress_bar(dl, parent=mb, leave=False)):
            # losses, ndcgs, ndcgs_at_6, accs = self.one_batch(losses, ndcgs, ndcgs_at_6, accs, **kwargs) # cb
            self.one_batch(**kwargs)
        # _li = [losses, ndcgs, ndcgs_at_6, accs] # cb
        # _li = [torch.stack(o) if o else torch.Tensor() for o in _li] # cb
        # [losses, ndcgs, ndcgs_at_6, accs] = _li #cb
        # logger = [round(o.mean().item(), 4) if o.sum() else "NA" for o in _li] # cb
        self('after_epoch')
        # pdb.set_trace()
        if not self.model.training and metric_logger is not None: metric_logger.append(logger)
        # return logger # cb
    
    def create_opt(self):
        self.opt = self.opt_func(self.model.parameters(), self.lr)
        # self.opt.clear_state()
        return self.opt
    
    def fit(self, n_epochs, best=None, track_trn=True, **kwargs):
        opt = getattr(self, 'opt', None)
        if opt is None: self.create_opt()
        self.n_epochs = n_epochs
        self.track_trn = track_trn
        self('before_fit')
        mb = master_bar(range(self.n_epochs))
        # columns=['train_loss', 'train_ndcg', 'train_ndcg@6', 'train_acc', 'val_loss', 'val_ndcg (candi. 32)', 'val ndcg@6 (candi. 32)', 'val_acc']
        # pdf = pd.DataFrame(columns=columns)#, index=index)
        # pdf.index.name = 'epoch'
        # if best is not None and best[0] not in columns: raise NameError(best[0]+'metric is not trackable, please check name!')
        try:
            for self.epoch,_ in enumerate(mb):
            #     pdf.loc[self.epoch] = pd.Series(dict(zip(columns, self.one_epoch(True, mb, **kwargs) + self.one_epoch(False, mb, **kwargs)))) #cb
                self.one_epoch(True, mb, **kwargs)
                self.one_epoch(False, mb, **kwargs)
            #     if best is not None: 
            #         current = pdf.loc[self.epoch][best[0]]
            #         if current >= best[1]:
            #             best[1] = current
            #             self.save(best[2])
            #     display_df(pdf.iloc[[self.epoch]])
            # clear_output(wait=True)
            # display_df(pdf)
        except CancelFitException: pass 
        self('after_fit')
    
    def validate(self, *args, **kwargs):
        columns=['val_loss', 'val_ndcg (candi. 32)', 'val ndcg@6 (candi. 32)', 'val_acc']
        pdf = pd.DataFrame(columns=columns)
        pdf.index.name = 'epoch'
        try: 
            val = dict(zip(columns, self.one_epoch(False, None, **kwargs)))
            pdf = pd.DataFrame([val])
            display_df(pdf)
        except CancelFitException: pass
    
    def __call__(self, name):
        for cb in self.cbs: getattr(cb, name, noop)()

# %% ../../nbs/09_l2r.learner.ipynb 8
@patch
@delegates(save_model)
def save(self:Learner, file, **kwargs):
    "Save model and optimizer state (if 'with_opt') to `self.path/file`"
    file = join_path_file(file, self.path, ext='.pth')
    save_model(file, self.model, getattr(self, 'opt', None), **kwargs)
    return file

# %% ../../nbs/09_l2r.learner.ipynb 9
@patch
@delegates(load_model)
def load(self:Learner, file, device=None, **kwargs):
    "Load model and optimizer state (if `with_opt`) from `self.path/file` using `device`"
    if device is None and hasattr(self.dls, 'device'): device = self.dls.device
    self.opt = getattr(self, 'opt', None)
    if self.opt is None: self.create_opt()
    file = join_path_file(file, self.path, ext='.pth')
    load_model(file, self.model, self.opt, device=device, **kwargs)
    return self

# %% ../../nbs/09_l2r.learner.ipynb 11
def get_learner(model, dls, grad_fn=rank_loss3, loss_fn=loss_fn2, lr=1e-5, cbs=None, opt_func=partial(SGD, mom=0.9), lambrank=False):
    if lambrank: grad_fn = partial(grad_fn, lambrank=lambrank)
    learner = Learner(model, dls, grad_fn, loss_fn, lr, cbs, opt_func=opt_func)
    return learner
