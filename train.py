from fastai.basics import *
from fastai.callback.all import *
from fastai.distributed import *
from fastprogress import fastprogress
from fastai.callback.mixup import *
from fastcore.script import *
from fastai.text.all import *

# added by deb
from torch.nn.parallel import DistributedDataParallel
# added by deb

torch.backends.cudnn.benchmark = True
fastprogress.MAX_COLS = 80

_model_meta = {
            AWD_LSTM: {'hid_name':'emb_sz', 'url':URLs.WT103_FWD, 'url_bwd':URLs.WT103_BWD,'config_lm':awd_lstm_lm_config, 'split_lm': awd_lstm_lm_split,            
                'config_clas':awd_lstm_clas_config, 'split_clas': awd_lstm_clas_split
              }
        }

def pr(s):
    if rank_distrib()==0: print(s)

def precision_at_k(yhat_raw, y, k=15):
    """
        Inputs: 
            yhat_raw: activation matrix of ndarray and shape (n_samples, n_labels)
            y: binary ground truth matrix of type ndarray and shape (n_samples, n_labels)
            k: for @k metric
    """
    yhat_raw, y = to_np(yhat_raw), to_np(y)
    # num true labels in the top k predictions / k
    sortd = yhat_raw.argsort()[:,::-1]
    topk = sortd[:, :k]
    
    # get precision at k for each sample
    vals = []
    for i, tk in enumerate(topk):
        num_true_in_top_k = y[i,tk].sum()
        vals.append(num_true_in_top_k / float(k))
    
    return np.mean(vals)

# added by deb
@patch
def before_fit(self:DistributedTrainer):
    opt_kwargs = { 'find_unused_parameters' : DistributedTrainer.fup  } if DistributedTrainer.fup is not None else {}

    # added by deb
    self.learn.model = self.learn.model.to(rank_distrib())
    # added by deb
    
    self.learn.model = DistributedDataParallel(
            nn.SyncBatchNorm.convert_sync_batchnorm(self.model) if self.sync_bn else self.model,
            device_ids=[self.cuda_id], output_device=self.cuda_id, broadcast_buffers=False, **opt_kwargs
            )
    self.old_dls = list(self.dls)
    self.learn.dls.loaders = [self._wrap_dl(dl) for dl in self.dls]
    if rank_distrib(): self.learn.logger=noop

@call_parse
def main(
    lr:    Param("base Learning rate", float)=1e-2,
    bs:    Param("Batch size", int)=128,
    epochs:Param("Number of epochs", int)=1,
    fp16:  Param("Use mixed precision training", store_true)=False,
    dump:  Param("Print model; don't train", int)=0,
    runs:  Param("Number of times to repeat training", int)=1,
):
    path = Path.cwd()
    path_data = path/'data'
    path_model = path/'models'

    dls_clas = rank0_first(torch.load, path_model/'dls_clas_sample_new.pkl')
    
    for run in range(runs):
        pr(f'Rank[{rank_distrib()}] Run: {run}; epochs: {epochs}; lr: {lr}; bs: {bs}')

        # learn = rank0_first(text_classifier_learner, dls_clas, AWD_LSTM, drop_mult=0.1, metrics=precision_at_k)
        
        # start: creating learner by bare hands
        
        arch = AWD_LSTM
        meta = _model_meta[arch]
        
        config = None
        config = ifnone(config, meta['config_clas']).copy()

        drop_mult = 0.1
        for k in config.keys():
            if k.endswith('_p'): config[k] *= drop_mult

        lin_ftrs, ps = None, None
        if lin_ftrs is None: lin_ftrs = [50]
        if ps is None: ps = [0.1] * len(lin_ftrs)

        n_out = get_c(dls_clas)

        emb_sz = config.get('emb_sz')

        layers = [emb_sz * 3] + lin_ftrs + [n_out]
        layers_attn = [n_out * (emb_sz * 3 + emb_sz)] + lin_ftrs + [n_out] # attention

        ps = [config.pop('output_p')] + ps

        init = config.pop('init') if 'init' in config else None

        vocab = dls_clas.vocab[0]

        seq_len = 72

        encoder = SentenceEncoder(seq_len, arch(vocab_sz=len(vocab), **config), pad_idx=1, max_len=seq_len*20)

        y_range = None
        
        class OurPoolingLinearClassifier(Module):
            def __init__(self, dims, ps, bptt, y_range=None):
                if len(ps) != len(dims)-1: raise ValueError("Number of layers and dropout values do not match.")
                acts = [nn.ReLU(inplace=True)] * (len(dims) - 2) + [None]
                layers = [LinBnDrop(i, o, p=p, act=a) for i,o,p,a in zip(dims[:-1], dims[1:], ps, acts)]
                if y_range is not None: layers.append(SigmoidRange(*y_range))
                self.layers = nn.Sequential(*layers)
                self.bptt = bptt

            def forward(self, input):
                out, mask = input
                x = masked_concat_pool(out, mask, self.bptt)
                x = self.layers(x)
                return x, out, out
        
        class OurPoolingAttentionClassifier(Module):
            def __init__(self, dims, ps, bptt, y_range=None):
                if len(ps) != len(dims)-1: raise ValueError("Number of layers and dropout values do not match.")
                acts = [nn.ReLU(inplace=True)] * (len(dims) - 2) + [None]
                layers = [LinBnDrop(i, o, p=p, act=a) for i,o,p,a in zip(dims[:-1], dims[1:], ps, acts)]
                if y_range is not None: layers.append(SigmoidRange(*y_range))
                self.layers = nn.Sequential(*layers)
                self.bptt = bptt

                # new
                self.emb_label = nn.Embedding(n_out, emb_sz)
                self.lin = nn.Linear(emb_sz, emb_sz)
                self.lin_for_tok_red = nn.Linear(seq_len*20, 50)
                self.V = self._init_param(emb_sz)
                # new

            def forward(self, input):
                out, mask = input
                x = masked_concat_pool(out, mask, self.bptt)

                # new
                num_tok = out.shape[1]
                out = F.pad(out, (0,0,0,seq_len*20-num_tok))
                bs = out.shape[0]
                label_indices = torch.arange(n_out, device=out.device)
                labels = label_indices.repeat(bs, 1)
                after_grabbing_label_embedding = self.emb_label(labels)
                after_first_matmul = self.lin(after_grabbing_label_embedding)
                out = out.permute(0,2,1).contiguous()
                out = self.lin_for_tok_red(out)
                out = out.permute(0,2,1).contiguous()
                after_nonlinearity = torch.tanh(out[:, :, None] + after_first_matmul[:,None])
                attn_wgts = (after_nonlinearity @ self.V)
                ctx = (out[:, :, None] * attn_wgts[..., None])
                ctx = ctx.sum(1)
                x = x[:, None]
                x = x.repeat(1, n_out, 1)
                x = torch.cat((x, ctx), dim=-1)
                x = x.view(x.shape[0], -1)
                # new

                x = self.layers(x)
                return x, out, out

            def _init_param(self, *sz): 
                return nn.Parameter(torch.randn(sz)/math.sqrt(sz[0]))

        # decoder = OurPoolingLinearClassifier(layers, ps, bptt=seq_len, y_range=y_range)
        decoder = OurPoolingAttentionClassifier(layers_attn, ps, bptt=seq_len, y_range=y_range) # attention

        model = SequentialRNN(encoder, decoder)

        if init is not None: model = model.apply(init)

        learn = rank0_first(TextLearner, dls_clas, model, splitter=meta['split_clas'], metrics=precision_at_k)

        learn.freeze()

        learn = rank0_first(learn.load_encoder, path_model/'lm_finetuned')

        # end: creating learner by bare hands
        
        if dump: pr(learn.model); exit()
        if fp16: learn = learn.to_fp16()

        DistributedTrainer.fup = True
        
        with learn.distrib_ctx(sync_bn=False):
            learn.fine_tune(epochs, lr)

