# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/03_text.learner.ipynb.

# %% auto 0
__all__ = ['match_collab', 'load_collab_keys', 'TextLearner', 'text_classifier_learner']

# %% ../../nbs/03_text.learner.ipynb 2
from fastai.basics import *
from fastai.text.learner import *
from fastai.callback.rnn import *
from fastai.text.models.awdlstm import *
from .models.core import *

# %% ../../nbs/03_text.learner.ipynb 6
def _get_text_vocab(dls:DataLoaders) -> list:
    "Get text vocabulary from `DataLoaders`"
    vocab = dls.vocab
    if isinstance(vocab, L): vocab = vocab[0]
    return vocab

# %% ../../nbs/03_text.learner.ipynb 7
def _get_label_vocab(dls:DataLoaders) -> list:
    "Get label vocabulary from `DataLoaders`"
    vocab = dls.vocab
    if isinstance(vocab, L): vocab = vocab[1]
    return vocab

# %% ../../nbs/03_text.learner.ipynb 8
def match_collab(
    old_wgts:dict, # Embedding weights of the colab model
    collab_vocab:dict, # Vocabulary of `token` and `label` used for colab pre-training
    lbs_vocab:list # Current labels vocabulary
) -> dict:
    "Convert the label embedding in `old_wgts` to go from `old_vocab` in colab to `lbs_vocab`"
    bias, wgts = old_wgts.get('i_bias.weight', None), old_wgts.get('i_weight.weight')
    wgts_m = wgts.mean(0)
    new_wgts = wgts.new_zeros((len(lbs_vocab), wgts.size(1)))
    if bias is not None:
        bias_m = bias.mean(0)
        new_bias = bias.new_zeros((len(lbs_vocab), 1))
    collab_lbs_vocab = collab_vocab['label']
    collab_o2i = collab_lbs_vocab.o2i if hasattr(collab_lbs_vocab, 'o2i') else {w:i for i,w in enumerate(collab_lbs_vocab)}
    missing = 0
    for i,w in enumerate(lbs_vocab):
        idx = collab_o2i.get(w, -1)
        new_wgts[i] = wgts[idx] if idx>=0 else wgts_m
        if bias is not None: new_bias[i] = bias[idx] if idx>=0 else bias_m
        if idx == -1: missing = missing + 1
    old_wgts['i_weight.weight'] = new_wgts
    if bias is not None: old_wgts['i_bias.weight'] = new_bias
    return old_wgts, missing

# %% ../../nbs/03_text.learner.ipynb 11
def load_collab_keys(
    model, # Model architecture
    wgts:dict # Model weights
) -> tuple:
    "Load only collab `wgts` (`i_weight` and `i_bias`) in `model`, keeping the rest as is"
    sd = model.state_dict()
    lbs_weight, i_weight = sd.get('1.attn.lbs_weight.weight', None), wgts.get('i_weight.weight', None)
    lbs_bias, i_bias = sd.get('1.attn.lbs_weight.bias', None), wgts.get('i_bias.weight', None) 
    if lbs_weight is not None and i_weight is not None: lbs_weight.data = i_weight.data
    if lbs_bias is not None and i_bias is not None: lbs_bias.data = i_bias.data
    if '1.attn.lbs_weight_dp.emb.weight' in sd:
        sd['1.attn.lbs_weight_dp.emb.weight'] = i_weight.data.clone()
    return model.load_state_dict(sd)

# %% ../../nbs/03_text.learner.ipynb 13
@delegates(Learner.__init__)
class TextLearner(Learner):
    "Basic class for a `Learner` in NLP."
    def __init__(self, 
        dls:DataLoaders, # Text `DataLoaders`
        model, # A standard PyTorch model
        alpha:float=2., # Param for `RNNRegularizer`
        beta:float=1., # Param for `RNNRegularizer`
        moms:tuple=(0.8,0.7,0.8), # Momentum for `Cosine Annealing Scheduler`
        **kwargs
    ):
        super().__init__(dls, model, moms=moms, **kwargs)
        self.add_cbs(rnn_cbs())

    def save_encoder(self, 
        file:str # Filename for `Encoder` 
    ):
        "Save the encoder to `file` in the model directory"
        if rank_distrib(): return # don't save if child proc
        encoder = get_model(self.model)[0]
        if hasattr(encoder, 'module'): encoder = encoder.module
        torch.save(encoder.state_dict(), join_path_file(file, self.path/self.model_dir, ext='.pth'))
    
    @delegates(save_model)
    def save(self,
        file:str, # Filename for the state_directory of the model
        **kwargs
    ):
        """
        Save model and optimizer state (if `with_opt`) to `self.path/self.model_dir/file`
        Save `self.dls.vocab` to `self.path/self.model_dir/clas_vocab.pkl`
        """
        model_file = join_path_file(file, self.path/self.model_dir, ext='.pth')
        vocab_file = join_path_file(file+'_vocab', self.path/self.model_dir, ext='.pkl')
        save_model(model_file, self.model, getattr(self, 'opt', None), **kwargs)
        save_pickle(vocab_file, self.dls.vocab)
        return model_file

    def load_encoder(self, 
        file:str, # Filename of the saved encoder 
        device:(int,str,torch.device)=None # Device used to load, defaults to `dls` device
    ):
        "Load the encoder `file` from the model directory, optionally ensuring it's on `device`"
        encoder = get_model(self.model)[0]
        if device is None: device = self.dls.device
        if hasattr(encoder, 'module'): encoder = encoder.module
        distrib_barrier()
        wgts = torch.load(join_path_file(file,self.path/self.model_dir, ext='.pth'), map_location=device)
        encoder.load_state_dict(clean_raw_keys(wgts))
        self.freeze()
        return self

    def load_pretrained(self, 
        wgts_fname:str, # Filename of saved weights 
        vocab_fname:str, # Saved vocabulary filename in pickle format
        model=None # Model to load parameters from, defaults to `Learner.model`
    ):
        "Load a pretrained model and adapt it to the data vocabulary."
        old_vocab = load_pickle(vocab_fname)
        new_vocab = _get_text_vocab(self.dls)
        distrib_barrier()
        wgts = torch.load(wgts_fname, map_location = lambda storage,loc: storage)
        if 'model' in wgts: wgts = wgts['model'] #Just in case the pretrained model was saved with an optimizer
        wgts = match_embeds(wgts, old_vocab, new_vocab)
        load_ignore_keys(self.model if model is None else model, clean_raw_keys(wgts))
        self.freeze()
        return self

    #For previous versions compatibility. Remove at release
    @delegates(load_model_text)
    def load(self, 
        file:str, # Filename of saved model 
        with_opt:bool=None, # Enable to load `Optimizer` state
        device:(int,str,torch.device)=None, # Device used to load, defaults to `dls` device
        **kwargs
    ):
        if device is None: device = self.dls.device
        if self.opt is None: self.create_opt()
        file = join_path_file(file, self.path/self.model_dir, ext='.pth')
        load_model_text(file, self.model, self.opt, device=device, **kwargs)
        return self
    
    def load_collab(self,
        wgts_fname:str, # Filename of the saved collab model
        collab_vocab_fname:str, # Saved Vocabulary of collab labels in pickle format 
        model=None # Model to load parameters from, defaults to `Learner.model`
    ):
        "Load the label embeddings learned by collab model`, and adapt it to the label vocabulary."
        collab_vocab = load_pickle(collab_vocab_fname)
        lbs_vocab = _get_label_vocab(self.dls)
        distrib_barrier()
        wgts = torch.load(wgts_fname, map_location=lambda storage,loc: storage)
        if 'model' in wgts: wgts = wgts['model'] #Just in case the pretrained model was saved with an optimizer
        wgts, _ = match_collab(wgts, collab_vocab, lbs_vocab)
        load_collab_keys(self.model if model is None else model, wgts)
        self.freeze()
        return self

# %% ../../nbs/03_text.learner.ipynb 16
from .models.core import _model_meta 

# %% ../../nbs/03_text.learner.ipynb 17
@delegates(Learner.__init__)
def text_classifier_learner(dls, arch, seq_len=72, config=None, backwards=False, pretrained=True, collab=False, drop_mult=0.5, n_out=None,
                           lin_ftrs=None, ps=None, max_len=72*20, y_range=None, **kwargs):
    "Create a `Learner` with a text classifier from `dls` and `arch`."
    vocab = _get_text_vocab(dls)
    if n_out is None: n_out = get_c(dls)
    assert n_out, "`n_out` is not defined, and could not be inferred from the data, set `dls.c` or pass `n_out`"
    model = get_text_classifier(arch, len(vocab), n_out, seq_len=seq_len, config=config, y_range=y_range,
                                drop_mult=drop_mult, lin_ftrs=lin_ftrs, ps=ps, max_len=max_len)
    meta = _model_meta[arch]
    learn = TextLearner(dls, model, splitter=meta['split_clas'], **kwargs)
    url = 'url_bwd' if backwards else 'url'
    if pretrained:
        if url not in meta:
            warn("There are no pretrained weights for that architecture yet!")
            return learn
        model_path = untar_data(meta[url], c_key='model')
        try: fnames = [list(model_path.glob(f'*.{ext}'))[0] for ext in ['pth', 'pkl']]
        except IndexError: print(f'The model in {model_path} is incomplete, download again'); raise
        learn = learn.load_pretrained(*fnames, model=learn.model[0])
    if collab:
        try: fnames = [list(learn.path.glob(f'**/collab/*collab*.{ext}'))[0] for ext in ['pth', 'pkl']]
        except IndexError: print(f'The collab model in {learn.path} is incomplete, re-train it!'); raise
        learn = learn.load_colab(*fnames, model=learn.model[1])
    learn.freeze()
    return learn   
