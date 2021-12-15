# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/03_text.learner.ipynb (unless otherwise specified).

__all__ = ['text_classifier_learner']

# Cell
from fastai.basics import *
from fastai.text.learner import *
from .models.core import *

# Cell
from .models.core import _model_meta

# Cell
def _get_text_vocab(dls):
    vocab = dls.vocab
    if isinstance(vocab, L): vocab = vocab[0]
    return vocab

# Cell
@delegates(Learner.__init__)
def text_classifier_learner(dls, arch, seq_len=72, config=None, backwards=False, pretrained=True, drop_mult=0.5, n_out=None,
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
        learn.freeze()
    return learn