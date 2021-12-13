from fastai.basics import *
from fastai.distributed import *
from fastprogress import fastprogress
from fastai.text.all import *
from xcube.text.learner import text_classifier_learner

torch.backends.cudnn.benchmark = True
fastprogress.MAX_COLS = 80
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

@call_parse
def main(
    lr:     Param("base Learning rate", float) = 1e-2,
    bs:     Param("Batch size", int) = 128,
    epochs: Param("Number of epochs", int) = 2,
    fp16:   Param("Use mixed precision training", store_true) = False,
    dump:   Param("Print model, don't train", int) = 0,
    runs:   Param("Number of times to repeat training", int) = 1,
):
    path = Path.cwd()
    path_model = path/'models'

    dls = torch.load(path_model/'dls_clas_sample_new.pkl')

    for run in range(runs):
        pr(f'Rank[{rank_distrib()}] Run: {run}; epochs: {epochs}; lr: {lr}; bs: {bs}')

        learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.1, metrics=precision_at_k)
        learn = learn.load_encoder(path_model/'lm_finetuned')

        if dump: pr(learn.model); exit()
        if fp16: learn = learn.to_fp16()

        lr_min, lr_steep, lr_valley, lr_slide = learn.lr_find(suggest_funcs=(minimum, steep, valley, slide))

        learn.fit_one_cycle(epochs, lr_min)
