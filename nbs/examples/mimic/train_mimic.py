from fastai.basics import *
from fastai.distributed import *
from fastprogress import fastprogress
from fastai.text.all import *
from xcube.text.learner import text_classifier_learner
from xcube.metrics import PrecisionK

torch.backends.cudnn.benchmark = True
fastprogress.MAX_COLS = 80
def pr(s):
    if rank_distrib()==0: print(s)

def splitter(df):
    train = df.index[~df['is_valid']].tolist()
    valid = df.index[df['is_valid']].tolist()
    return train, valid

def get_dls(data_file, lm_vocab_file, bs=128, workers=None):
    workers=ifnone(workers, min(8, num_cpus()))
    df = pd.read_csv(data_file, dtype={'text': str, 'labels': str})
    df[['text', 'labels']] = df[['text', 'labels']].astype('str')
    lm_vocab = torch.load(lm_vocab_file)
    label_freq = Counter()
    for labels in df.labels: label_freq.update(labels.split(';'))
    lbls = list(label_freq.keys())
    dblock = DataBlock(
            blocks   = (TextBlock.from_df('text', seq_len=72, vocab=lm_vocab), MultiCategoryBlock(vocab=lbls)),
            get_x    = ColReader('text'),
            get_y    = ColReader('labels', label_delim=';'),
            splitter = splitter
            )
    return dblock.dataloaders(df, bs=bs, num_workers=workers)

@call_parse
def main(
    lr:     Param("base Learning rate", float) = 1e-2,
    bs:     Param("Batch size", int) = 128,
    epochs: Param("Number of epochs", int) = 1,
    fp16:   Param("Use mixed precision training", store_true) = False,
    dump:   Param("Print model, don't train", int) = 0,
    runs:   Param("Number of times to repeat training", int) = 1,
):
    path = Path.cwd()
    path_model = path/'models'
    path_data = path/'data'

    # dls = torch.load(path_model/'dls_clas_sample.pkl')
    if (path_model/'dls_clas_full_32.pkl').exists():
        dls = torch.load(path_model/'dls_clas_full_32.pkl')
    else:    
        dls = get_dls(path_data/'notes_labelled.csv', path_model/'dls_lm_vocab.pkl', bs=bs)
        torch.save(dls, path_model/'dls_clas_full_32.pkl')

    for run in range(runs):
        pr(f'Rank[{rank_distrib()}] Run: {run}; epochs: {epochs}; lr: {lr}; bs: {bs}')

        learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.01, metrics=PrecisionK)
        learn = learn.load_encoder(path_model/'lm_finetuned')

        if dump: pr(learn.model); exit()
        if fp16: learn = learn.to_fp16()

        lr = 1e-1 * bs/128
        learn.fit_one_cycle(1, lr, moms=(0.8,0.7,0.8), wd=0.001)

        learn.freeze_to(-2)
        lr /= 2
        learn.fit_one_cycle(1, lr, moms=(0.8,0.7,0.8), wd=0.001)


        learn.freeze_to(-3)
        lr /= 2
        learn.fit_one_cycle(1, lr, moms=(0.8,0.7,0.8), wd=0.001)

        #learn = learn.load(path_model/'clas')
        learn.unfreeze()
        lr /= 5
        #import pdb; pdb.set_trace()
        learn.fit_one_cycle(1, lr, moms=(0.8,0.7,0.8), wd=0.001, cbs=[SaveModelCallback(fname=path_model/'clas', at_end=True), ReduceLROnPlateau(monitor='valid_loss', min_delta=0.0001, patience=4)])




