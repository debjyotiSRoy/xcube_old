from fastai.basics import *
from fastai.distributed import *
from fastprogress import fastprogress
from fastai.text.all import *
from xcube.text.learner import text_classifier_learner
from xcube.metrics import PrecisionK

torch.backends.cudnn.benchmark = True
fastprogress.MAX_COLS = 80

def get_dls_lm(data, workers=None):
    # import pdb; pdb.set_trace()
    workers=ifnone(workers, min(8, num_cpus()))
    df = pd.read_csv(data, header=0, names=['text', 'labels', 'is_valid'], dtype={'text': str, 'labels': str})
    df[['text', 'labels']] = df[['text', 'labels']].astype(str)
    
    dls_lm = DataBlock(
            blocks   = TextBlock.from_df('text', is_lm=True),
            get_x    = ColReader('text'),
            splitter = RandomSplitter(0.1)
            ).dataloaders(df, bs=128, seq_len=80, num_workers=workers)
    
    return dls_lm

@call_parse
def main(
        lr: Param("base Learning rate", float) = 1e-3,
        bs: Param("Batch size", int) = 128,
        ):
    path = Path.cwd()
    path_data = path/'data'
    path_model = path/'models'

    path_model.mkdir(exist_ok=True)
    path_data.mkdir(exist_ok=True)

    file_prefix = 'wiki-500k'

    data = path_data/(file_prefix+'.csv')
    dls_lm_path = path_model/f"{file_prefix}_dls_lm.pkl"
    dls_lm_vocab_path = path_model/f"{file_prefix}_dls_lm_vocab.pkl"
    dls_clas_path = path_model/f"{file_prefix}_dls_clas.pkl"

    print(f"{ path = },\n {path_data = },\n {path_model = },\n {file_prefix = },\n {data = },\n {dls_lm_path =},\n {dls_lm_vocab_path = },\n {dls_clas_path = }.")

    if dls_lm_path.exists():
        dls_lm = torch.load(dls_lm_path)
    else:
        dls_lm = get_dls_lm(data)
        torch.save(dls_lm, dls_lm_path)
        torch.save(dls_lm.vocab, dls_lm_vocab_path)
