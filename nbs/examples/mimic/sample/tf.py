import fastbook
fastbook.setup_book()

from fastbook import *
from multiprocessing import Pool
from tqdm import *

def count(tokens):
    vocab = Counter()
    vocab.update(tokens)
    return vocab

if __name__ == '__main__':

    start = time.time()
    dsets = torch.load('dsets.pkl')
    data_gen = (dsets.tfms[0][2].decode(x) for x,_ in dsets.train)

    with Pool(processes=8) as pool:
        max_ = len(dsets.train)
        final_vocab = Counter()
        with tqdm(total=max_) as pbar:
            for o in pool.imap(count, data_gen):
                final_vocab.update(o)
                pbar.update()
    torch.save(final_vocab, 'tf.pkl')
    print(f"time = {time.time() - start}")
