from fastbook import *
from multiprocessing import Pool
from tqdm import *

def count_star(args): return count(*args)

def count(tokens_, labels_):
    tlfq_ = Counter()
    tlfq_.update(itertools.product(tokens_, labels_))
    return tlfq_

def gen_():
    for x,y in dsets.train:
        tokens_ = dsets.tfms[0][2].decode(x)
        labels_ = dsets.tfms[1].decode(y)
        yield (tokens_, labels_)

if __name__ == '__main__':
    start = time.time()
    dsets = torch.load('dsets.pkl')
    data_gen = gen_()
    # data_gen = ((dsets.tfms[0][2].decode(x), dsets.tfms[1].decode(y)) for x, y in dsets.train)
    with Pool(processes=8) as pool:
        tlfq = Counter()
        with tqdm(total=len(dsets.train)) as pbar:
            for o in pool.imap_unordered(count_star, data_gen, chunksize=128):
                tlfq.update(o)
                pbar.update()
    torch.save(tlfq, 'tlfq.pkl')
    print(f"time = {time.time() - start}")
        






    
