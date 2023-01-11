# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/08_l2r.gradients.ipynb.

# %% auto 0
__all__ = ['rank_loss2', 'rank_loss3', 'loss_fn', 'loss_fn2']

# %% ../../nbs/08_l2r.gradients.ipynb 2
from fastai.torch_imports import *
from ..imports import *

# %% ../../nbs/08_l2r.gradients.ipynb 9
def _summation(sl, ij):
    sumer = []
    for i in range(sl):
        _x = torch.nonzero(ij == i, as_tuple=False)
        _x[:, -1] = torch.pow(-1, _x[:, 1])
        sumer.append(_x)
    return torch.stack(sumer, dim=0)

def _idcg(xb, k=None, gain_fn=None):
    # pdb.set_trace()
    x = xb[:, :, :, -1]
    ranks = x.argsort(dim=-1, descending=True).argsort(dim=-1) # ranking by the scores, highest score gets rank 0
    dfs = 1/torch.log2(ranks + 2)
    gains = torch.pow(2, x) if gain_fn == 'exp' else torch.pow(x, 3)
    idg = gains * dfs
    idcg = idg.sum(dim=-1)
    
    idcg_at_k = None
    if k is not None:
        topk, topk_idxs = torch.topk(x, k=k, dim=-1, largest=True)
        # topk_relvs = torch.take_along_dim(x, topk_idxs, dim=-1)
        dfs_at_k = 1/torch.log2(2 + torch.arange(k)).cuda()
        gains_at_k = torch.pow(2, topk) if gain_fn == 'exp' else torch.pow(topk, 3)
        idg_at_k = gains_at_k * dfs_at_k
        idcg_at_k = idg_at_k.sum(-1)
    
    return idcg, idcg_at_k

# %% ../../nbs/08_l2r.gradients.ipynb 10
def rank_loss2(preds, xb, sigma=0.5, lambrank=False, gain_fn=None, k=6):
    # In the following `ij` is essentially the set $I$
    sl = xb.shape[2]
    ij = torch.as_tensor(np.fromiter(itertools.combinations(np.arange(sl), 2), dtype=np.dtype((int,2))),
                                device=xb.device)#.expand(xb.shape[0], xb.shape[1], -1, -1)
    
    # Sort the tokens by the model prediction scores so that we can compute the set $I$ defined above:
    srtd_preds, srtd_idxs = preds[:, :, :,  0].sort(descending=True)
    
    srtd_ranks = srtd_preds.new_empty(srtd_preds.size())#srtd_idxs.argsort()
    srtd_ranks[:,:] = torch.arange(preds.shape[2])
    ri_rj = srtd_ranks[:, :, ij] # these are the ranks for token_i and token_j
    dfi_dfj = 1.0 / torch.log2(ri_rj + 2)
    dfi = dfi_dfj[:,:,:,0]
    dfj = dfi_dfj[:,:,:,1]
        
    srtd_relvs = torch.take_along_dim(xb[:, :, :, -1], srtd_idxs, dim=-1)
    pi_pj = srtd_preds[:, :, ij] # these are p_i and p_j 
    pi, pj = pi_pj[:, :, :, 0], pi_pj[:, :, :, 1]
    exp_ij = torch.exp(sigma * (pi - pj))
    si_sj = srtd_relvs[:, :, ij] # these are the relevance scores for token_i and token_j
    si, sj= si_sj[:, :, :, 0], si_sj[:, :, :, 1]
    gain_i, gain_j = ( torch.pow(2.0, si), torch.pow(2.0, sj) ) if gain_fn == 'exp' else ( torch.pow(si, 3.0), torch.pow(sj, 3.0) ) # cubic
    signs = torch.sign(si - sj)
    delta_dcg = torch.abs((gain_i - gain_j) * (dfi - dfj))
    idcg, idcg_at_k = _idcg(xb, k=k, gain_fn=gain_fn)
    delta_ndcg_at_k = delta_dcg / idcg_at_k.unsqueeze(-1)
    
    lambda_ij = sigma * (  0.5 * (1 - signs) -  1/(1 + exp_ij) )
    if lambrank: lambda_ij *= delta_ndcg_at_k # use this for Lambda-Rank
    
    sumer = _summation(sl, ij)
    idxr, signs = sumer[:, :, 0], sumer[:, :, -1]
    # Now we can compute $\lambda_i$ from eq: 4,
    lambda_i = (lambda_ij[:, :, idxr] * signs).sum(dim=-1)
    
    return srtd_preds, lambda_i

# %% ../../nbs/08_l2r.gradients.ipynb 11
def rank_loss3(preds, xb, sigma=0.5, lambrank=False, gain_fn=None, k=6):
    with torch.no_grad():
        # pdb.set_trace()
        x = xb[:, :, :, -1, None]
        x_t = xb[:, :, :, -1, None].transpose(-1,-2)
        preds_t = preds.transpose(-1,-2)
        preds_rank = preds[:, :, :, 0].argsort(dim=-1, descending=True).argsort(dim=-1).unsqueeze(-1)
        preds_rank_t = preds_rank.transpose(-1,-2)
        
        exp_ij= 1.0 + torch.exp(sigma* (preds - preds_t))
        rel_diff = x - x_t
        gain_diff = torch.pow(2.0, x) - torch.pow(2.0, x_t) if gain_fn == 'exp' else torch.pow(x, 3.0) - torch.pow(x_t, 3.0)
        decay_diff = 1.0/torch.log2(preds_rank + 2.0) - 1.0/torch.log2(preds_rank_t  + 2.0)
        idcg, idcg_at_k = _idcg(xb, k=k, gain_fn=gain_fn)
        idcg_at_k = idcg_at_k[..., None, None]
        # pdb.set_trace()
        delta_ndcg_at_k = torch.abs(gain_diff * decay_diff * 1/idcg_at_k)
        pos_pairs = (rel_diff > 0).float()
        neg_pairs = (rel_diff < 0).float()
        S_ij = pos_pairs - neg_pairs
        lambda_update = sigma * (  0.5 * (1 - S_ij) -  1/exp_ij )
        if lambrank: lambda_update *= delta_ndcg_at_k 
        lambda_update = lambda_update.sum(dim=-1, keepdim=True)
        # free memory
        del preds_t, preds_rank, preds_rank_t, exp_ij, rel_diff, gain_diff, decay_diff, idcg, idcg_at_k, delta_ndcg_at_k, pos_pairs, neg_pairs, S_ij
        import gc; gc.collect(); torch.cuda.empty_cache()
    return preds, lambda_update

# %% ../../nbs/08_l2r.gradients.ipynb 13
def loss_fn(preds, xb, sigma=0.5):
    
    srtd_relvs, srtd_idxs = xb[:, :, :, -1].sort(descending=True)
    srtd_preds = torch.take_along_dim(preds[:,:,:,0], srtd_idxs, dim=-1)

    sl = torch.arange(xb.shape[2], device=xb.device)
    ij = torch.cartesian_prod(sl, sl)
    idxs, = torch.nonzero(ij[:, 0] < ij[:, 1], as_tuple=True)
    ij = ij[idxs]
    
    si_sj = srtd_relvs[:, :, ij] # these are the relevance scores for token_i and token_j
    si, sj= si_sj[:, :, :, 0], si_sj[:, :, :, 1]
    signs = torch.sign(si - sj)
    pi_pj = srtd_preds[:, :, ij]
    pi, pj = pi_pj[:,:,:,0], pi_pj[:,:,:,1]
    exp_ij = torch.exp(-sigma*(pi -pj))
    exp_ij[exp_ij==torch.inf] = tensor(1e6)
    C = ( 0.5*(1 - signs)*sigma*(pi -pj) + torch.log(1 + exp_ij) ) #shape (64, 2234, 64)
    # C = C.sum(dim=-1) # shape (64, 2234)
    C = C.mean(dim=-1)
    return C#.mean()

def loss_fn2(preds, xb, sigma=.5):
    "Computes average pairwise cross-entropy loss"
    sl = xb.shape[2]
    rel_diff = xb[:, :, :, -1, None] - xb[:, :, :, -1, None].transpose(-1, -2)
    pos_pairs = (rel_diff > 0).float()
    neg_pairs = (rel_diff < 0).float()
    S_ij = pos_pairs - neg_pairs
    preds_diff = preds - preds.transpose(-1, -2)
    C = .5 * (1 - S_ij) * sigma * preds_diff - F.logsigmoid(sigma * preds_diff)
    C = torch.triu(C, diagonal=1) # to take each pair only once
    C = C.sum((-1,-2)) / (C.new_ones(C.shape[-2:]).triu(diagonal=1).sum())
    return C
