from rlzoo.algorithms.dppo_penalty.dppo_penalty import DPPO_PENALTY
from rlzoo.algorithms.dppo_clip.dppo_clip import DPPO_CLIP


def DPPO(**alg_params):
    method = alg_params['method']
    if method == 'penalty':
        del alg_params['epsilon']
        algo = DPPO_PENALTY
    elif method == 'clip':
        del alg_params['kl_target']
        del alg_params['lam']
        algo = DPPO_CLIP
    else:
        raise ValueError('Method input error. Method can only be penalty or clip')
    del alg_params['method']
    return algo(**alg_params)
