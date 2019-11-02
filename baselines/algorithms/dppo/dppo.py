from algorithms.dppo_penalty.dppo_penalty import DPPO_PENALTY
from algorithms.dppo_clip.dppo_clip import DPPO_CLIP


def DPPO(method, **alg_params):
    if method == 'penalty':
        del alg_params['epsilon']
        algo = DPPO_PENALTY
    elif method == 'clip':
        del alg_params['kl_target']
        del alg_params['lam']
        algo = DPPO_CLIP
    else:
        raise ValueError('Method input error. Method can only be penalty or clip')

    return algo(**alg_params)
