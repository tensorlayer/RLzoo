from rlzoo.algorithms.ppo_penalty.ppo_penalty import PPO_PENALTY
from rlzoo.algorithms.ppo_clip.ppo_clip import PPO_CLIP


def PPO(**alg_params):
    method = alg_params['method']
    if method == 'penalty':
        del alg_params['epsilon']
        algo = PPO_PENALTY
    elif method == 'clip':
        del alg_params['kl_target']
        del alg_params['lam']
        algo = PPO_CLIP
    else:
        raise ValueError('Method input error. Method can only be penalty or clip')
    del alg_params['method']
    return algo(**alg_params)
