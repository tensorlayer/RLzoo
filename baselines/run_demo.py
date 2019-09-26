"""This is a demo for showing the pipeline of RLzoo"""
import numpy as np
import tensorlayer as tl

# common imports for envs
from baselines.common.env_wrappers import build_env
# common imports for nets
from baselines.common import backbones, heads, distributions
# specific imports for algorithms
from baselines.common import buffers


# ----------- User Input -----------
user_inputs = {  # this maybe kwargs of the entry
    'env_type': 'atari',
    'env_id': 'PongNoFrameskip-v4',
    'dueling': True,
    'double': True,
    'prioritized_replay': False
}

# ----------- Generate desired input for algorithms -----------
"""First, load default parameters from the corresponding default.py
for clearly tracking modifications
"""
parameters = {
    'distribution': distributions.Categorical,  # this will be a option for ppo
    'double': False,
    'dueling': False,
    'prioritized_replay': True,
}

"""Second, merging user inputs and default parameters"""
parameters.update(**user_inputs)

"""Finally, initialize networks and modules that can be done before learning"""
env = build_env(parameters['env_id'], vectorized=parameters['vectorized'])
if parameters['env_type'] == 'atari':
    if parameters['qnetwork'] is None:
        def flatten_dims(shapes):  # will be moved to common
            dim = 1
            for s in shapes:
                dim *= s
            return dim

        class Network(tl.models.Model):  # will be moved to common
            def __init__(self, backbone, head):
                super(Network, self).__init__()
                self._backbone = backbone
                self._head = head

            def forward(self, ni):
                return self._head(
                    tl.layers.flatten_reshape(self._backbone(ni)))

        cnn_backbone = backbones.cnn(env.observation_space.shape)
        mlp_head = heads.mlp(flatten_dims(cnn_backbone.outputs[0].shape))
        parameters['qnetwork'] = Network(cnn_backbone, mlp_head)
        parameters['qnetwork'].train()
    else:
        pass  # already input by user
else:  # support other env types like classic control
    raise NotImplementedError
if parameters['prioritized_replay']:
    raise NotImplementedError
else:
    replay_buffer = buffers.ReplayBuffer


# ----------- Execute main logic of learning DQN -----------
"""Note that for efficiently training, we need to split each step in algorithm
and wrap forward operations by @tf.function. This part should only contain env
interaction with different ops according to users' inputs.
"""
env.reset()
for timestep in range(parameters['total_timesteps']):
    # sampling with epsilon-greedy
    pass

    # feed to replay buffer
    pass

    # learning
    if timestep > parameters['warmstart']:
        # sampling from replay buffer
        pass

        # estimate target q
        if parameters['double']:
            pass  # run double estimation OP
        else:
            pass  # run general estimation OP

        # calculate loss and backpropagate
        if not parameters['prioritized_replay']:
            weights = np.ones  # equal weights
        if parameters['c51']:
            pass  # run kl loss OP
        elif parameters['qr']:
            pass  # run quantile loss op
        else:
            pass  # run weighted huber loss training OP
        pass  # backpropagate

        if parameters['prioritized_replay']:
            pass  # update td-errors in per

    # save model, print log, etc.
    """Note that every file operation must provide option to users"""
    """True logs will be recorded in wrappers.Monitor, just read from it"""
    if parameters['***']:
        pass


# ----------- Execute main logic of evaluating DQN -----------
pass
