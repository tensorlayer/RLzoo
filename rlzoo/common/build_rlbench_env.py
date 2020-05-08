import sys
from collections import OrderedDict

import numpy as np
from gym import spaces

from pyrep.const import RenderMode
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor
from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import *


# Don't forget to add: export PYTHONPATH=PATH_TO_YOUR_LOCAL_RLBENCH_REPO

# list of state types
state_types = ['left_shoulder_rgb',
               'left_shoulder_depth',
               'left_shoulder_mask',
               'right_shoulder_rgb',
               'right_shoulder_depth',
               'right_shoulder_mask',
               'wrist_rgb',
               'wrist_depth',
               'wrist_mask',
               'joint_velocities',
               'joint_velocities_noise',
               'joint_positions',
               'joint_positions_noise',
               'joint_forces',
               'joint_forces_noise',
               'gripper_pose',
               'gripper_touch_forces',
               'task_low_dim_state']


class RLBenchEnv():
    """ make RLBench env to have same interfaces as openai.gym """

    def __init__(self, task_name: str, state_type: list = 'state', ):
        # render_mode=None):
        """
        create RL Bench environment
        :param task_name: task names can be found in rlbench.tasks
        :param state_type: state or vision or a sub list of state_types list like ['left_shoulder_rgb']
        """
        if state_type == 'state' or state_type == 'vision' or isinstance(state_type, list):
            self._state_type = state_type
        else:
            raise ValueError('State type value error, your value is {}'.format(state_type))
        # self._render_mode = render_mode
        self._render_mode = None
        obs_config = ObservationConfig()
        obs_config.set_all(True)
        action_mode = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY)
        self.env = Environment(
            action_mode, obs_config=obs_config, headless=True)
        self.env.launch()
        try:
            self.task = self.env.get_task(getattr(sys.modules[__name__], task_name))
        except:
            raise NotImplementedError

        _, obs = self.task.reset()
        self.spec = Spec(task_name)

        if self._state_type == 'state':
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=obs.get_low_dim_data().shape)
        elif self._state_type == 'vision':
            space_dict = OrderedDict()
            space_dict["state"] = spaces.Box(
                low=-np.inf, high=np.inf, shape=obs.get_low_dim_data().shape)
            for i in ["left_shoulder_rgb", "right_shoulder_rgb", "wrist_rgb", "front_rgb"]:
                space_dict[i] = spaces.Box(
                    low=0, high=1, shape=getattr(obs, i).shape)
            self.observation_space = spaces.Dict(space_dict)
        else:
            space_dict = OrderedDict()
            for name in self._state_type:
                if name.split('_')[-1] in ('rgb', 'depth', 'mask'):
                    space_dict[name] = spaces.Box(
                        low=0, high=1, shape=getattr(obs, name).shape)
                else:
                    space_dict[name] = spaces.Box(
                        low=-np.inf, high=np.inf,
                        shape=getattr(obs, name).shape)
                self.observation_space = spaces.Dict(space_dict)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.env.action_size,), dtype=np.float32)

        # if render_mode is not None:
        #     # Add the camera to the scene
        #     cam_placeholder = Dummy('cam_cinematic_placeholder')
        #     self._gym_cam = VisionSensor.create([640, 360])
        #     self._gym_cam.set_pose(cam_placeholder.get_pose())
        #     if render_mode == 'human':
        #         self._gym_cam.set_render_mode(RenderMode.OPENGL3_WINDOWED)
        #     else:
        #         self._gym_cam.set_render_mode(RenderMode.OPENGL3)

    def _extract_obs(self, obs):
        if self._state_type == 'state':
            return np.array(obs.get_low_dim_data(), np.float32)
        elif self._state_type == 'vision':
            return np.array([np.array(obs.get_low_dim_data(), np.float32),
                    np.array(obs.left_shoulder_rgb, np.float32),
                    np.array(obs.right_shoulder_rgb, np.float32),
                    np.array(obs.wrist_rgb, np.float32),
                    np.array(obs.front_rgb, np.float32), ])
        else:
            result = []
            for name in self._state_type:
                result.append(np.array(getattr(obs, name), np.float32))
            return np.array(result)

    def seed(self, seed_value):
        # set seed as in openai.gym env
        pass

    def render(self, mode='human'):
        # todo render available at any time
        if self._render_mode is None:
            self._render_mode = mode
            # Add the camera to the scene
            cam_placeholder = Dummy('cam_cinematic_placeholder')
            self._gym_cam = VisionSensor.create([640, 360])
            self._gym_cam.set_pose(cam_placeholder.get_pose())
            if mode == 'human':
                self._gym_cam.set_render_mode(RenderMode.OPENGL3_WINDOWED)
            else:
                self._gym_cam.set_render_mode(RenderMode.OPENGL3)

        if mode != self._render_mode:
            raise ValueError(
                'The render mode must match the render mode selected in the '
                'constructor. \nI.e. if you want "human" render mode, then '
                'create the env by calling: '
                'gym.make("reach_target-state-v0", render_mode="human").\n'
                'You passed in mode %s, but expected %s.' % (
                    mode, self._render_mode))
        if mode == 'rgb_array':
            return self._gym_cam.capture_rgb()

    def reset(self):
        descriptions, obs = self.task.reset()
        return self._extract_obs(obs)

    def step(self, action):
        obs, reward, terminate = self.task.step(action)
        return self._extract_obs(obs), reward, terminate, None

    def close(self):
        self.env.shutdown()


class Spec():
    """ a fake spec """

    def __init__(self, id_name):
        self.id = id_name
