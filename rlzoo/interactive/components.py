from __future__ import print_function
import copy
from collections import OrderedDict

from ipywidgets import Layout
from ipywidgets import GridspecLayout

from IPython.display import clear_output
from IPython.core.interactiveshell import InteractiveShell
from gym import spaces

from rlzoo.common.env_list import all_env_list
from rlzoo.common.utils import *
from rlzoo.interactive.common import *

all_env_list = OrderedDict(sorted(all_env_list.items()))


class EnvironmentSelector(widgets.VBox):
    def __init__(self):
        env_list = all_env_list
        # al = list(env_list.keys())
        al = ['atari', 'classic_control', 'box2d', 'mujoco', 'robotics', 'dm_control', 'rlbench']
        description = 'Environment Selector'
        caption = widgets.HTML(value="<font size=4><B>" + description + "</B>")

        text_0 = widgets.Label("Choose your environment")

        self.env_type = widgets.Dropdown(
            options=al,
            value=al[0],
            description='env type:',
            disabled=False,
        )

        self.env_name = widgets.Dropdown(
            options=env_list[al[0]],
            value=env_list[al[0]][0],
            description='env name:',
            disabled=False,
        )
        env_select_box = widgets.VBox([text_0, self.env_type, self.env_name])

        text_1 = widgets.Label(value="Environment settings")

        self.env_num = widgets.IntText(
            value=1,
            description='multi envs:',
            disabled=False,
            min=1,
            #             layout=Layout(width='150px')
        )

        self.env_state = widgets.Dropdown(
            options=['default'],
            value='default',
            description='state type:',
            disabled=False,
        )

        #         self.create_button = widgets.Button(
        #             description='Create!',
        #             disabled=False,
        #             tooltip='Create',
        #             icon='check'
        #         )

        #         multi_box = widgets.HBox([self.env_multi, self.env_num], layout=Layout(justify_content='flex-start'))
        env_setting_box = widgets.VBox([text_1, self.env_num, self.env_state])

        select_box = widgets.HBox([env_select_box, env_setting_box],
                                  layout=Layout(justify_content='Center'))

        #         self.frame = widgets.VBox([select_box, widgets.Box([self.create_button],
        #                                                            layout=Layout(justify_content='Center'))])
        #         self.frame = widgets.AppLayout(left_sidebar=select_box, center=info_border.frame)

        def env_type_change(change):
            d = env_list[self.env_type.value]
            self.env_name.options = d
            self.env_name.value = d[0]
            if self.env_type.value == 'rlbench':
                self.env_state.options = ['state', 'vision']
                self.env_state.value = 'state'
                self.env_num.value = 1
                self.env_num.disabled = True
            else:
                self.env_state.options = ['default']
                self.env_state.value = 'default'
                self.env_num.disabled = False

        #         def create_env(c):  # todo the program will be blocked if rlbench env is created here
        #             if self.env_type.value == 'rlbench':
        #                 print(self.env_name.value, self.env_type.value, self.env_num.value, self.env_state.value)
        #                 self._env = build_env(self.env_name.value, self.env_type.value,
        #                                      nenv=self.env_num.value, state_type=self.env_state.value)
        #             self._env = build_env(self.env_name.value, self.env_type.value, nenv=self.env_num.value)
        #             print('Environment created successfully!')

        def change_nenv(c):
            if self.env_num.value < 1:
                self.env_num.value = 1

        self.env_num.observe(change_nenv, names='value')
        self.env_type.observe(env_type_change, names='value')

        #         self.create_button.on_click(create_env)

        super().__init__([caption, select_box], layout=widgets.Layout(align_items='center', ))

    @property
    def value(self):
        return {'env_id': self.env_name.value,
                'env_type': self.env_type.value,
                'nenv': self.env_num.value,
                'state_type': self.env_state.value}


#     @property
#     def env(self):
#         return self._env

class SpaceInfoViewer(widgets.Box):
    def __init__(self, sp):
        assert isinstance(sp, spaces.Space)
        if isinstance(sp, spaces.Dict):
            it = list(sp.spaces.items())
            info = GridspecLayout(len(it), 2)
            for i, v in enumerate(it):
                info[i, 0], info[i, 1] = widgets.Label(v[0]), widgets.Label(str(v[1]))
        else:
            info = widgets.Label(str(sp))
        super().__init__([info])


class EnvInfoViewer(widgets.VBox):
    def __init__(self, env):
        if isinstance(env, list):
            env = env[0]
        env_obs = SpaceInfoViewer(env.observation_space)
        env_act = SpaceInfoViewer(env.action_space)
        tips = None
        if isinstance(env.action_space, gym.spaces.Discrete):
            tips = 'The action space is discrete.'
        elif isinstance(env.action_space, gym.spaces.Box):
            tips = 'The action space is continuous.'

        description = 'Environment Information'
        caption = widgets.HTML(value="<font size=4><B>" + description + "</B>")

        a00, a01 = widgets.Label('Environment name:'), widgets.Label(env.spec.id)
        a10, a11 = widgets.Label('Observation space:'), env_obs
        a20, a21 = widgets.Label('Action space:'), env_act

        if tips is None:
            # use GirdBox instead of GridspecLayout to ensure each row has a different height
            info = widgets.GridBox(children=[a00, a01, a10, a11, a20, a21],
                                   layout=Layout(grid_template_areas="""
                                               "a00 a01"
                                               "a10 a11"
                                               "a20 a21"
                                               """))
        else:
            t0 = widgets.Label('Tips:')
            t1 = widgets.Label(tips)
            info = widgets.GridBox(children=[a00, a01, a10, a11, a20, a21, t0, t1],
                                   layout=Layout(grid_template_areas="""
                                               "a00 a01"
                                               "a10 a11"
                                               "a20 a21"
                                               "t0 t1"
                                               """))

        super().__init__([caption, info], layout=widgets.Layout(align_items='center', ))


all_alg_list = ['A3C', 'AC', 'DDPG', 'DPPO', 'DQN', 'PG', 'PPO', 'SAC', 'TD3', 'TRPO']
all_alg_dict = {'discrete_action_space': ['AC', 'DQN', 'PG', 'PPO', 'TRPO'],
                'continuous_action_space': ['AC', 'DDPG', 'PG', 'PPO', 'SAC', 'TD3', 'TRPO'],
                'multi_env': ['A3C', 'DPPO']
                }


class AlgorithmSelector(widgets.VBox):
    def __init__(self, env):
        description = 'Algorithm Selector'
        caption = widgets.HTML(value="<font size=4><B>" + description + "</B>")
        info = 'Supported algorithms are shown below'
        if isinstance(env, list):
            #             info = 'Distributed algorithms are shown below'
            table = all_alg_dict['multi_env']
            self.env_id = env[0].spec.id
        elif isinstance(env.action_space, gym.spaces.Discrete):
            #             info = 'Algorithms which support discrete action space are shown below'
            table = all_alg_dict['discrete_action_space']
            self.env_id = env.spec.id
        elif isinstance(env.action_space, gym.spaces.Box):
            #             info = 'Algorithms which support continuous action space are shown below'
            table = all_alg_dict['continuous_action_space']
            self.env_id = env.spec.id
        else:
            raise ValueError('Unsupported environment')

        self.algo_name = widgets.Dropdown(
            options=table,
            value=table[0],
            description='Algorithms:',
            disabled=False,
        )

        super().__init__([caption, widgets.Label(info), self.algo_name],
                         layout=widgets.Layout(align_items='center', ))

    @property
    def value(self):
        return self.algo_name.value


def TransInput(value):
    if isinstance(value, bool):
        return widgets.Checkbox(value=value, description='', disabled=False, indent=False)
    elif isinstance(value, int) or isinstance(value, float) \
            or isinstance(value, np.int16) or isinstance(value, np.float16) \
            or isinstance(value, np.int32) or isinstance(value, np.float32) \
            or isinstance(value, np.int64) or isinstance(value, np.float64) \
            or isinstance(value, np.float128):
        return NumInput(value)
    else:
        return widgets.Label(value)


class AlgoInfoViewer(widgets.VBox):
    def __init__(self, alg_selector, org_alg_params, org_learn_params):
        alg_params, learn_params = copy.deepcopy(org_alg_params), copy.deepcopy(org_learn_params)

        # ---------------- alg_params --------------- #
        description = 'Algorithm Parameters'
        alg_caption = widgets.HTML(value="<font size=4><B>" + description + "</B>")
        net_label = widgets.Label('Network information:')
        show_net = lambda net: widgets.VBox([widgets.Label(str(layer)) for layer in net.all_layers])

        n = np.ndim(alg_params['net_list'])
        if n == 1:
            model_net = alg_params['net_list']
        elif n == 2:
            model_net = alg_params['net_list'][0]

        net_info = widgets.VBox([widgets.VBox([widgets.Label(str(net.__class__.__name__)),
                                               show_net(net), ],
                                              layout=widgets.Layout(border=border_list[2],
                                                                    align_items='center',
                                                                    align_content='center'
                                                                    )
                                              ) for net in model_net])
        self._net_list = alg_params['net_list']
        del alg_params['net_list']

        opt_label = widgets.Label('Optimizer information:')

        def show_params(params):
            params = copy.deepcopy(params)
            n = len(params)
            frame = widgets.GridspecLayout(n, 2, layout=widgets.Layout(border=border_list[2], ))
            show_info = lambda k: [widgets.Label(str(k)), widgets.Label(str(params[k]))]
            frame[0, 0], frame[0, 1] = show_info('name')
            frame[1, 0], frame[1, 1] = show_info('learning_rate')
            del params['name']
            del params['learning_rate']
            for i, k in enumerate(sorted(params.keys())):
                if k != 'name' and k != 'learning_rate':
                    frame[2 + i, 0], frame[2 + i, 1] = show_info(k)
            return frame

        opt_info = widgets.VBox([show_params(n.get_config()) for n in alg_params['optimizers_list']])
        self._optimizers_list = alg_params['optimizers_list']
        del alg_params['optimizers_list']

        stu_frame = widgets.GridBox(children=[net_label, net_info, opt_label, opt_info],
                                    layout=Layout(grid_template_areas="""
                                           "net_label net_info"
                                           "opt_label opt_info"
                                           """))

        alg_sel_dict = dict()
        sk = sorted(alg_params.keys())
        n = len(sk) + 1
        alg_param_sel = widgets.GridspecLayout(n, 2)
        b = 0
        if 'method' in sk:
            module = widgets.RadioButtons(options=['penalty', 'clip'], disabled=False)
            alg_param_sel[0, 0], alg_param_sel[0, 1] = widgets.Label('method'), module
            alg_sel_dict['method'] = module
            sk.remove('method')
            b += 1

        for i, k in enumerate(sk):
            module = TransInput(alg_params[k])
            alg_sel_dict[k] = module
            if k == 'dueling':
                module.disabled = True
            alg_param_sel[i + b, 0], alg_param_sel[i + b, 1] = widgets.Label(k), module

        alg_param_box = widgets.VBox([alg_caption, stu_frame, alg_param_sel], )
        name = alg_selector.value + '-' + alg_selector.env_id
        path = os.path.join('.', 'model', name)
        alg_param_sel[n - 1, 0] = widgets.Label('model save path')
        alg_param_sel[n - 1, 1] = widgets.Label(path)

        self.alg_sel_dict = alg_sel_dict
        # ================== alg_params ================= #

        # ----------------- learn_params ---------------- #
        description = 'Learn Parameters'
        learn_caption = widgets.HTML(value="<font size=4><B>" + description + "</B>")

        learn_sel_dict = dict()
        sk = sorted(learn_params.keys())

        n = len(sk)
        if 'mode' not in sk: n += 1
        if 'render' not in sk: n += 1
        learn_param_sel = widgets.GridspecLayout(n, 2)

        module = widgets.RadioButtons(options=['train', 'test'], disabled=False)
        learn_param_sel[0, 0], learn_param_sel[0, 1] = widgets.Label('mode'), module
        learn_sel_dict['mode'] = module
        try:
            sk.remove('mode')
        except:
            pass

        module = widgets.Checkbox(value=False, description='', disabled=False, indent=False)
        learn_param_sel[1, 0], learn_param_sel[1, 1] = widgets.Label('render'), module
        learn_sel_dict['render'] = module
        try:
            sk.remove('render')
        except:
            pass

        for i, k in enumerate(sk):
            module = TransInput(learn_params[k])
            learn_sel_dict[k] = module
            learn_param_sel[i + 2, 0], learn_param_sel[i + 2, 1] = widgets.Label(k), module
        learn_param_box = widgets.VBox([learn_caption, learn_param_sel],
                                       #                                      layout=Layout(align_items='center',)
                                       )
        self.learn_sel_dict = learn_sel_dict
        # ================= learn_params ================ #

        b = widgets.Output(layout=widgets.Layout(border='solid'))

        self.smooth_factor_slider = widgets.FloatSlider(
            value=0.8,
            min=0,
            max=1,
            step=0.01,
            description='learning curve smooth factor',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.2f',
            style={'description_width': 'initial'},
        )
        super().__init__([alg_param_box, b, learn_param_box, b, self.smooth_factor_slider])

    @property
    def alg_params(self):
        result = {'net_list': self._net_list, 'optimizers_list': self._optimizers_list}
        for k in self.alg_sel_dict.keys():
            result[k] = self.alg_sel_dict[k].value
        return result

    @property
    def smooth_factor(self):
        return self.smooth_factor_slider.value

    @property
    def learn_params(self):
        result = dict()
        for k in self.learn_sel_dict.keys():
            result[k] = self.learn_sel_dict[k].value
        return result


class RevOutput(widgets.Output):
    def _append_stream_output(self, text, stream_name):
        """Append a stream output."""
        self.outputs = (
                           {'output_type': 'stream', 'name': stream_name, 'text': text},
                       ) + self.outputs

    def append_display_data(self, display_object):
        """Append a display object as an output.

        Parameters
        ----------
        display_object : IPython.core.display.DisplayObject
            The object to display (e.g., an instance of
            `IPython.display.Markdown` or `IPython.display.Image`).
        """
        fmt = InteractiveShell.instance().display_formatter.format
        data, metadata = fmt(display_object)
        self.outputs = (
                           {
                               'output_type': 'display_data',
                               'data': data,
                               'metadata': metadata
                           },
                       ) + self.outputs


class OutputMonitor(widgets.HBox):
    def __init__(self, learn_params, smooth_factor):
        max_num = learn_params['train_episodes'] if learn_params['mode'] == 'train' else learn_params['test_episodes']
        self.progress = widgets.FloatProgress(value=0.0, min=0.0, max=max_num, description='Progress')

        self.plot_out = widgets.Output(layout=widgets.Layout(width='350px',
                                                             height='250px', ))
        self.smooth_factor = smooth_factor
        # self.smooth_factor = widgets.FloatSlider(
        #     value=self.sf,
        #     min=0,
        #     max=1,
        #     step=0.01,
        #     description='smooth factor',
        #     disabled=False,
        #     continuous_update=False,
        #     orientation='horizontal',
        #     readout=True,
        #     readout_format='.2f'
        # )

        # def link(c):
        #     self.sf = self.smooth_factor.value

        # self.smooth_factor.observe(link, 'value')
        # plot_out = widgets.VBox([widgets.Label('Learning curve'), self.plot_out, self.smooth_factor])
        plot_out = widgets.VBox([widgets.Label('Learning curve'), self.plot_out])

        self.print_out = RevOutput(layout=widgets.Layout(overflow='scroll',
                                                         width='60%',
                                                         height='300px',
                                                         # display='flex',
                                                         # positioning='bottom',
                                                         border='1px solid black',
                                                         ))
        self.plot_func([])
        super().__init__([widgets.VBox([plot_out, self.progress]), self.print_out])

    def plot_func(self, datas):
        # datas = signal.lfilter([1 - self.smooth_factor], [1, -self.smooth_factor], datas, axis=0)
        if datas:
            disD = [datas[0]]
            for d in datas[1:]:
                disD.append(disD[-1] * self.smooth_factor + d * (1 - self.smooth_factor))
        else:
            disD = datas
        with self.plot_out:
            self.progress.value = len(disD)
            plt.plot(disD)
            clear_output(wait=True)
            plt.show()
