import decimal

import ipywidgets as widgets
import numpy as np

border_list = [None, 'hidden', 'dotted', 'dashed', 'solid', 'double',
               'groove', 'ridge', 'inset', 'outset', 'inherit']


class NumInput(widgets.HBox):

    def __init__(self, init_value, step=None, range_min=None, range_max=None):
        self.range = [range_min, range_max]
        range_min = 0 if range_min is None else range_min
        range_max = init_value * 2 if range_max is None else range_max
        self.range_size = max([range_max - init_value, init_value - range_min])
        if step is None:
            fs = decimal.Decimal(str(init_value)).as_tuple().exponent
            self.decimals = -fs
            step = np.round(np.power(0.1, self.decimals), self.decimals)
        else:
            fs = decimal.Decimal(str(step)).as_tuple().exponent
            fv = decimal.Decimal(str(init_value)).as_tuple().exponent
            self.decimals = -min(fs, fv)

        self.step = step

        self.slider = widgets.FloatSlider(
            value=init_value,
            min=range_min,
            max=range_max,
            step=step,
            description='Slider input:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.' + str(self.decimals) + 'f'
        )

        self.text = widgets.FloatText(
            value=self.slider.value,
            description='Manual input:',
            disabled=False
        )

        def __extend_max(change):
            num_new = np.around(change['new'], decimals=self.decimals)
            num_old = change['old']
            if num_new > num_old:
                if num_new - num_old > (self.slider.max - num_old) / 2:
                    self.range_size *= 2
                else:
                    self.range_size *= 0.5
            else:
                if num_old - num_new > (num_old - self.slider.min) / 2:
                    self.range_size *= 2
                else:
                    self.range_size *= 0.5

            if self.range_size < self.step * 10:
                self.range_size = self.step * 10

            self.slider.min = num_new - self.range_size if self.range[0] is None else self.range[0]
            self.slider.max = num_new + self.range_size if self.range[1] is None else self.range[1]
            self.slider.value = num_new
            self.text.value = num_new

        self.slider.observe(__extend_max, names='value')
        self.text.observe(__extend_max, names='value')
        box_layout = widgets.Layout(display='flex',
                                    align_items='stretch',
                                    justify_content='center', )
        #         self.frame = widgets.HBox([self.slider, self.text], layout=box_layout)
        super().__init__([self.slider, self.text], layout=box_layout)
        self._int_type = False
        if (isinstance(init_value, int) or isinstance(init_value, np.int16) \
            or isinstance(init_value, np.int32) or isinstance(init_value, np.int64)) \
                and step % 1 == 0:
            self._int_type = True

    @property
    def value(self):
        result = self.slider.value
        if self._int_type:
            result = int(result)
        return result


class Border:
    def __init__(self, element_list, description=None, size=5, style=0):
        if not isinstance(element_list, list):
            element_list = [element_list]

        box_layout = widgets.Layout(display='flex',
                                    flex_flow='column',
                                    align_items='flex-start',
                                    align_content='flex-start',
                                    #                                     justify_content='center',
                                    justify_content='space-around',
                                    border=border_list[2]
                                    )
        frame = widgets.Box(children=element_list, layout=box_layout)

        if description is not None:
            caption = widgets.HTML(value="<font size="+str(size)+"><B>"+description+"</B>")
            children = [caption, frame]
        else:
            children = [frame]

        box_layout = widgets.Layout(display='flex',
                                    flex_flow='column',
                                    align_items='center',
                                    justify_content='center',
                                    border=border_list[style], )
        self.frame = widgets.Box(children=children, layout=box_layout)


class InfoDisplay:
    def __init__(self, description, detail):
        label = widgets.Label(description)
        self.data = widgets.Label(detail)
        self.frame = widgets.HBox([label, self.data], layout=widgets.Layout(justify_content='flex-start', ))
#                                                                           border=border_list[2]))
