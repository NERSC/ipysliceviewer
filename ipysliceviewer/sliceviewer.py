# Standard lib
import os
# Third party
from PIL import Image
from bqplot import (
    LinearScale, Figure, PanZoom,
    Toolbar, Image as BQImage,
)
from ipywidgets import (
    VBox, HBox, Layout, GridBox,
    Button, ToggleButton, ToggleButtons,
    IntSlider, FloatRangeSlider,
    Play, Output, jslink,
    Image as IPyImage,
)
import numpy as np
from ipypathchooser import PathChooser
# Local
from .datasets import VolumeDataset, FolderDataset
from .utils import (
    PIL_to_numpy, numpy_to_PIL,
    PIL_to_bytes, bytes_to_PIL,
    to_grayscale, to_rgb,
    draw_checkerboard_canvas,
    get_offsets_and_resize_for_canvas,
)

# During development, you'll want to decorate every call that observes a traitlet in
# `@output.capture` or else these methods can fail silently.
# When captured, they will display in `output` when it is rendered by evaluating it in a cell.
output = Output(layout={
    'border': '2px dotted lightgray',
    'padding': '2px',
})

class SliceViewer(VBox):

    def __init__(
        self,
        volume=None,
        default_directory=os.getcwd(),
        title='',
        enhancement_steps=1000,
        **kwargs
    ):
        def on_chosen_path_change(old_path, new_path):
            self.dataset = FolderDataset(new_path)
            # TODO: If the path doesn't contain images, display a warning

        # A widget for changing the image folder
        self.pathchooser = PathChooser(
            chosen_path_desc='Image folder:',
            default_directory=default_directory,
            on_chosen_path_change=on_chosen_path_change,
        )
        self.pathchooser.layout.margin = '0 0 10px 0'

        # The number of increments of the min/max slider
        self.enhancement_steps = enhancement_steps

        self.scales = {
            'x': LinearScale(),
            'y': LinearScale(),
        }

        # The currently displayed image will be in bytes at `self.image_plot.image.value`
        self.image_plot = BQImage(
            image=IPyImage(),
            scales=self.scales,
        )

        self.figure = Figure(
            marks=[self.image_plot],
            padding_x=0,
            padding_y=0,
            animation_duration=1000,
            fig_margin={
                'top': 0,
                'right': 0,
                'bottom': 0,
                'left': 0,
            },
            layout=Layout(
                grid_area='figure',
                margin='0',
                width='320px',
                height='320px',
            ),
        )

        # Custom toolbar
        toolbar_width = '100%'
        toolbar_margin = '0px 0 2px 0'
        self.pan_zoom = PanZoom(
            scales={
                'x': [self.scales['x']],
                'y': [self.scales['y']],
            },
        )

        self.save_button = Button(
            description='Save Image',
            tooltip='Save Image',
            icon='save',
            layout=Layout(
                width=toolbar_width,
                # flex='1 1 auto',
                margin=toolbar_margin,
            ),
        )
        self.save_button.on_click(self.save_current_image)

        self.hide_button = Button(
            description='Hide Image',
            tooltip='Hide Image',
            icon='eye-slash',
            layout=Layout(
                width=toolbar_width,
                # flex='1 1 auto',
                margin=toolbar_margin,
            )
        )
        self.hide_button.on_click(self.hide_current_image)

        self.pan_zoom_toggle_button = ToggleButton(
            description='Pan / Zoom',
            tooltip='Pan/Zoom',
            icon='arrows',
            layout=Layout(
                width=toolbar_width,
                # flex='1 1 auto',
                margin=toolbar_margin,
            ),
        )
        self.pan_zoom_toggle_button.observe(self.on_pan_zoom_toggle, names='value')

        self.reset_pan_zoom_button = Button(
            description='Undo Zoom',
            tooltip='Reset pan/zoom',
            icon='refresh',
            layout=Layout(
                width=toolbar_width,
                # flex='1 1 auto',
                margin=toolbar_margin,
            ),
        )
        self.reset_pan_zoom_button.on_click(self.reset_pan_zoom)

        self.reset_enhancements_button = Button(
            description='Un-Enhance',
            tooltip='Reset enhancements',
            icon='ban',
            layout=Layout(
                width=toolbar_width,
                # flex='1 1 auto',
                margin=toolbar_margin,
            ),
        )
        self.reset_enhancements_button.on_click(self.reset_enhancements)

        self.mini_map = IPyImage(layout=Layout(
            grid_area='mini-map',
            margin='0',
        ))
        self.mini_map.width = 180
        self.mini_map.height = 180
        # PERFORMANCE CONCERN
        # Ideally instead of four observations, this would observe 'scales' on `self.pan_zoom`
        # However, it doesn't fire updates
        # Ref: https://github.com/bloomberg/bqplot/issues/800
        self.image_plot.scales['x'].observe(self.on_pan_zoom_change('x_min'), names='min')
        self.image_plot.scales['x'].observe(self.on_pan_zoom_change('x_max'), names='max')
        self.image_plot.scales['y'].observe(self.on_pan_zoom_change('y_min'), names='min')
        self.image_plot.scales['y'].observe(self.on_pan_zoom_change('y_max'), names='max')

        self.plane_toggle = ToggleButtons(
            options=['yz', 'xz', 'xy'],
            description='',
            disabled=False,
            button_style='',
            tooltips=['Step in x direction', 'Step in y direction', 'Step in z direction'],
            layout=Layout(
                width='200px',
                # flex='1 1 auto',
                margin='7px 0 auto auto',
            ),
        )
        self.plane_toggle.style.button_width = 'auto'
        self.plane_toggle.observe(self.on_plane_change, names='value')

        self.toolbar = VBox(
            children = [
                self.save_button,
                self.hide_button,
                self.pan_zoom_toggle_button,
                self.reset_pan_zoom_button,
                self.reset_enhancements_button,
            ],
            layout=Layout(
                grid_area='toolbar',
                margin='0',
            ),
        )

        # Image enhancements
        self.min_max_slider = FloatRangeSlider(
            value=[0, 255],
            min=0,
            max=255,
            step=255 / self.enhancement_steps,
            description='Min/Max:',
            orientation='horizontal',
            readout=True,
            readout_format='.1f',
            continuous_update=True,
            layout=Layout(
                grid_area='min-max-slider',
                margin='10px 0 10px -10px',
                width='100%',
            ),
        )
        self.min_max_slider.observe(self.on_min_max_change, names='value')

        self.index_slider = IntSlider(
            value=0,
            min=0,
            max=1,
            step=1,
            description='Index:',
            orientation='horizontal',
            readout=True,
            readout_format='d',
            continuous_update=True,
            layout=Layout(
                grid_area='index-slider',
                margin='8px -20px 10px -36px',
                width='100%',
            ),
        )
        self.index_slider.observe(self.on_image_index_change, names='value')

        # Animation
        self.play = Play(
            value=self.index_slider.value,
            min=self.index_slider.min,
            max=self.index_slider.max,
            step=self.index_slider.step,
        )
        jslink((self.play, 'value'), (self.index_slider, 'value'))
        # Keep 'max' in sync as well
        self.index_slider.observe(self.on_index_slider_max_change, names='max')

        self.bottom_bar = HBox(
            children=[
                self.play,
                self.index_slider,
                self.plane_toggle,
            ],
            layout=Layout(
               grid_area='bottom-bar',
               margin=f'10px -20px 0 0',
               # overflow='hidden',
            )
        )

        # Layout
        self.gridbox = GridBox(
            children=[
                self.figure,
                self.toolbar,
                self.mini_map,
                self.min_max_slider,
                self.bottom_bar,
            ],
        )
        # Initially hidden without data
        self.gridbox.layout.display = 'none'

        self._dataset = None
        if volume is not None:
            self.dataset = VolumeDataset(volume)
            # Hide pathchooser when using a volume
            self.pathchooser.layout.display = 'none'

        # Call VBox super class __init__
        super().__init__(
            children=[
                self.pathchooser,
                self.gridbox,
            ],
            layout=Layout(width='auto'),
            **kwargs,
        )

    @property
    def dataset(self):
        """
        Get the dataset that the SliceViewer is displaying.
        """
        return self._dataset

    @dataset.setter
    def dataset(self, dataset):
        """
        Set the dataset that the SliceViewer is displaying.
        """
        if dataset:
            # TODO: set initial pan_zoom_scales
            # image_as_array = dataset[0]
            # width, height = image_as_array.shape
            # self.get_initial_pan_zoom_scales()
            self.index_slider.max = len(dataset) - 1
            # self.play.max = self.index_slider.max
            # Could be simpler with 0 margins, but for now is written generically
            self.plane_toggle.disabled = not isinstance(dataset, VolumeDataset)
            self.gridbox.layout = Layout(
                width='auto',
                # height='500px',
                grid_gap='0px 10px',
                # grid_template_columns='auto auto auto',
                grid_template_columns=f'{self.figure_size[0]}px 180px',
                # grid_template_rows=f'134px {self.figure_size[1] - 110}px 52px 52px',
                grid_template_rows=f'140px 180px 36px 60px',
                grid_template_areas='''
                    "figure toolbar"
                    "figure mini-map"
                    "min-max-slider min-max-slider"
                    "bottom-bar bottom-bar"
                ''',
            )
            self.gridbox.layout.display = None
        else:
            self.gridbox.layout.display = 'none'
        self._dataset = dataset
        # Crucially, this also calls self.redraw
        self.reset_enhancements()

    @property
    def figure_size(self):
        """
        Get the figure layout width and height as integers.
        """
        width = int(self.figure.layout.width[:-2])
        height = int(self.figure.layout.height[:-2])
        return [width, height]

    @figure_size.setter
    def figure_size(self, size):
        """
        Set the figure layout width and height with the provided list.
        """
        width, height = size
        self.figure.layout.width = f'{width}px'
        self.figure.layout.height = f'{height}px'

    @property
    def current_image(self):
        """
        Get the current image from backing `self.dataset` according to `self.index_slider`.
        It should be a normalized numpy array.
        """
        return self.dataset[self.index_slider.value]

    @property
    def value_range(self):
        """
        Get the value ranges of the unormalized dataset.
        """
        low = getattr(self.dataset, 'min', 0)
        high = getattr(self.dataset, 'max', 255)
        return [low, high]

    @output.capture()
    def get_current_image_name(self):
        """
        Return the name of the current image selected according to `self.index_slider`.
        """
        if not self.dataset:
            return ''
        else:
            index = self.index_slider.value
            image_names = getattr(self.dataset, 'image_names', [])
            if image_names:
                return image_names[index]
            else:
                return f'sliceviewer-image-{index}.jpg'

    @output.capture()
    def get_current_scales(self):
        """
        Get the current image_plot scales in a plain dictionary.
        """
        scales = self.image_plot.scales
        plain_scales = {
            'x': [scales['x'].min, scales['x'].max],
            'y': [scales['y'].min, scales['y'].max],
        }
        # Coerce None to 0
        plain_scales['x'] = [x if x else 0 for x in plain_scales['x']]
        plain_scales['y'] = [y if y else 0 for y in plain_scales['y']]
        return plain_scales

    @output.capture()
    def redraw(self, image_as_array=None):
        """
        Redraw main image and mini-map. Defaults to enhanced current image.
        """
        # Main image
        if image_as_array is None:
            image_as_array = self.enhance_image(self.current_image)
        image = PIL_to_bytes(numpy_to_PIL(image_as_array))
        self.image_plot.image = IPyImage(value=image)
        # Mini-map
        self.redraw_mini_map(image_as_array=image_as_array)

    @output.capture()
    def redraw_mini_map(self, image_as_array=None, scales=None):
        """
        Redraw the mini-map. Defaults to enhanced current image.
        """
        if image_as_array is None:
            image_as_array = self.enhance_image(self.current_image)
        if scales is None:
            scales = self.get_current_scales()
        mini_map = self.draw_mini_map(image_as_array, scales)
        self.mini_map.value = PIL_to_bytes(numpy_to_PIL(mini_map))

    @output.capture()
    def on_image_index_change(self, change):
        """
        Load and display the new image.
        """
        enhanced = self.enhance_image(self.current_image)
        self.redraw(image_as_array=enhanced)

    @output.capture()
    def on_index_slider_max_change(self, change):
        """
        Sync play.max with index_slider.max and reset index slider.
        """
        self.play.max = change.new

    @output.capture()
    def get_initial_pan_zoom_scales(self):
        """
        Calculate the necessary pan_zoom scales to fill-in the main viewport
        when the input image is not square.
        """
        raise NotImplementedError

    @output.capture()
    def on_pan_zoom_toggle(self, change):
        """
        Update the `self.figure` interaction.
        """
        if change.new:
            self.figure.interaction = self.pan_zoom
        else:
            self.figure.interaction = None

    @output.capture()
    def reset_pan_zoom(self, button):
        """
        Reset figure/plot scales.
        """
        self.image_plot.scales['x'].min = None
        self.image_plot.scales['x'].max = None
        self.image_plot.scales['y'].min = None
        self.image_plot.scales['y'].max = None
        self.redraw_mini_map()

    @output.capture()
    def draw_mini_map(self, image_as_array, scales):
        """
        Draw a mini version of image_as_array with a rectangle indicating pan/zoom location.
        """
        # Commented code is preparation for non-square image
        # canvas_as_array = draw_checkerboard_canvas(
        #     height=int(self.mini_map.height),
        #     width=int(self.mini_map.width),
        # )
        # offsets, image_as_array = get_offsets_and_resize_for_canvas(
        #     image_as_array,
        #     canvas_as_array,
        # )
        shape = image_as_array.shape
        # Convert grayscale to RGB
        mini_map = to_rgb(image_as_array)
        # Draw a red square indicating the zoom location
        xs = [int(x * shape[1]) for x in scales['x']]
        ys = [int((1.0 - y) * shape[0]) for y in scales['y']]
        # Make sure values are in range
        def clamp(values, low, high):
            temp = [v if v > low else low for v in values]
            return [v if v < high else high for v in temp]
        # This will give a border width of 2 pixels
        xs.extend([x + 1 for x in xs])
        ys.extend([y - 1 for y in ys])
        xs = clamp(xs, 0, shape[1] - 2)
        ys = clamp(ys, 0, shape[0] - 2)
        for x in xs + [x + 1 for x in xs]:
            for y in range(ys[1], ys[0] + 1):
                # Color these locations full-on red
                mini_map[y][x][0] = 1.0
        for y in ys + [y + 1 for y in ys]:
            for x in range(xs[0], xs[1] + 1):
                # Color these locations full-on red
                mini_map[y][x][0] = 1.0
        return mini_map
        # Commented code is preparation for non-square image
        # canvas_as_array = to_rgb(canvas_as_array)
        # canvas_as_array[offsets[0]:, offsets[1]:] = mini_map
        # return canvas_as_array

    @output.capture()
    def on_pan_zoom_change(self, change_type):
        """
        Return a function that produces new scales when the user pans or zooms.

        :param change_type: One of ['x_min', 'x_max', 'y_min', 'y_max']
        """
        def handle_case(change):
            if change.new == None:
                return
            scales = self.get_current_scales()
            cases = {
                'x_min': {
                    'x': [change.new, scales['x'][1]],
                    'y': scales['y'],
                },
                'x_max': {
                    'x': [scales['x'][0], change.new],
                    'y': scales['y'],
                },
                'y_min': {
                    'x': scales['x'],
                    'y': [change.new, scales['y'][1]],
                },
                'y_max': {
                    'x': scales['x'],
                    'y': [scales['y'][0], change.new],
                },
            }
            new_scales = cases[change_type]
            self.redraw_mini_map(scales=new_scales)
        return handle_case

    @output.capture()
    def reset_enhancements(self, button=None):
        """
        Reset all of the image enhancement sliders.
        """
        [low, high] = self.value_range
        self.min_max_slider.min = low
        self.min_max_slider.max = high
        self.min_max_slider.value = [low, high]
        self.min_max_slider.step = (high - low) / self.enhancement_steps
        self.redraw()

    @output.capture()
    def save_current_image(self, button):
        """
        Save the current image with any processing applied.
        """
        directory = getattr(self.dataset, 'directory', os.getcwd())
        processed_directory = os.path.join(directory, 'ipysliceviewer')
        if not os.path.exists(processed_directory):
            os.makedirs(processed_directory)
        filepath = os.path.join(processed_directory, self.get_current_image_name())
        with open(filepath, 'wb') as f:
            f.write(self.image_plot.image.value)

    @output.capture()
    def hide_current_image(self, button):
        """
        Hide the current image and remember this as a setting.
        This is like a soft form of deleting the image.
        """
        # Need more thought on how this should be remembered across restarts for the current dataset
        # Rough idea: text file or similar containing SliceViewer settings, just need to figure out
        # a good naming scheme for datasets - possibly can take name from Experimenter or SliceViewer's own text box
        raise NotImplementedError

    @output.capture()
    def enhance_image(self, image_as_array):
        """
        Apply enhancement sliders to image_as_array and return as a numpy array
        """
        # These values are not normalized even though the image above is
        # (this allows the user to work with the input range)
        new_min, new_max = self.min_max_slider.value
        # So, we'll convert them before using them to scale the normalized image
        [low, high] = self.value_range
        def rescale(x, low, high):
            return (x - low) / (high - low)
        new_min = rescale(new_min, low, high)
        new_max = rescale(new_max, low, high)
        processed_image = rescale(to_rgb(image_as_array), new_min, new_max)
        processed_image = np.clip(processed_image, 0, 1)
        return processed_image

    @output.capture()
    def on_min_max_change(self, change):
        """
        Handle changes to the min/max slider.
        """
        self.redraw()

    @output.capture()
    def on_plane_change(self, change):
        """
        Called when the slice plane is toggled.
        """
        if hasattr(self.dataset, 'plane'):
            self.dataset.plane = change.new
            old_max = self.index_slider.max
            new_max = len(self.dataset) - 1
            self.index_slider.max = new_max
            self.index_slider.value = min(self.index_slider.value, new_max)
            # Guarantee the image updates even if index does not change
            self.redraw()
