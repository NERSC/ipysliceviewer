# Standard lib
import os
import csv
from collections import OrderedDict
# Third party
from PIL import Image, ImageEnhance
from io import BytesIO
from bqplot import (
    LinearScale, Figure, PanZoom,
    Toolbar, Image as BQImage,
)
from ipywidgets import (
    VBox, HBox, Layout, GridBox,
    Button, ToggleButton,
    IntSlider, FloatSlider,
    Play, Output, jslink,
    Image as IPyImage,
    # Text, Combobox, Tab, Output
)
# Local
# from ipypathchooser import PathChooser

class SliceViewer(VBox):
    def __init__(
        self,
        default_directory=os.path.join(os.getcwd(), 'images'),
        default_params=None,
        title='',
        **kwargs
    ):
        # For debugging
        self.output = Output(layout={'border': '1px solid white'})

        # Vertical margin between blocks of UI elements
        # TODO: Use this
        self.vertical_spacing = 10

        self._images_directory = default_directory
        self.image_types = ['.png', '.jpg']
        images = filter(lambda f: f[-4:] in self.image_types, os.listdir(self._images_directory))
        self.image_names = sorted(images)
        # self.print(f'image_names: {self.image_names}')
        # Assume consistent image_size and read from first image
        filepath = self.get_filepath(self.image_names[0])
        image = Image.open(filepath)
        self.image_size = image.size
        # The currently displayed image backing `self.Figure`
        self.current_image = IPyImage(
            width=self.image_size[0],
            height=self.image_size[1],
        )
        self.current_image.observe(self.redraw_marks, names='value')

        self.index_slider = IntSlider(
            value=0,
            min=0,
            max=len(self.image_names),
            step=1,
            description='Index:',
            orientation='horizontal',
            readout=True,
            readout_format='d',
            continuous_update=True,
            layout=Layout(
                grid_area='index-slider',
                margin='auto -20px auto 0',
                width='100%',
            ),
        )
        self.index_slider.observe(self.on_image_index_change, names='value')

        self.scales = {
            'x': LinearScale(),
            'y': LinearScale(),
        }

        self.image = BQImage(
            image=self.current_image,
            scales=self.scales,
        )

        self.max_image_width=600
        self.max_image_height=600

        self.figure = Figure(
            marks=[self.image],
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
            ),
        )
        margin = self.figure.fig_margin
        width = margin['left'] + self.image_size[0] + margin['right']
        height = margin['top'] + self.image_size[1] + margin['bottom']
        self.figure.layout.width = f'{min(width, self.max_image_width)}px'
        self.figure.layout.height = f'{min(height, self.max_image_height)}px'

        # Custom toolbar
        toolbar_width = '100%'
        toolbar_margin = '0px 0 10px 0'
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

        # Image adjustments
        self.brightness_slider = FloatSlider(
            value=1,
            min=0,
            max=10,
            step=0.1,
            description='Brightness',
            orientation='vertical',
            readout=True,
            readout_format='.1f',
            continuous_update=True,
            layout=Layout(
                grid_area='brightness-slider',
                margin='0',
                height='100%',
            ),
        )
        self.brightness_slider.observe(self.enhance_current_image, names='value')

        self.contrast_slider = FloatSlider(
            value=1,
            min=0,
            max=10,
            step=0.1,
            description='Contrast',
            orientation='vertical',
            readout=True,
            readout_format='.1f',
            continuous_update=True,
            layout=Layout(
                grid_area='contrast-slider',
                margin='0',
                height='100%',
            ),
        )
        self.contrast_slider.observe(self.enhance_current_image, names='value')

        self.sharpness_slider = FloatSlider(
            value=1,
            min=0,
            max=10,
            step=0.1,
            description='Sharpness',
            orientation='vertical',
            readout=True,
            readout_format='.1f',
            continuous_update=True,
            layout=Layout(
                grid_area='sharpness-slider',
                margin='0',
                height='100%',
            ),
        )
        self.sharpness_slider.observe(self.enhance_current_image, names='value')

        # Animation
        self.play = Play(
            value=self.index_slider.value,
            min=self.index_slider.min,
            max=self.index_slider.max,
            step=self.index_slider.step,
        )
        jslink((self.play, 'value'), (self.index_slider, 'value'))

        self.bottom_bar = HBox(
            children=[
                self.play,
                self.index_slider,
            ],
            layout=Layout(
               grid_area='bottom-bar',
               margin=f'10px 0 0 0',
               overflow='hidden',
            )
        )

        # Layout
        self.gridbox = GridBox(
            children=[
                self.figure,
                self.toolbar,
                self.brightness_slider,
                self.contrast_slider,
                self.sharpness_slider,
                self.bottom_bar,
            ],
            layout=Layout(
                width='auto',
                # height='500px',
                # Add margin when filepicker is in place
                # margin=f'{self.vertical_spacing}px 0 0 0',
                grid_gap='0px 10px',
                # grid_template_columns='auto auto auto',
                grid_template_columns=f'{self.get_figure_w()}px 60px 60px 60px',
                grid_template_rows=f'172px {self.get_figure_h() - 110}px 52px',
                grid_template_areas='''
                    "figure toolbar toolbar toolbar"
                    "figure brightness-slider contrast-slider sharpness-slider"
                    "bottom-bar bottom-bar bottom-bar bottom-bar"
                '''
            ),
        )

        self.set_current_image(self.image_names[0])

        # Call VBox super class __init__
        super().__init__(
            children=[
                # self.path_chooser,
                self.gridbox,
            ],
            layout=Layout(width='auto'),
            **kwargs,
        )

    def print(self, message):
        """
        Print `message` to self.output for debugging purposes.
        """
        self.output.append_stdout(f'{message}\n')

    def get_filepath(self, image_name):
        """
        Return the full path to `image_name`.
        """
        return os.path.join(self._images_directory, image_name)

    def get_current_image_name(self):
        """
        Return the name of the current image selected according to `self.index_slider`.
        """
        return self.image_names[self.index_slider.value]

    def set_current_image(self, image_name):
        """
        Update the current displayed image according to `image_name`.
        """
        filepath = self.get_filepath(image_name)
        with open(filepath, 'rb') as f:
            self.current_image.value = f.read()
            self.current_image.original = self.current_image.value
            self.current_image.format=filepath[-3:]
        self.enhance_current_image()

    def redraw_marks(self, change):
        """
        Redraw the image marks on the figure.
        """
        self.figure.marks = []
        self.figure.marks = [self.image]

    def on_image_index_change(self, change):
        """
        Load and display the new image.
        """
        self.set_current_image(self.image_names[change.new])

    def on_pan_zoom_toggle(self, change):
        """
        Update the `self.figure` interaction.
        """
        if change.new:
            self.figure.interaction = self.pan_zoom
        else:
            self.figure.interaction = None

    def reset_pan_zoom(self, button):
        """
        Reset `self.figure` scales.
        """
        self.image.scales['x'].min = None
        self.image.scales['x'].max = None
        self.image.scales['y'].min = None
        self.image.scales['y'].max = None

    def reset_enhancements(self, button):
        """
        Reset all of the image enhancement sliders.
        """
        self.brightness_slider.value = 1
        self.contrast_slider.value = 1
        self.sharpness_slider.value = 1

    def save_current_image(self, button):
        """
        Save the current image with any processing applied.
        """
        processed_directory = os.path.join(self._images_directory, 'ipysliceviewer')
        if not os.path.exists(processed_directory):
            os.makedirs(processed_directory)
        filepath = os.path.join(processed_directory, self.get_current_image_name())
        with open(filepath, 'wb') as f:
            f.write(self.current_image.value)

    def hide_current_image(self, button):
        """
        Hide the current image and remember this as a setting.
        This is like a soft form of deleting the image.
        """
        raise Exception('Not implemented')

    def get_figure_w(self):
        """
        Return figure layout width as an integer.
        """
        return int(self.figure.layout.width[:-2])

    def get_figure_h(self):
        """
        Return figure layout height as an integer.
        """
        return int(self.figure.layout.height[:-2])

    def enhance_current_image(self, change=None):
        """
        Apply enhancement sliders to `self.current_image`.
        """
        image = Image.open(BytesIO(self.current_image.original))
        image = ImageEnhance.Brightness(image).enhance(self.brightness_slider.value)
        image = ImageEnhance.Contrast(image).enhance(self.contrast_slider.value)
        image = ImageEnhance.Sharpness(image).enhance(self.sharpness_slider.value)
        temp = BytesIO()
        image.save(temp, format=self.current_image.format)
        self.current_image.value = temp.getvalue()
