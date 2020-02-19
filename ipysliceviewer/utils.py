# Standard lib
import math
from io import BytesIO
# Third party
import numpy as np
from PIL import Image

def normalize(arr):
    """
    Normalize a numpy array to [0, 1].
    """
    low = np.min(arr)
    high = np.max(arr)
    if high == low:
        return arr
    return np.clip((arr.astype(float) - low) / (high - low), 0, 1)

def numpy_to_PIL(arr):
    """
    Convert a normalized numpy array with values in [0, 1] to a PIL Image.
    """
    return Image.fromarray((255 * arr).astype(np.uint8))

def PIL_to_numpy(image):
    """
    Convert a PIL image to a normalized numpy array with values in range [0, 1].
    """
    return normalize(np.array(image))

def bytes_to_PIL(byte_string):
    """
    Convert an image stored as a byte string to PIL format.
    """
    return Image.open(BytesIO(byte_string))

def PIL_to_bytes(image, format='jpeg'):
    """
    Convert a PIL image to byte string format.
    """
    if image.mode == 'RGBA':
        format='png'
    temp = BytesIO()
    image.save(temp, format=format)
    return temp.getvalue()

def to_grayscale(arr):
    """
    Convert an RGB numpy array to grayscale.
    """
    # Just return if it's grayscale
    if arr.shape[-1] == 1:
        return arr
    return np.mean(arr, axis=-1)

def to_rgb(arr):
    """
    Convert a grayscale numpy array to rgb.
    """
    # Just return if it's rgb or rgba
    if arr.shape[-1] == 3 or arr.shape[-1] == 4:
        return arr
    return np.dstack([arr]*3)

def draw_checkerboard_canvas(height=180, width=180, box_size=10):
    """
    Draw a checkerboard_canvas like the classic transparency background.
    """
    i_max = math.ceil(height/box_size)
    j_max = math.ceil(width/box_size)
    canvas = np.zeros((i_max*box_size, j_max*box_size))
    for i in range(i_max):
        is_gray = i % 2 == 0
        for j in range(j_max):
            is_gray = not is_gray
            color = 0.5 if is_gray else 1.0
            box = np.ones((box_size, box_size)) * color
            canvas[i*box_size:(i+1)*box_size, j*box_size:(j+1)*box_size] = box
    return canvas[:height, :width]

def get_offsets_and_resize_for_canvas(image_as_array, canvas_as_array):
    """
    Returns the offests and new_image_as_array that would let us draw
    image_as_array over canvas_as_array with
    `canvas.paste(new_image, box=offsets)`.
    """
    canvas = numpy_to_PIL(canvas_as_array)
    image = numpy_to_PIL(image_as_array)
    ratios = [canvas.size[0] / image.size[0], canvas.size[1] / image.size[1]]
    big_index = 0 if ratios[0] > ratios[1] else 1
    new_image = image.resize((
        min(int(image.size[0]*ratios[big_index^1]), canvas.size[0]),
        min(int(image.size[1]*ratios[big_index^1]), canvas.size[1]),
    ))
    diff = canvas.size[big_index] - new_image.size[big_index]
    offsets = (0, diff//2) if big_index == 1 else (diff//2, 0)
    new_image_as_array = PIL_to_numpy(new_image)
    return [offsets, new_image_as_array]
