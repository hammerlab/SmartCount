import numpy as np
from skimage import exposure
from skimage import color


def assert_rgb(image):
    """Ensure that an image has 3 dimensions, 3 channels and is 8-bits unsigned"""
    assert image.ndim == 3 and image.dtype == np.uint8 and image.shape[-1] == 3, \
        'Expecting RGB uint8 image, not image with shape {} and type {}'.format(image.shape, image.dtype)


def rgb2gray(image):
    """Convert 3 channel RGB to gray scale uint8

    RGB channels are weighted individually here, summed, and then converted back from the resulting
    0-1 range to uint8.  See here for more details:
    http://scikit-image.org/docs/dev/user_guide/transforming_image_data.html

    Args:
        image: RGB uint8 image
    Returns:
        2D uint8 image
    """
    return exposure.rescale_intensity(color.rgb2gray(image), in_range=(0, 1), out_range=np.uint8).astype(np.uint8)
