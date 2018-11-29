import numpy as np
from skimage import exposure
from skimage import color


def assert_rgb(image):
    """Ensure that an image has 3 dimensions, 3 channels and is 8-bits unsigned"""
    assert image.ndim == 3 and image.dtype == np.uint8 and image.shape[-1] == 3, \
        'Expecting RGB uint8 image, not image with shape {} and type {}'.format(image.shape, image.dtype)
    return image


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


def extract_patch(img, center, offsets, raise_on_oob=True):
    """Extract a patch from a central point using offsets

    Args:
        img: Image array with leading two dimensions as HW (i.e. RC)
        center: Central pixel location for patch extraction; must be 2 item sequence with format (row, col)
        offsets: 2 item sequence with offsets around center (row, col); e.g. a (3, 2) size patch
            will result in a 7x5 (rows x cols) patch image
        raise_on_oob: Raise errors if patch size exceeds image bounds; otherwise, extract as much of the image
            as possible
    """
    if img.ndim < 2:
        raise ValueError('Image must contain at least 2 dimensions for patch extraction (shape = {})'.format(img.shape))
    if len(center) != 2:
        raise ValueError('Center pixel location must have length 2 (given = {})'.format(center))
    if len(offsets) != 2:
        raise ValueError('Pixel offsets must have length 2 (given = {})'.format(offsets))

    center, offsets, shape = np.array(center), np.array(offsets), np.array(img.shape)
    if np.any(offsets < 1):
        raise ValueError('Pixel offsets must be greater than 0 (given = {})'.format(offsets))

    start, stop = center - offsets, center + offsets + 1
    if raise_on_oob:
        if np.any(start < 0) or np.any(stop > shape):
            raise ValueError(
                'Patch coordinates out of bounds (image shape = {}, center = {}, offsets = {}, start = {}, stop = {}'
                .format(shape, center, offsets, start, stop)
            )
    start, stop = np.maximum(start, 0), np.minimum(stop, shape)
    return img[start[0]:stop[0], start[1]:stop[1]]
