from skimage import transform
import pandas as pd
import numpy as np


def _rotate_vectors_2d(arr, rotation, origin):
    assert arr.ndim == 2
    alpha = np.deg2rad(rotation)
    a = np.array([[np.cos(alpha), -np.sin(alpha)],
                  [np.sin(alpha), np.cos(alpha)]])
    arr = arr.T - origin.reshape(-1, 1)
    return (origin.reshape(-1, 1) + np.dot(a, arr)).T


def apply_normalization(img, centers, rotation=None, scale=None):
    """Apply inferred rotations and scales to a raw image and its corresponding marker locations"""

    # If a rotation is given, apply it to the image and [carefully] to the marker vectors as well
    if rotation:
        img = transform.rotate(img, rotation)

        # Use same formula for origin as in transform.rotate
        origin = np.array(img.shape[:2]) / 2. - .5
        centers = pd.DataFrame(
            _rotate_vectors_2d(centers[['y', 'x']].values, rotation, origin),
            index=centers.index, columns=centers.columns)

    # If a scale is given, apply that too
    if scale:
        img = transform.rescale(img, scale)
        centers = centers[['y', 'x']] * scale

    return img, centers


def extract_around_marker(img, center, margins):
    """Extract an image patch (from a raw microscope image) using the specified marker location and margins"""
    cy, cx = center

    # Defining bounding box to crop out
    ymin = cy - margins['top']
    ymax = cy + margins['bottom']
    xmin = cx - margins['left']
    xmax = cx + margins['right']

    # Return nothing if any part of the box is outside the original image
    if xmin < 0 or ymin < 0 or xmax > img.shape[1] or ymax > img.shape[0]:
        return None

    img = img[ymin:ymax, xmin:xmax]
    assert np.all(np.array(img.shape) > 0)
    return img


def extract_images(img, centers, chip_config):
    """Extract all relevant patches from raw, multi-component images"""
    imgs = []
    for i, r in centers.astype(int).iterrows():
        center = r['y'], r['x']

        apt_img = extract_around_marker(img, center, chip_config['apt_margins'])
        if apt_img is None:
            continue

        apt_num_img = extract_around_marker(img, center, chip_config['apt_num_margins'])
        if apt_num_img is None:
            continue

        imgs.append(dict(
            marker_center=center,
            apartment_image=apt_img,
            apartment_num_image=apt_num_img
        ))
    return imgs
