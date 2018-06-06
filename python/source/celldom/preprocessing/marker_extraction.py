from skimage import morphology
from skimage import transform
from skimage import filters
import pandas as pd
import numpy as np


def get_marker_centers(img, rois):
    """Identify the center point for each marker in the given ROI detections

    Args:
        img: Original image
        rois: Array with shape (n_roi, 4) where the 4 column values are y1, x1, y2, x2 (in that order);
            See https://github.com/matterport/Mask_RCNN/blob/4129a27275c48c672f6fd8c6303a88ba1eed643b/mrcnn/visualize.py#L276
            for an example of where this roi format is used
    Return:
        A (n_roi, 2) array of (y, x) coordinates corresponding to weighted centers of markers (with subpixel resolution)
    """
    if img.ndim != 2:
        raise ValueError('Image to extract markers from should have 2 dimensions (given shape = {})'.format(img.shape))
    if rois.shape[1] != 4 or rois.ndim != 2:
        raise ValueError('Expecting rois to be 2D with 4 columns (given shape = {})'.format(rois.shape))

    n_roi = rois.shape[0]
    centroids = []
    for i in range(n_roi):
        y1, x1, y2, x2 = rois[i]

        # Extract marker from image
        marker_img = img[y1:y2, x1:x2]

        # Normalize and clean the raw marker image
        marker_img = prep_marker_img(marker_img)

        # Compute marker centroid
        marker_center = get_weighted_centroid(marker_img)

        # Convert relative centroid coordinates back to coords of image
        centroids.append(np.array(marker_center) + np.array([y1, x1]))
    return pd.DataFrame(centroids, columns=['y', 'x'])


def prep_marker_img(img):
    """Prepare a raw marker image by filtering and subtracting a disk around edges

    More precisely, this function will median filter on a 3x3 patch to remove outliers,
    create a disk that captures something close to a centered circle, and then set all
    values outside of that disk to 0.

    Args:
        img: Marker image (usually close to a 15x15 square with a circular marker in middle)
    Return:
        Transformed marker image with same shape as input
    """
    img = filters.rank.median(img)
    disk = morphology.disk(radius=np.array(img.shape).min() // 2)
    disk = transform.resize(disk, img.shape, mode='constant', order=1)
    return img * disk


def get_weighted_centroid(img):
    if np.any(img < 0):
        raise ValueError('Image must be non-negative to use for weighted centroid calculation')
    row, col = np.indices(img.shape[:2])
    return np.average(row, weights=img), np.average(col, weights=img)


def _angle_between(p1, p2):
    """Compute angle of p2 - p1 vector in degrees

    Args:
        p1: A 2D point in (y, x) form
        p2: A 2D point in (y, x) form
    Return:
        Angle in degrees between -180 and 180
    """
    return np.rad2deg(np.arctan2(*(p2 - p1)))


def get_marker_neighbors(centers, angle_range=(-25, 25)):
    if centers.ndim != 2:
        raise ValueError('Expecting centers to be 2D array (shape given = {})'.format(centers.shape))
    n_points = centers.shape[0]

    res = []
    for i in range(n_points):
        candidates = []
        for j in range(n_points):
            if i == j:
                continue

            point = centers[i]
            neighbor = centers[j]
            angle = _angle_between(point, neighbor)

            if angle_range[0] <= angle <= angle_range[1]:
                distance = np.linalg.norm(neighbor - point)
                candidates.append((i, point[0], point[1], j, neighbor[0], neighbor[1], angle, distance))

        if not candidates:
            continue
        candidates = np.array(candidates)
        closest_idx = np.argsort(candidates[:, -1])[0]
        res.append(candidates[closest_idx])

    return pd.DataFrame(res,
                        columns=['point_idx', 'point_y', 'point_x', 'neighbor_idx', 'neighbor_y', 'neighbor_x', 'angle',
                                 'distance'])

