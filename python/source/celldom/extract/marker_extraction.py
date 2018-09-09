from skimage import morphology
from skimage import transform
from skimage import filters
from skimage import draw
from celldom.config import marker_config
import pandas as pd
import numpy as np


def extract(img, marker_model):
    """Extract marker centers from a multi-apartment image"""
    detections = marker_model.detect([img])[0]

    # All RGB image channels should be equivalent before proceeding
    if not np.all(img[..., 0] == img[..., 1]):
        raise ValueError('Expecting greyscale RGB images with equal channels')

    # Use bounding box predictions to establish points as marker centers
    centers = get_marker_centers(img[..., 0], detections['rois'])

    return centers


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

    # Given rois with cols y1, x1, y2, x2, compute average points on each axis
    centroids = np.stack([
        rois[:, 0] + ((rois[:, 2] - rois[:, 0]) / 2.0),
        rois[:, 1] + ((rois[:, 3] - rois[:, 1]) / 2.0)
    ], -1)
    return pd.DataFrame(centroids, columns=['y', 'x'])


def _angle_between(p1, p2):
    """Compute angle of p2 - p1 vector in degrees

    Args:
        p1: A 2D point in (y, x) form
        p2: A 2D point in (y, x) form
    Return:
        Angle in degrees between -180 and 180
    """
    return np.rad2deg(np.arctan2(*(p2 - p1)))


def get_marker_neighbors(centers, angle_range=(-25, 25), distance_range=None):
    if centers.ndim != 2:
        raise ValueError('Expecting centers to be 2D array (shape given = {})'.format(centers.shape))
    if distance_range is None:
        distance_range = (0, np.inf)
    n_points = centers.shape[0]

    res = []
    for i in range(n_points):
        candidates = []
        for j in range(n_points):
            if i == j:
                continue

            point = centers[i]
            neighbor = centers[j]

            # Compute angle between points for either ordering, ultimately looking
            # for horizontal angle offsets between them
            angle1 = _angle_between(point, neighbor)
            angle2 = _angle_between(neighbor, point)

            # Choose the angle with difference vector pointing to the right
            angle = angle1 if abs(angle1) < abs(angle2) else angle2

            # Only add neighbors if they are within a certain angle and distance from the current point
            if angle_range[0] <= angle <= angle_range[1]:
                distance = np.linalg.norm(neighbor - point)
                if distance_range[0] <= distance <= distance_range[1]:
                    candidates.append((i, point[0], point[1], j, neighbor[0], neighbor[1], angle, distance))

        if not candidates:
            continue
        candidates = np.array(candidates)
        closest_idx = np.argsort(candidates[:, -1])[0]
        res.append(candidates[closest_idx])

    columns = [
        'point_idx', 'point_y', 'point_x', 'neighbor_idx',
        'neighbor_y', 'neighbor_x', 'angle', 'distance'
    ]
    return pd.DataFrame(res, columns=columns)


def overlay_marker_centers(img, centers, color=0, radius=2):
    res = img.copy()
    for i, r in centers.astype(int).iterrows():
        rr, cc = draw.circle(r['y'], r['x'], radius=radius, shape=img.shape)
        res[rr, cc] = color
    return res
