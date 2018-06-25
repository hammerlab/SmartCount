from celldom.constant import CLASS_NAME_CELL
from celldom.config.cell_config import CLASS_INDEX_CELL
from skimage.measure import regionprops
import numpy as np


def extract(img, cell_model):
    # Get MRCNN detections
    detections = cell_model.detect([img])[0]

    # Determine which detected objects are cells
    cell_detections = (detections['class_ids'] == CLASS_INDEX_CELL)

    # Get cell masks (masks have shape (h, w, n_objects))
    assert detections['masks'].ndim == 3
    if detections['masks'].shape[-1] > 0:
        assert detections['masks'].dtype == np.bool
    cell_masks = detections['masks'][..., cell_detections]

    # Quantify individual cells
    cells = []
    for i in range(cell_masks.shape[-1]):
        props = regionprops(cell_masks[..., i].astype(int))
        assert len(props) <= 1

        # In the rare case that an object is detected with no mask, ignore it
        if len(props) == 0:
            continue

        props = props[0]
        cells.append(dict(
            area=props.area,
            location=props.centroid
        ))
    return cells
