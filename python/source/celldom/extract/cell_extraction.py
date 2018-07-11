from celldom.constant import CLASS_NAME_CELL
from celldom.config.cell_config import CLASS_INDEX_CELL
from celldom.extract import NO_IMAGES
from celldom.utils import assert_rgb, rgb2gray
from skimage.measure import regionprops
import numpy as np


def extract(img, cell_model, chip_config, dpf=NO_IMAGES):

    # Validate image input
    assert_rgb(img)

    # Get MRCNN detections -- Notes from docs on results:
    # Returns a list of dicts, one dict per image. The dict contains:
    # rois: [N, (y1, x1, y2, x2)] detection bounding boxes
    # class_ids: [N] int class IDs
    # scores: [N] float probability scores for the class IDs
    # masks: [H, W, N] instance binary masks
    detections = cell_model.detect([img])[0]

    # Verify results to avoid obscure errors if MRCNN changes
    assert detections['masks'].ndim == 3
    # Check masks are boolean or empty in last (count) dimension (in which case they are float64)
    assert detections['masks'].dtype == np.bool or detections['masks'].shape[-1] == 0, \
        'Expecting boolean masks but got shape = {}, dtype = {}'\
        .format(detections['masks'].shape, detections['masks'].dtype)
    assert detections['rois'].ndim == 2
    assert detections['scores'].ndim == 1
    assert detections['class_ids'].ndim == 1

    # Determine indexes of detected cell objects (and return immediately if there are none)
    cell_idx = np.nonzero(detections['class_ids'] == CLASS_INDEX_CELL)[0]
    if cell_idx.size == 0:
        return []

    # Subset detections to those for cells
    cell_masks = detections['masks'][..., cell_idx]
    cell_rois = detections['rois'][cell_idx]
    cell_scores = detections['scores'][cell_idx]
    n_cells = len(cell_idx)

    # Convert RGB image to grayscale for intensity measurement
    intensity_image = rgb2gray(img)

    # Process individual cells
    cells = []
    for i in range(n_cells):
        props = regionprops(
            cell_masks[..., i].astype(int), cache=False,
            intensity_image=intensity_image, coordinates='rc'
        )
        assert len(props) <= 1

        # In the rare case that an object is detected with no mask, ignore it
        if len(props) == 0:
            continue

        # TODO: Use chip config to add further classifications on location of cell (or
        # possibly, use masks for inferred locations of chip components)
        # e.g. in chamber, in trap, in channel based on centroid

        props = props[0]
        centroid = props.centroid
        cells.append(dict(
            cell_image=props.intensity_image if dpf.cell_image else None,
            roi_ymin=int(cell_rois[i][0]),
            roi_xmin=int(cell_rois[i][1]),
            roi_ymax=int(cell_rois[i][2]),
            roi_xmax=int(cell_rois[i][3]),
            score=cell_scores[i],
            area=props.area,
            eccentricity=props.eccentricity,
            solidity=props.solidity,
            centroid_y=centroid[0],
            centroid_x=centroid[1]
        ))
    return cells
