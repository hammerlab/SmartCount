from celldom.constant import CLASS_NAME_CELL
from celldom.config.cell_config import CLASS_INDEX_CELL
from celldom.extract import DPF_NONE
from skimage.measure import regionprops
from skimage.color import rgb2gray
import numpy as np


def extract(img, cell_model, chip_config, dpf=DPF_NONE):

    # Validate image input
    if img.ndim != 3 or img.dtype != np.uint8 or img.shape[2] != 3:
        raise ValueError(
            'Expecting RGB uint8 image, not image with shape {} and type {}'
            .format(img.shape, img.dtype)
        )

    # Get MRCNN detections -- Notes from docs on results:
    # Returns a list of dicts, one dict per image. The dict contains:
    # rois: [N, (y1, x1, y2, x2)] detection bounding boxes
    # class_ids: [N] int class IDs
    # scores: [N] float probability scores for the class IDs
    # masks: [H, W, N] instance binary masks
    detections = cell_model.detect([img])[0]

    # Verify results to avoid obscure errors if MRCNN changes
    assert detections['masks'].ndim == 3
    assert detections['masks'].dtype == np.bool
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

    # If given a multi-channel image, assume they are all the same and convert back to 2D for
    # regionprops intensity image
    assert np.all(img[..., 0] == img[..., 1]), \
        '3 channel image does not have equal channel values (true RGB images are not yet supported)'
    intensity_image = img[..., 0]

    # Process individual cells
    cells = []
    for i in range(n_cells):
        props = regionprops(cell_masks[..., i].astype(int), cache=False, intensity_image=intensity_image)
        assert len(props) <= 1

        # In the rare case that an object is detected with no mask, ignore it
        if len(props) == 0:
            continue

        # TODO: Use chip config to add further classifications on location of cell (or
        # possibly, use masks for inferred locations of chip components)
        # e.g. in chamber, in trap, in channel based on centroid

        props = props[0]
        cells.append(dict(
            image=props.intensity_image if dpf.cell_image else None,
            roi=cell_rois[i],
            score=cell_scores[i],
            area=props.area,
            eccentricity=props.eccentricity,
            solidity=props.solidity,
            location=props.centroid
        ))
    return cells
