from celldom.constant import CLASS_NAME_CELL
from celldom.config.cell_config import CLASS_INDEX_CELL
from celldom.extract import NO_IMAGES
from celldom.utils import assert_rgb, rgb2gray
from skimage.measure import regionprops, points_in_poly
from skimage.morphology import remove_small_holes, binary_closing
import numpy as np


def _default_components_data(chip_config):
    res = []
    if 'components' in chip_config:
        for component in chip_config['components'].keys():
            res.append(dict(component=component, occupancy=0.0))
    return res


def extract(img, cell_model, chip_config, dpf=NO_IMAGES, in_components_only=True):

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

    # Initialize results
    cells, components = [], _default_components_data(chip_config)

    # Determine indexes of detected cell objects (and return immediately if there are none)
    cell_idx = np.nonzero(detections['class_ids'] == CLASS_INDEX_CELL)[0]
    if cell_idx.size == 0:
        return cells, components

    # Subset detections to those for cells
    cell_masks = detections['masks'][..., cell_idx]
    # cell_rois = detections['rois'][cell_idx]
    cell_scores = detections['scores'][cell_idx]
    n_cells = len(cell_idx)

    # Convert RGB image to grayscale for intensity measurement
    intensity_image = rgb2gray(img)

    # Process individual cells
    for i in range(n_cells):
        props = regionprops(
            cell_masks[..., i].astype(int), cache=False,
            intensity_image=intensity_image, coordinates='rc'
        )
        assert len(props) <= 1

        # In the rare case that an object is detected with no mask, ignore it
        if len(props) == 0:
            continue

        props = props[0]

        # Centroid is (y, x) tuple
        centroid = props.centroid

        cells.append(dict(
            cell_id=i,
            cell_image=props.intensity_image if dpf.cell_image else None,
            # bbox format is min_row, min_col, max_row, max_col
            # - see http://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.regionprops
            # - use bbox instead of roi since they are not entirely identical
            roi_ymin=props.bbox[0],
            roi_xmin=props.bbox[1],
            roi_ymax=props.bbox[2],
            roi_xmax=props.bbox[3],
            score=cell_scores[i],
            area=props.area,
            eccentricity=props.eccentricity,
            solidity=props.solidity,
            centroid_y=centroid[0],
            centroid_x=centroid[1]
        ))

    # Return defaults if no valid cells were found
    if len(cells) == 0:
        return cells, components

    # Build vector of valid cell indexes
    cell_idx = np.array([c['cell_id'] for c in cells], dtype=int)

    # Categorize membership of cells in chip components, if any are present
    components = []
    if 'components' in chip_config:
        idx = []
        for component, measurements in chip_config['components'].items():
            points = measurements['boundary']

            # points_in_poly expects (x, y) tuples
            centroids = [(c['centroid_x'], c['centroid_y']) for c in cells]
            in_poly = points_in_poly(centroids, points)
            assert len(in_poly) == len(cells), \
                'Expecting {} boolean values but shape {}'.format(len(cells), in_poly.shape)

            # Add membership flags as booleans to each cell dict
            for i in range(len(cells)):
                cells[i]['in_' + component] = in_poly[i]
                if in_poly[i]:
                    idx.append(i)

            # Add component occupancy across all cells as distinct area / area of component
            component_masks = cell_masks[..., cell_idx][..., in_poly]
            assert component_masks.shape[-1] == in_poly.sum(), \
                'Expecting {} masks, found {}'.format(in_poly.sum(), component_masks.shape[-1])

            occupancy = 0.0
            component_area = chip_config.get_component_area(component)
            if component_masks.size > 0:
                # Fill holes up to 1% of component area
                occupancy = float(remove_small_holes(
                    binary_closing(component_masks.max(axis=-1)),
                    area_threshold=max(int(component_area*.01), 8)
                ).sum())
            occupancy = np.clip(occupancy / component_area, 0.0, 1.0)
            components.append(dict(component=component, occupancy=occupancy))

            # Mask Debugging
            # if component == 'chamber' and component_masks.size > 0:
            #     binary_mask = remove_small_holes(binary_closing(component_masks.max(axis=-1)), min_size=128)
            #     from skimage import io as skio
            #     from skimage import draw
            #     import time
            #     p = np.array(points)
            #     rr, cc = draw.polygon_perimeter(p[:, 1], p[:, 0], shape=binary_mask.shape)
            #     binary_mask[rr, cc] = True
            #     filepath = '/lab/data/celldom/cellmasks/binary_mask_{:.0f}_{}.png'\
            #         .format(occupancy*100, int(time.time()))
            #     print('File: {} | Occupancy: {}'.format(filepath, occupancy))
            #     skio.imsave(filepath, binary_mask.astype(int) * 255)
            # /Mask Debugging

        # If configured to do so, filter cells to only those within at least one component
        if in_components_only:
            # Remove using sorted, de-duplicated index list
            cells = [cells[i] for i in sorted(list(set(idx)))]

    return cells, components
