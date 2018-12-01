import pandas as pd
import numpy as np
from skimage import transform
from skimage.exposure import rescale_intensity
from skimage.feature import register_translation
from celldom.extract import marker_extraction, digit_extraction, cell_extraction
from celldom.extract import NO_IMAGES
from celldom.exception import NoMarkerException
from celldom.utils import assert_rgb, rgb2gray, extract_patch


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


def partition_around_marker(img, center, margins):
    """Extract an image patch (from a raw microscope image) using the specified marker location and margins"""
    cy, cx = center

    # Defining bounding box to crop out
    ymin = cy + margins['top']
    ymax = cy + margins['bottom']
    xmin = cx + margins['left']
    xmax = cx + margins['right']

    # Return nothing if any part of the box is outside the original image
    if xmin < 0 or ymin < 0 or xmax > img.shape[1] or ymax > img.shape[0]:
        return None

    img = img[ymin:ymax, xmin:xmax]
    assert np.all(np.array(img.shape) > 0)
    return img


def partition_digit_images(img, bounds, rotation):
    """Extract digit images in a multi-digit image using the given x (i.e. column) bounds

    Args:
        img: Image containing multiple-digits
        bounds: Sequence of dictionaries containing the keys 'left', 'right', 'top', 'bottom' used
            to slide digit images (values are offsets within entire number image)
        rotation: Angle in degrees by which to rotate individual images
    Returns:
        List of individual digit images (with length equal to len(bounds))
    """
    imgs = [img[b['top']:b['bottom'], b['left']:b['right']] for b in bounds]
    if rotation:
        dinfo = np.iinfo(img.dtype)
        imgs = [
            # Note that rotations are counter-clockwise in degrees and that results are float64 (and
            # need to be converted back to provided type)
            np.clip(
                transform.rotate(i, rotation, preserve_range=True, resize=True),
                dinfo.min, dinfo.max
            ).astype(img.dtype)
            for i in imgs
        ]
    return imgs


def get_apartment_image_translation(img, chip_config, patch_size=(16, 16)):
    """Infer the translation of an extracted apartment image against a template image used for chip configuration"""

    # Convert RGB images to 2D for registration
    tgt_img = rgb2gray(assert_rgb(img))

    # Get template image for registration and verify that it is of the same size
    ref_img = chip_config.get_template_image()
    assert ref_img.shape == tgt_img.shape, \
        'Template and extracted apartment image must have same shape (template image shape = {}, extracted = {})' \
        .format(ref_img.shape, tgt_img.shape)

    # Extract identically sized patchs from both reference and target images
    center = chip_config.get_marker_center()
    ref_patch = extract_patch(ref_img, center, patch_size)
    tgt_patch = extract_patch(tgt_img, center, patch_size)

    # Compute translation and error
    shifts, error, _ = register_translation(ref_patch, tgt_patch)
    assert len(shifts) == 2

    return shifts.astype(int), error, (ref_patch, tgt_patch)


def partition_chip(img, centers, chip_config, focus_model=None, enable_registration=True):
    """Extract all relevant patches from raw, multi-component images (ie a chip)"""
    partitions = []

    # Sort centers to get a non-arbitrary apartment id setting
    for i, r in centers.astype(int).sort_values(['y', 'x']).iterrows():
        center = r['y'], r['x']

        apt_img = partition_around_marker(img, center, chip_config['apt_margins'])
        if apt_img is None:
            continue

        # Register image against template if configured to do so
        if enable_registration:
            # Determine translation, redefine center based on this, and then re-extract apartment image
            registration_shifts, registration_error, _ = get_apartment_image_translation(apt_img, chip_config)
            center = center[0] - registration_shifts[0], center[1] - registration_shifts[1]
            apt_img = partition_around_marker(img, center, chip_config['apt_margins'])
            if apt_img is None:
                continue
        else:
            registration_shifts, registration_error = [0, 0], 0.

        # Ensure that apartment image is RGB
        assert_rgb(apt_img)

        # If an image focus/quality model was given, use it to score the apartment image
        focus_score = 0
        if focus_model is not None:
            # Note that images are carried around through most of the codebase as RGB with repeated channels,
            # and the focus classifier expects 2D images so it is converted here
            focus_score = focus_model.score(rgb2gray(apt_img))

        apt_num_img = partition_around_marker(img, center, chip_config['apt_num_margins'])
        apt_num_digit_imgs = partition_digit_images(
            apt_num_img, chip_config['apt_num_digit_bounds'], chip_config['apt_num_rotation'])

        st_num_img = partition_around_marker(img, center, chip_config['st_num_margins'])
        st_num_digit_imgs = partition_digit_images(
            st_num_img, chip_config['st_num_digit_bounds'], chip_config['st_num_rotation'])

        partitions.append(dict(
            apt_id=i,
            marker_center_y=center[0],
            marker_center_x=center[1],
            apt_image=apt_img,
            apt_image_height=apt_img.shape[0],
            apt_image_width=apt_img.shape[1],
            focus_score=focus_score,
            apt_num_image=apt_num_img,
            apt_num_digit_images=apt_num_digit_imgs,
            st_num_image=st_num_img,
            st_num_digit_images=st_num_digit_imgs,
            registration_shift_y=registration_shifts[0],
            registration_shift_x=registration_shifts[1],
            registration_error=registration_error
        ))
    return partitions


def extract(
        image, marker_model, chip_config,
        digit_model=None, cell_model=None, focus_model=None, enable_registration=True,
        dpf=NO_IMAGES, angle_tolerance=10, distance_tolerance=20):

    # Make sure provided image is RGB
    assert_rgb(image)

    ##################
    ## Extract Markers
    ##################

    # Determine center points of markers
    centers = marker_extraction.extract(image, marker_model)
    if len(centers) == 0:
        raise NoMarkerException('No markers found in image')

    # Determine marker neighbors based on an angular offset threshold and proximity
    spacing = chip_config['marker_spacing']['horizontal']
    neighbors = marker_extraction.get_marker_neighbors(
        centers.values,
        # Specify angle and distance windows used to identify valid neighboring markers
        angle_range=(-angle_tolerance, angle_tolerance),
        distance_range=(spacing - distance_tolerance, spacing + distance_tolerance)
    )

    # Restrict centers of markers to only those found to have valid neighbors
    if len(neighbors) == 0:
        raise NoMarkerException('No valid markers found in image')
    centers = centers.iloc[neighbors['point_idx']]
    assert len(centers) > 0

    ########################
    ## Apply Transformations
    ########################

    # Infer the overall rotation and scale of the image as the median of those same
    # quantities determined for each adjacent marker pair
    rotation = neighbors['angle'].median()

    # Apply inferred rotation
    norm_image, norm_centers = apply_normalization(image, centers, rotation=rotation)

    # Rotation and rescaling may result in a change of bit depth which should be undone
    # as soon as possible to maintain consistency with uint8 processing
    if norm_image.dtype != np.uint8:
        assert np.all(norm_image >= 0) and np.all(norm_image <= 1)
        norm_image = rescale_intensity(norm_image, in_range=(0, 1), out_range=np.uint8).astype(np.uint8)

    ################################
    ## Extract Around Marker Offsets
    ################################

    partitions = partition_chip(
        norm_image, norm_centers, chip_config,
        focus_model=focus_model, enable_registration=enable_registration
    )

    # Add digit inference to address images if a digit model was provided
    if digit_model is not None:
        for partition in partitions:
            partition['apt_num'], partition['apt_num_digit_scores'] = digit_extraction\
                .extract_single_digits(partition['apt_num_digit_images'], digit_model)
            partition['st_num'], partition['st_num_digit_scores'] = digit_extraction\
                .extract_single_digits(partition['st_num_digit_images'], digit_model)

    # Add cell inference if cell model was provided
    if cell_model is not None:
        for partition in partitions:
            partition['cells'], partition['components'] = cell_extraction.extract(
                partition['apt_image'], cell_model, chip_config, dpf=dpf)

    return partitions, norm_image, norm_centers, neighbors, rotation


def visualize_partition(partition, prep_fn=None):
    import matplotlib.pyplot as plt

    imgs = []

    imgs.append(partition['apt_image'])

    imgs.append(partition['apt_num_image'])
    imgs.extend(partition['apt_num_digit_images'])

    imgs.append(partition['st_num_image'])
    imgs.extend(partition['st_num_digit_images'])

    ncol = 3
    nrow = int(np.ceil(len(imgs) / ncol))
    fig, ax = plt.subplots(nrow, ncol)
    fig.set_size_inches(12, 12)
    ax = ax.ravel()
    for a in ax:
        a.axis('off')
    for i, img in enumerate(imgs):
        ax[i].imshow(img if prep_fn is None else prep_fn(img))
        ax[i].axis('on')



