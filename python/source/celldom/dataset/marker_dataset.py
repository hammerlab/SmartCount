import numpy as np
from cvutils.mrcnn import dataset as mrcnn_dataset
from celldom.config import marker_config
from skimage import transform, exposure


class MarkerDataset(mrcnn_dataset.RectLabelDataset):

    def __init__(self, reflect_images=True, scale_factor=None):
        """Dataset used to model markers in raw microscope images (multi-chamber)

        Args:
            reflect_images: Whether or not to reflect images around y-axis (i.e. reverse the columns)
            scale_factor: Factor by which both dimensions of an image should be resized (default is None)
        """
        super(MarkerDataset, self).__init__()
        self.reflect_images = reflect_images
        self.scale_factor = scale_factor

    def initialize(self, image_paths):
        super(MarkerDataset, self).initialize(image_paths, marker_config.CLASS_NAMES, 'celldom-marker')

    def load_mask(self, image_id):
        # Masks load as 3D array w/ shape (h, w, n_mask)
        mask, class_ids = super(MarkerDataset, self).load_mask(image_id)
        if self.reflect_images:
            mask = mask[:, ::-1, :]
        return mask, class_ids

    def load_image(self, image_id):
        image = super(MarkerDataset, self).load_image(image_id)

        # At TOW, only 8 or 16 bit images are expected
        if image.dtype != np.uint16 and image.dtype != np.uint8:
            raise ValueError(
                'Image {} has type {} but only 8-bit and 16-bit are supported'
                .format(image_id, image.dtype)
            )

        # Resize image at original bit depth before applying type conversions
        if self.scale_factor is not None:
            dtype = image.dtype
            image = transform.rescale(
                image, (self.scale_factor,)*2, mode='constant',
                anti_aliasing=True, multichannel=True
            )
            # Convert from 0-1 float64 back to original type
            image = exposure.rescale_intensity(image, in_range=(0, 1), out_range=str(dtype)).astype(dtype)

        # Rescale to uint8 if necessary as MRCNN expects this (silently)
        if image.dtype == np.uint16:
            image = exposure.rescale_intensity(image, out_range=np.uint8).astype(np.uint8)

        if self.reflect_images:
            image = image[:, ::-1, :]

        return image
