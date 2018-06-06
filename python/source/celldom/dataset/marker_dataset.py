import numpy as np
from skimage.exposure import rescale_intensity
from cvutils.mrcnn import dataset as mrcnn_dataset


class MarkerDataset(mrcnn_dataset.RectLabelDataset):

    def __init__(self, reflect_images=True):
        """Dataset used to model markers in raw microscope images (multi-chamber)

        Args:
            reflect_images: Whether or not to reflect images around y-axis (i.e. reverse the columns)
        """
        super(MarkerDataset, self).__init__()
        self.reflect_images = reflect_images

    def initialize(self, image_paths, classes):
        super(MarkerDataset, self).initialize(image_paths, classes, 'celldom-marker')

    def load_mask(self, image_id):
        # Masks load as 3D array w/ shape (h, w, n_mask)
        mask, class_ids = super(MarkerDataset, self).load_mask(image_id)
        if self.reflect_images:
            mask = mask[:, ::-1, :]
        return mask, class_ids

    def load_image(self, image_id):
        image = super(MarkerDataset, self).load_image(image_id)

        # At TOW, only 8 or 16 bit images are expected
        assert image.dtype == np.uint16 or image.dtype == np.uint8

        # Rescale to uint8 if necessary as MRCNN expects this (silently)
        if image.dtype == np.uint16:
            image = rescale_intensity(image, out_range=np.uint8).astype(np.uint8)

        if self.reflect_images:
            image = image[:, ::-1, :]

        return image
