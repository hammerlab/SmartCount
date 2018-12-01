from scipy import spatial
from skimage import io
from celldom import utils
import numpy as np

INNER_AREA_FRACTIONS = {'chamber': .92}


class ChipConfig(object):

    def __init__(self, conf):
        self.conf = conf
        self.areas = {}
        self.template_image = None

    def __getitem__(self, key):
        return self.conf[key]

    def __contains__(self, key):
        return key in self.conf

    def __setitem__(self, key, value):
        self.conf[key] = value

    @property
    def name(self):
        return self.conf['name']

    def get_component_area(self, component):
        if component in self.areas:
            return self.areas[component]

        # If area is set explicitly for this component, use it
        if 'area' in self.conf['components'][component]:
            self.areas[component] = self.conf['components'][component]['area']
        # Otherwise, use the exterior component boundary to compute area times a somewhat magic constant
        # for specific components that better capture the area excluding physical boundaries
        # for cells within a chip (e.g. 90% is close to the ratio of the inner area of a chamber
        # over the area including the width of the chamber walls)
        else:
            component_boundary = self.conf['components'][component]['boundary']

            # ConvexHull returns area and volume, but in 2D area is the perimeter
            # and volume is the actual area enclosed by the hull
            area = spatial.ConvexHull(component_boundary).volume

            if component in INNER_AREA_FRACTIONS:
                area = area * INNER_AREA_FRACTIONS[component]

            self.areas[component] = area

        return self.areas[component]

    def get_template_image(self):
        """Return template apartment image used to generate chip configuration

        Returns:
            Single apartment image used for annotation already cropped to boundaries of apartment (to remove
            any extra margins included in the raw template image)
        """
        if self.template_image is None:
            path = self.conf['template_image_path']
            img = io.imread(path)

            # Convert to 8-bit grayscale if RGB
            if img.ndim == 3 and img.shape[2] == 3:
                img = utils.rgb2gray(img)

            # Validate that image is now 8-bit grayscale
            if img.dtype != np.uint8 or img.ndim != 2:
                raise ValueError(
                    'Template images should be of type 8-bit 2D grayscale '
                    '(given image dtype = {}, shape = {}, path = {})'
                    .format(img.dtype, img.shape, path)
                )

            # Fetch bounding box within raw template image defining apartment and crop to it
            bbox = self.conf['apt_bbox']
            self.template_image = img[bbox['top']:bbox['bottom'], bbox['left']:bbox['right']]
        return self.template_image

    def get_marker_center(self):
        """Return marker center within annotated apartment image boundaries

        Returns:
            Tuple as (row, col) containing location of marker center pixel
        """
        margins = self.conf['apt_margins']
        return -margins['top'], -margins['left']

