from scipy import spatial


class ChipConfig(object):

    def __init__(self, conf):
        self.conf = conf
        self.occupancy = {}

    def __getitem__(self, key):
        return self.conf[key]

    def __contains__(self, key):
        return key in self.conf

    def get_component_area(self, component):
        if component in self.occupancy:
            return self.occupancy[component]
        component_boundary = self.conf['components'][component]

        # ConvexHull returns area and volume, but in 2D area is the perimeter
        # and volume is the actual area enclosed by the hull
        self.occupancy[component] = spatial.ConvexHull(component_boundary).volume
        return self.occupancy[component]
