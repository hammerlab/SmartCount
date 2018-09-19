from scipy import spatial

INNER_AREA_FRACTIONS = {'chamber': .92}


class ChipConfig(object):

    def __init__(self, conf):
        self.conf = conf
        self.areas = {}

    def __getitem__(self, key):
        return self.conf[key]

    def __contains__(self, key):
        return key in self.conf

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
