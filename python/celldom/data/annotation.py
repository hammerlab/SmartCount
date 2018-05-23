import xml.etree.ElementTree
from collections import namedtuple
from skimage.draw import polygon_perimeter, polygon, ellipse
from skimage.measure import regionprops
import numpy as np

Annot = namedtuple('Annot', ['mask', 'border', 'points', 'object_type', 'bound_type', 'properties'])
Img = namedtuple('Img', ['image', 'annotations', 'file'])


def get_centroid_mask(annotation, radius=1, mask_value=np.iinfo(np.uint8).max):
    m = np.zeros_like(annotation.mask)
    centroid = np.array(annotation.properties.centroid).astype(np.int32)
    rr, cc = ellipse(centroid[0], centroid[1], radius, radius, shape=m.shape)
    m[rr, cc] = mask_value
    return m


def parse_object(o, img_shape, mask_value=np.iinfo(np.uint8).max):
    o_type = o.find('name').text

    if o.find('polygon'):
        # <polygon>
        #   <x1>122</x1>
        #   <y1>32</y1>
        #   <x2>106</x2>
        #   <y2>48</y2>
        #   ...
        # </polygon>
        bound = 'polygon'
        coords = o.find('polygon').getchildren()
        assert len(coords) % 2 == 0
        pts = []
        for i in range(0, len(coords), 2):
            xc, yc = coords[i], coords[ i +1]
            xi, xv = int(xc.tag.replace('x', '')), int(xc.text)
            yi, yv = int(yc.tag.replace('y', '')), int(yc.text)
            assert xi == yi == i // 2 + 1
            pts.append([xv, yv])
    elif o.find('bndbox'):
        # <bndbox>
        #   <xmin>245</xmin>
        #   <ymin>54</ymin>
        #   <xmax>300</xmax>
        #   <ymax>98</ymax>
        # </bndbox>
        bound = 'box'
        bb = o.find('bndbox')
        xmin, ymin, xmax, ymax = [int(bb.find(p).text) for p in ['xmin', 'ymin', 'xmax', 'ymax']]
        pts = [
            [xmin, ymin],
            [xmin, ymax],
            [xmax, ymax],
            [xmax, ymin]
        ]
    else:
        raise ValueError('Cound not determine mask shape')

    pts = np.array(pts)
    r, c = pts[: ,1], pts[: ,0]

    mask = np.zeros(img_shape, dtype=np.uint8)
    rr, cc = polygon(r, c, shape=img_shape)
    mask[rr, cc] = mask_value

    border = np.zeros(img_shape, dtype=np.uint8)
    rr, cc = polygon_perimeter(r, c, shape=img_shape)
    border[rr, cc] = mask_value

    props = regionprops(mask)
    assert len(props) == 1

    return Annot(mask, border, pts, o_type, bound, props[0])


def load_rectlabel_annotations(annot_path):
    e = xml.etree.ElementTree.parse(annot_path).getroot()
    w = int(e.find('size').find('width').text)
    h = int(e.find('size').find('height').text)
    img_shape = (h, w)
    annot = []
    for o in e.findall('object'):
        try:
            obj = parse_object(o, img_shape)
            annot.append(obj)
        except:
            print \
                ('Failed to process object {}'.format(xml.etree.ElementTree.tostring(o, encoding='utf8', method='xml')))
            raise
    return img_shape, annot


