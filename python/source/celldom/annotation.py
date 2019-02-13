import os
import os.path as osp
from shutil import copyfile
from scipy import spatial
from collections import OrderedDict
from cvutils.rectlabel import io as rl_io
from celldom.execute import processing


def get_cell_rectlabel_xml_object(cell):
    """Convert cell detection object (from cell_extraction) to rectlabel compatible object"""
    hull = spatial.ConvexHull(cell['coords'])
    polygon = cell['coords'][hull.vertices]
    coords = [dict(x=polygon[i, 1], y=polygon[i, 0]) for i in range(polygon.shape[0])]
    return {'name': 'Cell', 'coords': coords}


def get_image_rectlabel_xml(image_path, image_shape, cells):
    """Convert cell detections for an image to rectlabel annotation XML

    This is primarily intended to be used to process detections from ``celldom.execute.processing.run_cell_detection``
    """
    return rl_io.get_annotation_xml(image_path, image_shape, [get_cell_rectlabel_xml_object(c) for c in cells])


def save_image_rectlabel_annotations(output_dir, docs, copy=False):
    """Write RectLabel annotation xml docs to directory

    This is useful for building training datasets bootstrapped from previously trained models

    Args:
        output_dir: Directory to contain image and annotation files
        docs: Dict of RectLabel xml documents (as strings) with keys equal to path for original image
        copy: Copy original image files into `output_dir` along with annotation xml files (default False)
    """
    if not osp.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    for path, doc in docs.items():
        output_path = osp.join(output_dir, osp.basename(path))

        # Copy the original image if necessary
        if copy:
            copyfile(path, output_path)

        # Write the corresponding xml doc for the image
        annot_path = rl_io.get_annotation_path(output_path)
        if not osp.exists(osp.dirname(annot_path)):
            os.makedirs(osp.dirname(annot_path), exist_ok=True)
        with open(annot_path, 'w') as fd:
            fd.write(doc)


def generate(files, exp_config, type='rectlabel', **kwargs):
    if type != 'rectlabel':
        raise ValueError('Currently only RectLabel annotations are supported')
    res = list(processing.run_cell_detection(exp_config, files, **kwargs))

    # Convert cell objects to RectLabel annotation files (one XML doc per given image file)
    docs = [get_image_rectlabel_xml(r[0], r[1], r[2]) for r in res]
    if len(docs) != len(files):
        raise AssertionError('Expecting {} results but got {} from xml generator'.format(len(files), len(docs)))
    return OrderedDict(zip(files, docs))


def export(docs, output_dir, type='rectlabel', copy=False):
    if type != 'rectlabel':
        raise ValueError('Currently only RectLabel annotations are supported')
    save_image_rectlabel_annotations(output_dir, docs, copy=copy)
