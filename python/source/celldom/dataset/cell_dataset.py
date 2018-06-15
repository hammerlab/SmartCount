import numpy as np
import pandas as pd
from cvutils.rectlabel import io as rectlabel_io
from cvutils.mrcnn import dataset as mrcnn_dataset
from celldom.config import cell_config


class CellDataset(mrcnn_dataset.RectLabelDataset):

    def __init__(self):
        """Dataset used to model cells in apartment images"""
        super(CellDataset, self).__init__()

    def initialize(self, image_paths):
        super(CellDataset, self).initialize(image_paths, cell_config.CLASS_NAMES, 'celldom-cell')


def quantify_data_files(data_files, classes, cell_class='Cell', cellclump_class='CellClump'):
    """Compute basic summary information about images and associated annotations"""

    df = []
    for i, r in data_files.iterrows():
        stats = r.copy()
        if not r['annot_exists']:
            df.append(stats)
            continue

        annot_path = r['annot_path']
        shape, annotations = rectlabel_io.load_annotations(annot_path)
        annotations = pd.DataFrame(annotations)
        g = annotations.groupby('object_type')

        for c in classes:
            stats['ct:' + c.lower()] = g.groups.get(c, pd.Series([])).size
        df.append(stats)
    return pd.DataFrame(df)
