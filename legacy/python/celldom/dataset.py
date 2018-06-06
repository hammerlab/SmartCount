import numpy as np
import pandas as pd
from cvutils.rectlabel import io as rectlabel_io
from cvutils.mrcnn import dataset as mrcnn_dataset


SOURCE = 'celldom'


class CelldomDataset(mrcnn_dataset.RectLabelDataset):

    def initialize(self, image_paths, classes):
        super(CelldomDataset, self).initialize(image_paths, classes, SOURCE)


def get_data_files(data_dir):
    """Fetch data frame containing basic information on image files and associated annotations"""
    df = pd.DataFrame(rectlabel_io.list_dir(data_dir))

    # Sort by image name by default
    if len(df > 0):
        df = df.sort_values('image_name')

    return df


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

        cellunion_mask = None
        if cell_class in g.groups:
            cellunion_mask = np.stack(list(g.get_group(cell_class)['mask'].values), 0)
            assert cellunion_mask.ndim == 3
            cellunion_mask = cellunion_mask.max(axis=0)

        cellclump_mask = None
        if cellclump_class in g.groups:
            cellclump_mask = g.get_group(cellclump_class)['mask'].iloc[0]

        if cellunion_mask is not None and cellclump_mask is not None:
            assert cellunion_mask.shape == cellclump_mask.shape
            stats['pct:cell_to_cellclump'] = np.logical_and(cellunion_mask,
                                                            cellclump_mask).sum() / cellclump_mask.astype(np.bool).sum()
        else:
            stats['pct:cell_to_cellclump'] = np.nan
        df.append(stats)
    return pd.DataFrame(df)
