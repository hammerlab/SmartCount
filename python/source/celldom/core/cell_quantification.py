import os
import os.path as osp
import glob
import pandas as pd
import numpy as np



def get_dataset_files(dataset_path, pattern):
    """Retrieve dataset file metadata for a microscope images containing detailed naming conventions

    In this case, this function is intended to parse images with paths like this:

    2018.06.01.1710 MOLM13 1nMQuiz 1Mperml Chip2/BF_ST_000_APT_000_z_0.tif

    Args:
        dataset_path: Root directory of dataset
        pattern: Glob pattern used to match top level directories
    """
    files = glob.glob(osp.join(dataset_path, pattern))
    res = []
    for f in files:
        folder = osp.dirname(f).split(osp.sep)[-1]
        date, cells, cohort, conc, chip = folder.split()
        day = ''.join(date.split('.')[:3])
        date = date.replace('.', '')
        filename = osp.basename(f)
        address = '_'.join(filename.split('_')[:5])
        z = '_'.join(filename.split('_')[5:]).split('.')[0]
        res.append(dict(
            path=f, folder=folder, filename=filename,
            date=date, day=day, cells=cells, cohort=cohort,
            conc=conc, chip=chip, address=address, z=z
        ))
    return pd.DataFrame(res).sort_values('path')



