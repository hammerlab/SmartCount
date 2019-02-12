from cvutils.rectlabel import io as rectlabel_io
import pandas as pd


def get_data_files(data_dir, sort_by='image_name', **kwargs):
    """Fetch data frame containing basic information on image files and associated annotations"""
    df = pd.DataFrame(rectlabel_io.list_dir(data_dir, **kwargs))

    # Sort by image name by default
    if len(df > 0):
        df = df.sort_values(sort_by)

    return df
