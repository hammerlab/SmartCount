import pandas as pd
import celldom
from celldom.core import cytometry
from celldom.config import experiment_config


def get_apartment_info(experiment_config_path, output_dir, keys):
    """Get apartment data for a specific set of "keys"

    Args:
        experiment_config_path: Path to experiment configuration
            (e.g. /lab/repos/celldom/config/experiment/experiment_example_01.yaml)
        output_dir: Path to output directory; this is the `output_dir` given to `run_processor`
            (e.g. /lab/data/celldom/output/20180820-G3-full)
        keys: One or more key string(s) representing apartment address in the form experimental condition fields +
            apartment number + street number (':' delimited); Examples:
            - gravity:White:3:Control:01:70
            - gravity:Pink:3:0.5uM:27:02
            - gravity:Blue:3:Control:04:04
    """
    if isinstance(keys, str):
        keys = [keys]

    store = cytometry.get_readonly_datastore(output_dir)
    config = experiment_config.ExperimentConfig(celldom.read_config(experiment_config_path))

    df = store.get('apartment').reset_index(drop=True)
    raw_files_map = store.get('acquisition').set_index('acq_id')['raw_image_path']

    key_fields = config.experimental_condition_fields + ['apt_num', 'st_num']
    df['key'] = df[key_fields].apply(lambda r: ':'.join(r.values.astype(str)), axis=1)
    df['raw_image_path'] = df['acq_id'].map(raw_files_map)

    return df[df['key'].isin(keys)].sort_values(['key', 'acq_datetime'])
