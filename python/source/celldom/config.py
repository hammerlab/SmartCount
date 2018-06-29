import yaml
import os
import os.path as osp
import celldom


def get_chip_config(name):
    """Get a chip configuration object

    Args:
        name: Name of chip (assumed to be a filename minus extension present in CELLDOM_REPO_DIR/config/chip);
            e.g. 'chip_01'
    Return:
        Deserialized configuration object
    """
    path = osp.join(celldom.get_repo_dir(), 'config', 'chip', name + '.yaml')
    with open(path, 'r') as fd:
        return yaml.load(fd)


def get_cytometer_config(name):
    """Get a cytometer config with all paths resolved to absolute, local files

    Args:
        name: Name of cytometer (assumed to be a filename minus extension present in CELLDOM_REPO_DIR/config/cytometer);
            e.g. 'cytometer_01'
    Returns:
        Deserialized configuration object with any external path references resolved to a local cache dir
            and replace by those paths
    """





