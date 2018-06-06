import os
import os.path as osp

ENV_SEED = 'SEED'
seed = int(os.getenv(ENV_SEED, 38923))

ENV_CELLDOM_VERSION = 'CELLDOM_VERSION'
ENV_CELLDOM_DATA_DIR = 'CELLDOM_DATA_DIR'
ENV_CELLDOM_DATASET_DIR = 'CELLDOM_DATASET_DIR'
ENV_CELLDOM_MODEL_DIR = 'CELLDOM_MODEL_DIR'
DEFAULT_VERSION = 'r0.2'


def get_version():
    return os.getenv(ENV_CELLDOM_VERSION, DEFAULT_VERSION)


def get_data_dir():
    """Return main directory under which all Celldom data should be stored (models, images, results, etc).

    This is controlled by the 'CELLDOM_DATA_DIR' environment variable
    """
    if not os.getenv(ENV_CELLDOM_DATA_DIR):
        raise ValueError('Must set "{}" environment variable'.format(ENV_CELLDOM_DATA_DIR))
    return os.getenv(ENV_CELLDOM_DATA_DIR)


def get_dataset_dir():
    """Return directory to contain separate dataset directories (e.g. dataset01, dataset02, etc)

    This can be overriden by 'CELLDOM_DATASET_DIR' environment variable but if not set will
    default to $CELLDOM_DATA_DIR/dataset
    """
    if os.getenv(ENV_CELLDOM_DATASET_DIR):
        return os.getenv(ENV_CELLDOM_DATASET_DIR)
    return osp.join(get_data_dir(), 'dataset')


def get_model_dir():
    """Return directory to contain separate model directories (based on application)

    This can be overriden by 'ENV_CELLDOM_MODEL_DIR' environment variable but if not set will
    default to $CELLDOM_DATA_DIR/model/$CELLDOM_VERSION
    """
    if os.getenv(ENV_CELLDOM_MODEL_DIR):
        return os.getenv(ENV_CELLDOM_MODEL_DIR)
    return osp.join(get_data_dir(), 'model', get_version())
