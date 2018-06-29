import os
import os.path as osp

ENV_SEED = 'SEED'
seed = int(os.getenv(ENV_SEED, 38923))

ENV_CELLDOM_VERSION = 'CELLDOM_VERSION'
ENV_CELLDOM_DATA_DIR = 'CELLDOM_DATA_DIR'
ENV_CELLDOM_DATASET_DIR = 'CELLDOM_DATASET_DIR'
ENV_CELLDOM_MODEL_DIR = 'CELLDOM_MODEL_DIR'
ENV_CELLDOM_REPO_DIR = 'CELLDOM_REPO_DIR'
ENV_CELLDOM_CACHE_DIR = 'CELLDOM_CACHE_DIR'
ENV_CELLDOM_MAX_DOWNLOAD_ATTEMPTS = 'CELLDOM_MAX_DOWNLOAD_ATTEMPTS'
ENV_CELLDOM_GPU_MEMORY_FRACTION = 'CELLDOM_GPU_MEMORY_FRACTION'
ENV_CELLDOM_NB_LOG_LEVEL = 'CELLDOM_NB_LOG_LEVEL'
ENV_CELLDOM_DEFAULT_CHIP_NAME = 'CELLDOM_DEFAULT_CHIP_NAME'

DEFAULT_VERSION = 'r0.2'
DEFAULT_CHIP_NAME = 'chip_01'


def get_version():
    return os.getenv(ENV_CELLDOM_VERSION, DEFAULT_VERSION)


########################
# Directory Resolution #
########################

def get_data_dir():
    """Return main directory under which all Celldom data should be stored (models, images, results, etc).

    This is controlled by the 'CELLDOM_DATA_DIR' environment variable
    """
    if not os.getenv(ENV_CELLDOM_DATA_DIR):
        raise ValueError('Must set "{}" environment variable'.format(ENV_CELLDOM_DATA_DIR))
    return os.getenv(ENV_CELLDOM_DATA_DIR)


def get_cache_dir():
    """Return semi-temporary cache directory for load-on-demand files

    This is controlled by the 'CELLDOM_CACHE_DIR' environment variable and will default to
    CELLDOM_DATA_DIR/.celldom/cache otherwise
    """
    if os.getenv(ENV_CELLDOM_CACHE_DIR):
        return os.getenv(ENV_CELLDOM_CACHE_DIR)
    path = osp.join(get_data_dir(), '.celldom', 'cache')
    if not osp.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


def get_dataset_dir():
    """Return directory to contain separate dataset directories (e.g. dataset01, dataset02, etc)

    This can be overriden by 'CELLDOM_DATASET_DIR' environment variable but if not set will
    default to $CELLDOM_DATA_DIR/dataset
    """
    return os.getenv(ENV_CELLDOM_DATASET_DIR, osp.join(get_data_dir(), 'dataset'))


def get_model_dir():
    """Return directory to contain separate model directories (based on application)

    This can be overriden by 'ENV_CELLDOM_MODEL_DIR' environment variable but if not set will
    default to $CELLDOM_DATA_DIR/model/$CELLDOM_VERSION
    """
    return os.getenv(ENV_CELLDOM_MODEL_DIR, osp.join(get_data_dir(), 'model', get_version()))


def get_repo_dir():
    """Return root directory to celldom repository location

    This can be overriden by 'ENV_CELLDOM_REPO_DIR' environment variable
    """
    if not os.getenv(ENV_CELLDOM_REPO_DIR):
        raise ValueError('Must set "{}" environment variable'.format(ENV_CELLDOM_REPO_DIR))
    return os.getenv(ENV_CELLDOM_REPO_DIR)


#############################
# Application Configuration #
#############################

def _get_config(typ, name):
    return read_config(osp.join(celldom.get_repo_dir(), 'config', typ, name + '.yaml'))


def read_config(path):
    with open(path, 'r') as fd:
        return yaml.load(fd)


def get_default_chip_config():
    return get_chip_config(os.getenv(ENV_CELLDOM_DEFAULT_CHIP_NAME, DEFAULT_CHIP_NAME))


def get_chip_config(name):
    """Get a chip configuration object

    Args:
        name: Name of chip (assumed to be a filename minus extension present in CELLDOM_REPO_DIR/config/chip);
            e.g. 'chip_01'
    Return:
        Deserialized configuration object
    """
    return _get_config('chip', name)


def get_cytometer_config(name):
    """Get a cytometer configuration object

    Args:
        name: Name of cytometer (assumed to be a filename minus extension present in CELLDOM_REPO_DIR/config/cytometer);
            e.g. 'cytometer_01'
    Returns:
        Deserialized configuration object
    """
    return _get_config('cytometer', name)


###############################
# Tensorflow/Keras Management #
###############################

def get_gpu_memory_fraction():
    return float(os.getenv(ENV_CELLDOM_GPU_MEMORY_FRACTION, .95))


def get_tf_session_config():
    import tensorflow as tf
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=get_gpu_memory_fraction())
    config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    return config


def initialize_keras_session():
    import keras.backend.tensorflow_backend as KTF
    import tensorflow as tf
    tf_session_conf = get_tf_session_config()
    KTF.set_session(tf.Session(config=tf_session_conf))
    return tf_session_conf


############
# Preloads #
############

# Force the cvutils mrcnn model module to use this path for loadining pretrained mrcnn
# models if not already set in environment
if 'MASK_RCNN_CACHE_DIR' not in os.environ:
    os.environ['MASK_RCNN_CACHE_DIR'] = osp.join(get_cache_dir(), 'models', 'pretrained', 'mrcnn')