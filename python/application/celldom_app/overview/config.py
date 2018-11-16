import os
import os.path as osp
import celldom
from celldom.config import experiment_config
from celldom.core.experiment import Experiment
from celldom.execute import view

CONFIG = None

ENV_APP_PORT = 'APP_PORT'
ENV_APP_HOST_IP = 'APP_HOST_IP'
ENV_APP_APT_IMG_HEIGHT_PX = 'APP_APT_IMG_HEIGHT_PX'
ENV_APP_REMOVE_OOB_ADDRESS = 'APP_REMOVE_OOB_ADDRESS'
ENV_APP_ARRAY_CELL_COUNT_FILL = 'APP_ARRAY_CELL_COUNT_FILL'
ENV_APP_FORCE_VIEW_CALCULATION = 'APP_FORCE_VIEW_CALCULATION'

DEFAULT_APP_HOST_IP = '0.0.0.0'
DEFAULT_APP_PORT = 8050
DEFAULT_APP_APT_IMG_HEIGHT_PX = '350'
DEFAULT_APP_ARRAY_CELL_COUNT_FILL = '-1'


class AppConfig(object):

    def __init__(self, exp):
        self.exp = exp
        self.exp_config = exp.config
        self.exp_data_dir = exp.data_dir
        self.analysis_config = exp.config.get_analysis_config()
        self.experimental_condition_fields = self.exp_config.experimental_condition_fields

    @property
    def app_port(self):
        return int(os.getenv(ENV_APP_PORT, DEFAULT_APP_PORT))

    @property
    def app_host_ip(self):
        return os.getenv(ENV_APP_HOST_IP, DEFAULT_APP_HOST_IP)

    @property
    def apartment_image_height(self):
        return os.getenv(ENV_APP_APT_IMG_HEIGHT_PX, DEFAULT_APP_APT_IMG_HEIGHT_PX) + 'px'

    @property
    def growth_rate_range(self):
        return [-2, 2]

    @property
    def max_table_rows(self):
        return 8

    @property
    def remove_oob_address(self):
        return os.getenv(ENV_APP_REMOVE_OOB_ADDRESS, 'true').lower() == 'true'

    @property
    def array_cell_count_fill(self):
        value = os.getenv(ENV_APP_ARRAY_CELL_COUNT_FILL, DEFAULT_APP_ARRAY_CELL_COUNT_FILL)
        return None if value.lower() == 'none' else float(value)

    @property
    def force_view_calculation(self):
        return os.getenv(ENV_APP_FORCE_VIEW_CALCULATION, 'false').lower() == 'true'

    @property
    def app_output_dir(self):
        return osp.join(self.exp_data_dir, 'app', 'overview')


def initialize(exp_config_path, output_dir):
    global CONFIG
    exp_config = experiment_config.ExperimentConfig(celldom.read_config(exp_config_path))
    CONFIG = AppConfig(Experiment(exp_config, output_dir))


def get():
    return CONFIG
