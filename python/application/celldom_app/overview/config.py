import os
import celldom
from celldom.config import experiment_config

CONFIG = None

ENV_APP_PORT = 'APP_PORT'
ENV_APP_HOST_IP = 'APP_HOST_IP'
ENV_APP_APT_IMG_HEIGHT_PX = 'APP_APT_IMG_HEIGHT_PX'
ENV_APP_REMOVE_OOB_ADDRESS = 'APP_REMOVE_OOB_ADDRESS'
ENV_APP_ARRAY_CELL_COUNT_FILL = 'APP_ARRAY_CELL_COUNT_FILL'
ENV_APP_FORCE_RESULT_CALCULATION = 'APP_FORCE_RESULT_CALCULATION'

DEFAULT_APP_HOST_IP = '0.0.0.0'
DEFAULT_APP_PORT = 8050
DEFAULT_APP_APT_IMG_HEIGHT_PX = '350'
DEFAULT_APP_ARRAY_CELL_COUNT_FILL = '-1'


class AppConfig(object):

    def __init__(self, exp_config):
        self.exp_config = exp_config
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
    def force_result_calculation(self):
        return os.getenv(ENV_APP_FORCE_RESULT_CALCULATION, 'false').lower() == 'true'


def initialize(exp_config_path):
    global CONFIG
    exp_config = experiment_config.ExperimentConfig(celldom.read_config(exp_config_path))
    CONFIG = AppConfig(exp_config)


def get():
    return CONFIG
