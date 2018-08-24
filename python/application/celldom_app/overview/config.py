import os
import celldom
from celldom.config import experiment_config

CONFIG = None

ENV_APP_PORT = 'APP_PORT'
ENV_APP_HOST_IP = 'APP_HOST_IP'
ENV_APP_APT_IMG_HEIGHT_PX = 'APP_APT_IMG_HEIGHT_PX'

DEFAULT_APP_HOST_IP = '0.0.0.0'
DEFAULT_APP_PORT = 6006
DEFAULT_APP_APT_IMG_HEIGHT_PX = '350'


class AppConfig(object):

    def __init__(self, exp_config):
        self.exp_config = exp_config

    @property
    def app_port(self):
        return int(os.getenv(ENV_APP_PORT, DEFAULT_APP_PORT))

    @property
    def app_host_ip(self):
        return os.getenv(ENV_APP_HOST_IP, DEFAULT_APP_HOST_IP)

    @property
    def enabled_cached_data(self):
        return True

    @property
    def experimental_condition_fields(self):
        # pass-through property for brevity
        return self.exp_config.experimental_condition_fields

    @property
    def apartment_image_height(self):
        return os.getenv(ENV_APP_APT_IMG_HEIGHT_PX, DEFAULT_APP_APT_IMG_HEIGHT_PX) + 'px'

    @property
    def summary_n_group_treshold(self):
        return 4

    @property
    def growth_rate_range(self):
        return [-2, 2]

    @property
    def max_table_rows(self):
        return 8


def initialize(exp_config_path):
    global CONFIG
    exp_config = experiment_config.ExperimentConfig(celldom.read_config(exp_config_path))
    CONFIG = AppConfig(exp_config)


def get():
    return CONFIG
