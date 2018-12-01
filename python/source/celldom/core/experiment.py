from celldom.core import cytometry
from celldom.execute import view
from celldom.config import experiment_config


class Experiment(object):

    def __init__(self, config, data_dir):
        self.config = experiment_config.resolve(config)
        self.data_dir = data_dir

    def get_data_store(self):
        return cytometry.get_readonly_datastore(self.data_dir)

    def get_image_store(self):
        return cytometry.get_readonly_images(self.data_dir)

    def get_view_store(self):
        return view.get_readonly_views(self.data_dir)

    def build_views(self, force=False):
        view.build_all(self, force=force)
        return self

    def __repr__(self):
        return (
            'Name: {}\n'.format(self.config.name) +
            'Path: {}\n'.format(self.data_dir) +
            'Chip: {}\n'.format(self.config.get_chip_config().name) +
            'Analysis: {}\n'.format(self.config.get_analysis_config().name)
        )
