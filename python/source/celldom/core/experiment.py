from celldom.core import cytometry


class Experiment(object):

    def __init__(self, config, data_dir):
        self.config = config
        self.data_dir = data_dir

    def get_data_store(self):
        return cytometry.get_readonly_datastore(self.data_dir)

    def get_image_store(self):
        return cytometry.get_readonly_images(self.data_dir)

