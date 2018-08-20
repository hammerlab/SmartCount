

class AppConfig(object):

    @property
    def enabled_cached_data(self):
        return True

cfg = AppConfig()