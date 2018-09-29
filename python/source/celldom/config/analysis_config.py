import celldom


class AnalysisConfig(object):

    def __init__(self, conf):
        # Load configuration if provided as path instead of deserialized object
        if isinstance(conf, str):
            conf = celldom.read_config(conf)
        self.conf = conf

    def __getitem__(self, key):
        return self.conf[key]

    def __contains__(self, key):
        return key in self.conf


def get_analysis_config_by_name(name):
    """Load analysis configuration by name
    Args:
        name: Name of file with known experiment configurations folder
    Returns:
        AnalysisConfig object
    """
    return AnalysisConfig(celldom.get_config('analysis', name))
