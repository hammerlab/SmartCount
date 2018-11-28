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

    def __setitem__(self, key, value):
        self.conf[key] = value

    def _get_mode(self, key):
        return self.conf.get(key, {}).get('mode', 'default')

    @property
    def cell_classification_mode(self):
        return self._get_mode('cell_classification')
    
    @property
    def apartment_classification_mode(self):
        return self._get_mode('apartment_classification')

    @property
    def apartment_classification_cell_class(self):
        return self.conf['apartment_classification']['cell_class']
    
    @property
    def confluence_detection_component(self):
        return self.conf['confluence_detection']['component']
    
    @property
    def confluence_detection_threshold(self):
        return self.conf['confluence_detection']['threshold']
    
    @property
    def apartment_summary_cell_class(self):
        return self.conf['apartment_summary']['cell_class']

    @property
    def apartment_summary_initial_condition(self):
        return self.conf['apartment_summary']['initial_condition']

    @property
    def growth_rate_modeling_fit_intercept(self):
        return self.conf.get('growth_rate_modeling', {}).get('fit_intercept', True)


def get_analysis_config_by_name(name):
    """Load analysis configuration by name
    Args:
        name: Name of file with known experiment configurations folder
    Returns:
        AnalysisConfig object
    """
    return AnalysisConfig(celldom.get_config('analysis', name))
