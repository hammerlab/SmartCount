import re
import pandas as pd
import celldom
from celldom.config import chip_config


class ExperimentConfig(object):

    def __init__(self, conf):
        self.conf = conf
        self._validate()

    @property
    def _path_format(self):
        return self.conf['metadata']['path_format']

    @property
    def _fields(self):
        return self.conf['metadata']['fields']

    @property
    def field_names(self):
        return list(self._fields.keys())

    @property
    def _datetime_conf(self):
        return self.conf['metadata']['fields']['datetime']

    @property
    def name(self):
        return self.conf['name']

    @property
    def field_regex(self):
        return {k: v if isinstance(v, str) else v['regex'] for k, v in self._fields.items()}

    @property
    def path_regex(self):
        replacements = {k: '(?P<{}>{})'.format(k, v) for k, v in self.field_regex.items()}
        return self._path_format.format(**replacements)

    @property
    def path_field_names(self):
        return re.findall(r"{(\w+)}", self._path_format)

    @property
    def acquisition_magnification(self):
        return self.conf['acquisition']['magnification']

    @property
    def acquisition_scale_factor(self):
        mag = self.acquisition_magnification
        res = None

        # At TOW, all models were built using 10x images so this function will assume
        # that anything not at 10x should be resized (assuming original magnification
        # isn't too far from 10x)
        if mag != 10:
            if mag < 8 or mag > 20:
                raise ValueError(
                    'Experiment magnification given, {}, is invalid as it is not in [8, 20] ('
                    'this is because downsampling/upsampling is unlikely to be representative of '
                    'the 10x images originally used to train learning models)'.format(mag)
                )
            res = 10. / mag
        return res

    @property
    def acquisition_reflection(self):
        return self.conf['acquisition']['reflection']

    @property
    def groupings(self):
        return self.conf.get('groupings', {})

    @property
    def experimental_condition_fields(self):
        if 'groupings' not in self.conf:
            raise ValueError('Experiment configuration does not have required property "groupings"')
        if 'experimental_conditions' not in self.conf['groupings']:
            raise ValueError(
                '"groupings" object in experiment configuration does not have '
                'required property "experimental_conditions"'
            )
        return ['acq_' + c for c in self.conf['groupings']['experimental_conditions']]

    def _get_config(self, typ):
        if typ not in self.conf:
            raise ValueError('Experiment configuration does not have required property "{}"'.format(typ))

        res = None
        if 'path' in self.conf[typ]:
            res = celldom.read_config(self.conf[typ]['path'])
        elif 'name' in self.conf[typ]:
            res = celldom.get_config(typ, self.conf[typ]['name'])
        if res is None:
            raise ValueError(
                '{} configuration "{}" does not have one of "path" or "name"'
                .format(typ.title(), self.conf[typ])
            )
        return res

    def get_cytometer_config(self):
        return self._get_config('cytometer')

    def get_chip_config(self):
        return chip_config.ChipConfig(self._get_config('chip'))

    def _validate(self):
        path_field_names = sorted(self.path_field_names)
        conf_field_names = sorted(self.field_names)

        if 'datetime' not in conf_field_names:
            raise ValueError('Required field name "datetime" not present in configuration')

        datetime_conf = self._datetime_conf
        for f in ['regex', 'format']:
            if f not in datetime_conf:
                raise ValueError(
                    'Datetime field configuration "{}" missing required field "{}"'.format(datetime_conf, f))

        if conf_field_names != path_field_names:
            raise ValueError(
                'Configuration contains specifications for fields that do not match the field'
                'placeholders in the path format.\nPath format={}\nPath placeholder names found = {}\n'
                'Field names configured = {}'
                .format(self._path_format, path_field_names, conf_field_names)
            )

        if 'acquisition' not in self.conf:
            raise ValueError('Experiment configuration must contain "acqusition" properties')
        for p in ['magnification', 'reflection']:
            if p not in self.conf['acquisition']:
                raise ValueError('Experiment configuration must contain "acquisition.{}" property'.format(p))

    def parse_path(self, path):

        # Use regular expressions for each field to parse them out of the path
        path_regex = self.path_regex
        m = re.match(path_regex, path)
        if m is None:
            raise ValueError('Failed to parse path "{}" using pattern "{}"'.format(path, path_regex))
        m = m.groupdict()

        # Validate that all fields were found
        conf_field_names = sorted(self.field_names)
        matched_field_names = sorted(list(m.keys()))

        if matched_field_names != conf_field_names:
            raise ValueError(
                'Expected fields not able to be parsed from path.\n'
                'Path = {}\n'
                'Path regex = {}\n'
                'Path fields found = {}\n'
                'Field names expected = {}'
                .format(path, path_regex, matched_field_names, conf_field_names)
            )

        # Convert parsed date to true date
        m['datetime'] = pd.to_datetime(m['datetime'], format=self._datetime_conf['format'])

        return m


def get_exp_config_by_name(name):
    """Load experiment configuration by name

    This is a convenience on this idiom, for more convenient loading
    of configurations stored within a known repo directory:

    ```
    # Long version:
    exp_configuration_path = '/repo/celldom/config/experiment/experiment_example_01.yaml'
    exp_config = experiment_config.ExperimentConfig(exp_configuration_path)

    # Short version (provided by this function):
    exp_config = experiment_config.get_experiment_configuration('experiment_example_01')
    ```

    Args:
        name: Name of file with known experiment configurations folder
    Returns:
        ExperimentConfig object
    """
    return ExperimentConfig(celldom.get_config('experiment', name))


