import re
import pandas as pd
import celldom


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
    def field_regex(self):
        return {k: v if isinstance(v, str) else v['regex'] for k, v in self._fields.items()}

    @property
    def path_regex(self):
        replacements = {k: '(?P<{}>{})'.format(k, v) for k, v in self.field_regex.items()}
        return self._path_format.format(**replacements)

    @property
    def path_field_names(self):
        return re.findall(r"{(\w+)}", self._path_format)

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
        return self._get_config('chip')

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




