import copy
import hashlib
import numpy as np
import pandas as pd
from celldom.constant import ACQ_TYPE_BF, ACQ_TYPE_BFFL, ACQ_CH_TYPE_BF, ACQ_CH_TYPE_FL
from celldom.dataset import marker_dataset
from celldom.execute import analysis
from celldom.utils import assert_rgb, rgb2gray


def collapse_acquisitions(paths, config, min_gap_seconds=30 * 60, max_invalid_acquisitions=0, return_df=False):
    """Converts image paths to acquisition objects based on heuristics for grouping multi-channel images

    Args:
        paths: List of file paths to analyze/group
        config: Experiment configuration
        min_gap_seconds: Minimum time in seconds between separate acquisitions (for the same apartment); this can
            also be interpreted as the maximum amount of time between the collection of the earliest channel image
            and the latest channel image (default is 30 minutes)
        max_invalid_acquisitions: Maximum number of acquisitions allowed to have an invalid set of associated
            channels (e.g. one acquisition could have only 2 channel images when 3 are expected)
        return_df: Return data frame containing acquisition field rather than list of acquisitions (useful
            for debugging and reporting)
    Returns:
        List of Acquisition instances (or unfiltered DataFrame with at least 'acq' field if `return_df` True)
    """
    df = pd.DataFrame([config.parse_path(p) for p in paths])
    df['path'] = paths

    # Get channel names excepted to be found in each grouping of image files
    channels = sorted([c['name'] for c in config.acquisition_channels])

    # Split cols into two sets based on whether or not they should differ between
    # paths to images for the same acquisition
    img_cols = ['datetime', 'channel', 'path']
    non_img_cols = [c for c in df if c not in img_cols]

    def infer_groups(g):
        gf = g[img_cols].copy()

        # Get series with actual date index and values corresponding to min date of group
        date_groups = analysis.get_date_groups(g['datetime'], min_gap_seconds=min_gap_seconds)

        # Map true dates to groups (where values are effectively the cluster id (as a datetime))
        gf['group'] = gf['datetime'].map(date_groups)
        assert gf['group'].notnull().all()

        # Group by date group and create individual acquisition instances
        acqs = []
        for k, gfa in gf.groupby('group'):
            acq_channels = sorted(gfa['channel'].tolist())
            valid = acq_channels == channels
            paths = gfa['path'].tolist()
            acqs.append(dict(
                paths=paths,
                channels=acq_channels,
                valid=valid,
                acq=None if not valid else Acquisition(config, paths=paths)
            ))
        return pd.DataFrame(acqs)

    # Group files by all properties except for those excepted to differ across files
    # in the same acquisition
    df = df.groupby(non_img_cols).apply(infer_groups)

    # Raise if number of invalid acquisitions exceeds threshold
    is_valid = df['valid'].values
    n_invalid = (~is_valid).sum()
    if n_invalid > max_invalid_acquisitions:
        with pd.option_context('display.expand_frame_repr', False, 'display.max_colwidth', 10000):
            raise ValueError(
                'Number of invalid acquisitions ({}) exceeds max of {}.  Invalid acquisitions found:\n{}'
                .format(n_invalid, max_invalid_acquisitions, df.loc[~is_valid].drop('valid', axis=1))
            )

    return df if return_df else df.loc[is_valid]['acq'].tolist()


def mrcnn_loader(path, channel, reflect_images, scale_factor, **kwargs):
    # Load image using marker dataset class, which always returns 3-channel uint8 RGB images
    dataset = marker_dataset.MarkerDataset(reflect_images=reflect_images, scale_factor=scale_factor, **kwargs)
    dataset.initialize([path])
    dataset.prepare()
    img = dataset.load_image(0)

    # If channel is BF, return as-is
    if channel['type'] == ACQ_CH_TYPE_BF:
        assert_rgb(img)
        return img
    # If channel is fluorescent, return uint8 2D image
    # NOTE: this may need to expand in the future to support landmark images based on fluorescent images
    # but for now this will do
    elif channel['type'] == ACQ_CH_TYPE_FL:
        # Convert 3-channel RGB to 2D uint8
        img = rgb2gray(img)
        assert img.ndim == 2
        return img
    else:
        raise NotImplementedError('Channels of type "{}" not yet supported'.format(channel['type']))


DEFAULT_LOADER = mrcnn_loader


class Acquisition(object):

    def __init__(self, config, paths, properties=None, loader=DEFAULT_LOADER):
        """ Acquisition model for single or multi-channel imaging datasets

        Args:
            config: Experiment configuration
            paths: Single path (as string) or list/sequence of string paths (for multi-channel images to be treated
                as a single image)
            properties: Metadata to add to all properties inferred based on paths
            loader: Function with signature `fn(path, channel, reflect_images, scale_factor) -> image` used to
                load individual images (this function is responsible for any raw image transformations also
                typically applied in model training)
        """
        self.config = config
        self.paths = [paths] if isinstance(paths, str) else paths
        self.loader = loader
        # This may be useful for adding in constant metadata fields that
        # should be present for all acquisitions in a dataset
        self.common_properties = properties

        self._channels = None
        self._primary = None

        if len(self.paths) == 0:
            raise ValueError('Image path list cannot be empty')

    def _generate_id(self, props):
        keys = sorted(list(props.keys()))
        key = ':'.join([str(props[k]) for k in keys])
        return hashlib.md5(key.encode('utf-8')).hexdigest()

    def is_initialized(self):
        return self._channels is not None

    def initialize(self):
        if self.is_initialized():
            return

        # Loop through paths and organize extracted metadata by channel
        channel_config = {c['name']: c for c in self.config.acquisition_channels}
        self._channels = {}
        for path in self.paths:
            p = self.config.parse_path(path)
            if 'id' in p:
                raise ValueError(
                    'Properties inferred from paths cannot include an "id" attribute '
                    '(properties = {}, path = {})'
                    .format(p, path)
                )

            # Extract required properties
            channel, datetime = p['channel'], p['datetime']

            # Generate hash associated with non-required properties that should be
            # identical for all channel images
            pgrp = self._generate_id({
                k: v for k, v in p.items()
                if k not in ['channel', 'datetime']
            })

            # Generate id/hash associated with primary channel,
            # if possible (to be associated with acquisition overall)
            pid = None
            if channel == self.config.acquisition_primary_channel_name:
                pid = self._generate_id(p)

            # Check for duplicate channels
            if channel in self._channels:
                raise ValueError(
                    'Found multiple paths associated with channel "{}" (paths = {})'
                    .format(channel, self.paths)
                )

            if channel not in channel_config:
                raise ValueError(
                    'Channel name found in path ("{}") is not in configured channel list "{}"'
                    .format(channel, list(channel_config.keys()))
                )
            config = channel_config[channel]
            self._channels[channel] = dict(datetime=datetime, path=path, props=p, config=config, group=pgrp, id=pid)

        # Check for conflicting metadata entries
        if len(set([e['group'] for e in self._channels.values()])) > 1:
            raise ValueError(
                'Found acquisition path property conflicts across shared properties for paths "{}"'
                .format(self.paths)
            )

        # Check for absent primary channel
        primary_entries = [e for e in self._channels.values() if e['id'] is not None]
        if len([primary_entries]) != 1:
            raise ValueError(
                'Failed to find unique properties associated with primary channel name "{}" (paths = {})'
                .format(self.config.acquisition_primary_channel_name, self.paths)
            )

        # Assign acquisition properties as properties for primary channel
        self._primary = dict(primary_entries[0])
        self._primary['props']['id'] = self._primary['id']

        # Finally, merge acquisition properties with common properties if any were provided
        if self.common_properties is not None:
            self._primary['props'].update(self.common_properties)

    def _load(self, path, channel):
        return self.loader(
            path,
            channel,
            reflect_images=self.config.acquisition_reflection,
            scale_factor=self.config.acquisition_scale_factor
        )

    def load_image(self):
        """Load the image associated with cell landmarks for this acquisition while accounting for pre-processing

        Pre-processing generally includes things like reflection and scaling to compensate for how
        microscope images were captured.
        """
        if not self.is_initialized():
            self.initialize()
        if self.config.acquisition_type in [ACQ_TYPE_BF, ACQ_TYPE_BFFL]:
            channel = self.config.acquisition_primary_channel_name
            return self._load(self._channels[channel]['path'], self._channels[channel]['config'])
        else:
            raise NotImplementedError('Acquisition type "{}" not yet supported'.format(self.config.acquisition_type))

    def load_expression(self):
        """Load secondary "expression" channels associated with object images

        Returns:
            (image, channels) where:
                image: Image array as (height, width, channels) where height and width match that returned by
                    landmark image available in `load_image`; Returns None if returned if acquisition
                    has no expression channels
                channels: List of channel names associated with trailing dimension of image array
        """
        if not self.is_initialized():
            self.initialize()
        if self.config.acquisition_type == ACQ_TYPE_BF:
            return None, None
        # When using expression channels and BF landmarks, stack expression channels along new axis
        # to give images with shape (rows, cols, channels)
        elif self.config.acquisition_type == ACQ_TYPE_BFFL:
            channels = self.config.acquisition_expression_channel_names
            img = np.stack([
                self._load(self._channels[c]['path'], self._channels[c]['config'])
                for c in channels
            ], axis=-1)
            assert img.ndim == 3
            return img, channels
        else:
            raise NotImplementedError('Acquisition type "{}" not yet supported'.format(self.config.acquisition_type))

    def get_primary_properties(self):
        if not self.is_initialized():
            self.initialize()
        return copy.deepcopy(self._primary['props'])

    def get_primary_path(self):
        if not self.is_initialized():
            self.initialize()
        return self._primary['path']
