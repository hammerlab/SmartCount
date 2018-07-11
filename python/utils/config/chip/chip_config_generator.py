""" Utility CLI for generating chip configurations

Example:

```
# Environment Initialization:
# conda create -n celldom python=3.6
# source activate celldom
# echo "$CELLDOM_REPO_DIR/python/source" > $(python -m site --user-site)/local.pth
# pip install fire pyyaml pandas numpy

export CELLDOM_REPO_DIR=$HOME/repos/hammer/celldom
export CELLDOM_DATA_DIR=/tmp/celldom
cd $CELLDOM_REPO_DIR/python/utils/config/chip
python chip_config_generator.py convert-annotations-to-config \
--annotation-csv=annotations/chip_01.csv \
--chip-name='chip_01' \
--save-result=False > /tmp/chip_01.yaml
```
"""
#!/usr/bin/env python3
import pandas as pd
import fire
import celldom
import io
import json
import yaml
import os.path as osp

TEMPLATE_CHIP = 'chip_01'
DIGIT_PAD = 3


class CLI(object):

    def convert_annotations_to_config(self, annotation_csv, chip_name, save_result=False):
        """Create a 'Chip' configuration from a VIA annotations export file (as csv)

        Args:
            annotation_csv: Path to exported annotations; must contain bounding box and point annotations necessary
                to create a chip configuration (see utils/config/chip/README.md for more details)
            chip_name: Name of chip to be used; this should be alphanumeric snake-case (e.g. "chip_09");
                Note that if `save_result` is true, then this name will be used as the file name
            save_result: If false, the new configuration is printed to stdout for redirection; if true, then the
                resulting configuration is saved at $CELLDOM_REPO_DIR/config/chip/${chip_name}.yaml
        """
        # Read in template configuration and set new name
        temp_config = celldom.get_config('chip', TEMPLATE_CHIP)
        temp_config['name'] = chip_name

        # Read in annotations, to be used to overwrite configuration properties in template
        annot = pd.read_csv(annotation_csv)

        # Convert json fields from VIA back to json from strings
        annot['region_attributes'] = annot['region_attributes'].apply(json.loads)
        annot['region_shape_attributes'] = annot['region_shape_attributes'].apply(json.loads)

        # Get type of each region (e.g. "chip_border", "st_num_border", etc.)
        annot['reg_type'] = annot['region_attributes'].apply(lambda v: v['type'])

        if annot['reg_type'].value_counts().max() > 1:
            raise ValueError(
                'Can only have one annotation per type, but given csv has multiple; Counts = \n{}'
                .format(annot['reg_type'].value_counts())
            )

        # Extract map of shape types to shape descriptions
        shapes = annot.set_index('reg_type')['region_shape_attributes'].to_dict()

        def assert_type(ot, at):
            if shapes[ot]['name'] != at:
                raise ValueError(
                    'Expecting annotation type "{}" for object {} but found {}',
                    at, ot, shapes[ot]['name']
                )

        # Validate presence of necessary annotated classes
        req_types = [
            'chip_border', 'apt_num_border', 'st_num_border',
            'st_num_center_digit', 'apt_num_center_digit',
            'marker_center'
        ]
        for t in req_types:
            if t not in shapes:
                raise ValueError(
                    'Failed to find required annotation type "{}" in csv {}'
                    .format(t, annotation_csv)
                )
            assert_type(t, 'rect' if t != 'marker_center' else 'point')

        ##########################################
        # Create bbox margins for entire apartment
        ##########################################

        # {'name': 'rect', 'x': 6, 'y': 3, 'width': 362, 'height': 455}
        cb = shapes['chip_border']
        # {'name': 'point', 'cx': 256, 'cy': 393}
        mc = shapes['marker_center']

        temp_config['apt_margins'] = dict(
            left=-(mc['cx'] - cb['x']),
            right=cb['x'] + cb['width'] - mc['cx'],
            bottom=cb['y'] + cb['height'] - mc['cy'],
            top=-(mc['cy'] - cb['y'])
        )

        ###################################
        # Create offsets for st/apt numbers
        ###################################
        def get_margins(mc, bbox):
            return dict(
                left=bbox['x'] - mc['cx'],
                right=(bbox['x'] + bbox['width']) - mc['cx'],
                bottom=(bbox['y'] + bbox['height']) - mc['cy'],
                top=bbox['y'] - mc['cy']
            )

        # shapes['apt_num_border']: {'name': 'rect', 'x': 261, 'y': 62, 'width': 104, 'height': 58}
        temp_config['apt_num_margins'] = get_margins(mc, shapes['apt_num_border'])

        # shapes['st_num_border']: {'name': 'rect', 'x': 21, 'y': 328, 'width': 107, 'height': 61}
        temp_config['st_num_margins'] = get_margins(mc, shapes['st_num_border'])

        #################################
        # Create individual digit offsets
        #################################
        def get_digit_bounds(bbout, bbin, pad=DIGIT_PAD):
            w = bbin['width']
            c1 = bbin['x'] - bbout['x']
            c2 = c1 + w
            return [
                [max(c1 - w - pad, 0), c1 + pad],
                [c1 - pad, c1 + w + pad],
                [c2 - pad, min(c2 + w + pad, bbout['width'])]
            ]

        temp_config['apt_num_digit_bounds'] = get_digit_bounds(shapes['apt_num_border'], shapes['apt_num_center_digit'])
        temp_config['st_num_digit_bounds'] = get_digit_bounds(shapes['st_num_border'], shapes['st_num_center_digit'])

        #######################
        # Save or print results
        #######################

        # If save selected, save to static path in repo
        if save_result:
            exp_path = osp.join(celldom.get_repo_dir(), 'config', 'chip', chip_name + '.yaml')
            with open(exp_path, 'w') as fd:
                yaml.dump(temp_config, fd)
            print('Resulting configuration saved to "{}"'.format(exp_path))
        # Otherwise, print to stdout for redirection
        else:
            sio = io.StringIO()
            yaml.dump(temp_config, sio)
            print(sio.getvalue())


if __name__ == '__main__':
    fire.Fire(CLI)

