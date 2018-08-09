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
                'There can only be one annotation per type, but given csv has multiple; Counts = \n{}'
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
        for t in ['chip_border', 'marker_center']:
            if t not in shapes:
                raise ValueError(
                    'Failed to find required annotation type "{}" in csv {}'
                    .format(t, annotation_csv)
                )
            assert_type(t, 'rect' if t != 'marker_center' else 'point')

        #############################################
        # Extract and validate st/apt num annotations
        #############################################

        # Sort type names like 'st_num_1', 'st_num_2', etc.
        apt_num_types = sorted([v for v in shapes.keys() if v.startswith('apt_num_')])
        st_num_types = sorted([v for v in shapes.keys() if v.startswith('st_num_')])
        if not apt_num_types:
            raise ValueError('At least one annotation with the type "apt_num_[INDEX]" must be given')
        if not st_num_types:
            raise ValueError('At least one annotation with the type "st_num_[INDEX]" must be given')
        apt_num_bbox = [shapes[t] for t in apt_num_types]
        st_num_bbox = [shapes[t] for t in st_num_types]


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

        def get_digit_bbox(shapes):
            # Assume digit shapes are ordered left to right
            # Note: via annotations set y as increasing downward
            return dict(
                left=shapes[0]['x'],
                right=shapes[-1]['x'] + shapes[-1]['width'],
                top=min([s['y'] for s in shapes]),
                bottom=max([s['y'] + s['height'] for s in shapes])
            )

        def make_relative(mc, bbox):
            return dict(
                left=bbox['left'] - mc['cx'],
                right=bbox['right'] - mc['cx'],
                bottom=bbox['bottom'] - mc['cy'],
                top=bbox['top'] - mc['cy']
            )

        # shapes['apt_num_1']: "{""name"":""rect"",""x"":437,""y"":75,""width"":51,""height"":92}"
        apt_border_bbox = get_digit_bbox(apt_num_bbox)
        temp_config['apt_num_margins'] = make_relative(mc, apt_border_bbox)

        # shapes['st_num_1']: "{""name"":""rect"",""x"":43,""y"":283,""width"":53,""height"":91}"
        st_border_bbox = get_digit_bbox(st_num_bbox)
        temp_config['st_num_margins'] = make_relative(mc, st_border_bbox)

        #################################
        # Create individual digit offsets
        #################################
        def get_digit_bounds(bb_border, bb_digits, pad=DIGIT_PAD):
            ranges = []
            width = bb_border['right'] - bb_border['left']
            for b in bb_digits:
                left = b['x'] - bb_border['left']
                right = left + b['width']
                ranges.append([max(left - pad, 0), min(right + pad, width)])
            return ranges

        # Set left/right range for each digit as 0-based offset from left of outside border
        temp_config['apt_num_digit_bounds'] = get_digit_bounds(apt_border_bbox, apt_num_bbox)
        temp_config['st_num_digit_bounds'] = get_digit_bounds(st_border_bbox, st_num_bbox)

        # Set digit counts
        temp_config['apt_num_digit_count'] = len(apt_num_bbox)
        temp_config['st_num_digit_count'] = len(st_num_bbox)

        ##########################
        # Create component offsets
        ##########################

        for k, v in {k: v for k, v in shapes.items() if k.startswith('component_')}.items():
            comp_name = '_'.join(k.split('_')[1:])
            if v['name'] not in ['rect', 'polygon']:
                raise ValueError(
                    'Component "{}" has annotation type "{}" but only rectangular '
                    'and polygon annotations are supported'.format(k, v['name'])
                )
            if 'components' not in temp_config:
                temp_config['components'] = {}

            if v['name'] == 'rect':
                points = [
                    # Specify points clockwise
                    [v['x'], v['y']],
                    [v['x'] + v['width'], v['y']],
                    [v['x'] + v['width'], v['y'] + v['height']],
                    [v['x'], v['y'] + v['height']]
                ]
            else:
                points = list(zip(v['all_points_x'], v['all_points_y']))

            # Make all points relative to the chip border so that they can be used
            # as direct references for containment within an extracted chip image
            points = [[p[0] - cb['x'], p[1] - cb['y']] for p in points]

            temp_config['components'][comp_name] = points

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

