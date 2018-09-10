

def get_apartment_image_shape(chip_config):
    margins = chip_config['apt_margins']
    return margins['bottom'] - margins['top'], margins['right'] - margins['left']
