from collections import namedtuple

# Struct to contain flags corresponding to images/data that should be preserved in processing
# Note: These names must stay in sync with names used in apartment_extraction records
DataPersistenceFlags = namedtuple('DataPersistenceFlags', [
    'apt_image', 'apt_num_image', 'apt_num_digit_images',
    'st_num_image', 'st_num_digit_images', 'cell_image',
    'cell_coords', 'raw_norm_image'
])
NO_IMAGES = DataPersistenceFlags(
    apt_image=False, apt_num_image=False, apt_num_digit_images=False,
    st_num_image=False, st_num_digit_images=False, cell_image=False,
    cell_coords=False, raw_norm_image=False
)
ALL_IMAGES = DataPersistenceFlags(
    apt_image=True, apt_num_image=True, apt_num_digit_images=True,
    st_num_image=True, st_num_digit_images=True, cell_image=True,
    cell_coords=True, raw_norm_image=True
)
APT_IMAGES = DataPersistenceFlags(
    apt_image=True, apt_num_image=False, apt_num_digit_images=False,
    st_num_image=False, st_num_digit_images=False, cell_image=False,
    cell_coords=False, raw_norm_image=False
)
