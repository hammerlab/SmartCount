from collections import namedtuple

# Struct to contain flags corresponding to images/data that should be preserved in processing
# Note: These names must stay in sync with names used in apartment_extraction records
DataPersistenceFlags = namedtuple('DataPersistenceFlags', [
    'apt_image', 'apt_num_image', 'apt_num_digit_images',
    'st_num_image', 'st_num_digit_images', 'cell_image',
    'acq_norm_image'
])
DPF_NONE = DataPersistenceFlags(
    apt_image=False, apt_num_image=False, apt_num_digit_images=False,
    st_num_image=False, st_num_digit_images=False, cell_image=False,
    acq_norm_image=False
)
