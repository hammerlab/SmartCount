import warnings


def disable_mrcnn_warnings():
    # Ignore these warnings for now as they seem to be irrelevant
    warnings.filterwarnings(
        'ignore', category=UserWarning,
        message='Converting sparse IndexedSlices to a dense Tensor of unknown shape'
    )
    warnings.filterwarnings(
        'ignore', category=UserWarning,
        message='Using a generator with `use_multiprocessing=True` and multiple workers may duplicate your data'
    )
    warnings.filterwarnings(
        'ignore', category=UserWarning,
        message='Anti-aliasing will be enabled by default in skimage 0.15 to avoid aliasing artifacts'
    )