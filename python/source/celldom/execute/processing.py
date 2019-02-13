import tqdm
import celldom
from celldom.core import cytometry
from celldom.core import acquisition
from celldom.extract import APT_IMAGES, ALL_IMAGES
from celldom.extract import cell_extraction
from celldom.exception import NoMarkerException
from skimage import io
import logging

logger = logging.getLogger(__name__)

MAX_PROC_FAILURES = 10
MAX_FILES_IN_MEM_RES = 500


def run_cytometer(exp_config, output_dir, files, max_failures=MAX_PROC_FAILURES,
                  return_results=None, sample_count=None,
                  max_invalid_acquisitions=0, dpf=APT_IMAGES, **kwargs):

    if return_results and len(files) > MAX_FILES_IN_MEM_RES:
        raise ValueError(
            'Processing cannot be run with `return_results=True` and a large number of input data files '
            '(given {} files and threshold for this error is {}).  Either set return_results=False '
            '(in which case data is saved on disk) or specify smaller batches of files yourself '
            'and deal with trimming results in memory.  Nuclear option: '
            'celldom.processing.MAX_FILES_IN_MEM_RES = float("inf")'
            .format(len(files), MAX_FILES_IN_MEM_RES)
        )
    # Default to returning results if no output directory given
    if return_results is None:
        return_results = output_dir is None

    # Group files into acquisition instances
    logger.info('Collapsing image files into acquisitions ...')
    acqs = acquisition.collapse_acquisitions(files, exp_config, max_invalid_acquisitions=max_invalid_acquisitions)
    logger.info('%s image files collapsed into %s acquisitions', len(files), len(acqs))

    # Apply sampling, if requested
    if sample_count is not None:
        if sample_count < 1:
            raise ValueError('Sample count must be >= 1 (not {})'.format(sample_count))
        logger.info('Selecting first %s acquisitions', sample_count)
        # Making this random may be useful in the future, but for now an arbitrary selection will do
        acqs = acqs[:sample_count]

    logger.info('Beginning processing of %s acquisitions ...', len(acqs))
    results = []
    with cytometry.Cytometer(exp_config, data_dir=output_dir, **kwargs) as cytometer:
        n_fail = 0
        for i, acq in tqdm.tqdm(enumerate(acqs), total=len(acqs)):
            try:
                logger.debug('Processing acquisition for path "%s"', acq.get_primary_path())

                # Analyze the image
                result = cytometer.analyze(acq, dpf=dpf)

                if return_results:
                    results.append(result)

                # Save the results
                if output_dir is not None:
                    cytometer.save(*result)

            # If there are no markers, log a more succinct message and still count this event as a failure
            except NoMarkerException:
                n_fail += 1
                logger.error(
                    'No markers found in file %s (failure threshold = %s, current failure count = %s)',
                    f, max_failures, n_fail
                )
            # Otherwise log whole trace
            except Exception:
                n_fail += 1
                logger.exception(
                    'A failure occurred processing file %s (failure threshold = %s, current failure count = %s)',
                    f, max_failures, n_fail
                )
            if n_fail >= max_failures:
                logger.error('Threshold for max number of failures exceeded; skipping any further processing')
                break

    return results if return_results else None


def run_cell_detection(exp_config, files, **kwargs):
    with cytometry.Cytometer(exp_config, data_dir=None, **kwargs) as cytometer:
        cell_model = cytometer.cell_model
        chip_config = exp_config.get_chip_config()
        for file in tqdm.tqdm(files):
            img = io.imread(file)
            result = cell_extraction.extract(img, cell_model, chip_config, dpf=ALL_IMAGES)
            result = file, img.shape, *result
            yield result
