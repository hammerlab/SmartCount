import tqdm
import celldom
from celldom.core import cytometry
from celldom.extract import APT_IMAGES
from celldom.exception import NoMarkerException
import logging

logger = logging.getLogger(__name__)

MAX_PROC_FAILURES = 10
MAX_FILES_IN_MEM_RES = 500


def run_cytometer(exp_config, output_dir, files, max_failures=MAX_PROC_FAILURES,
                  return_results=False, dpf=APT_IMAGES, **kwargs):

    if return_results and len(files) > MAX_FILES_IN_MEM_RES:
        raise ValueError(
            'Processing cannot be run with `return_results=True` and a large number of input data files '
            '(given {} files and threshold for this error is {}).  Either set return_results=False '
            '(in which case data is saved on disk) or specify smaller batches of files yourself '
            'and deal with trimming results in memory'.format(len(files), MAX_FILES_IN_MEM_RES)
        )

    results = []
    with cytometry.Cytometer(exp_config, output_dir, **kwargs) as cytometer:
        n_fail = 0
        for i, f in tqdm.tqdm(enumerate(files), total=len(files)):
            try:
                logger.debug('Processing file "%s"', f)

                # Specify an "Acquisition", which exists to also make it possible
                # to associate custom metadata with records as well as standardize
                # image pre-processing
                acq = cytometry.Acquisition(exp_config, f)

                # Analyze the image
                result = cytometer.analyze(acq, dpf=dpf)

                if return_results:
                    results.append(result)

                # Save the results
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
