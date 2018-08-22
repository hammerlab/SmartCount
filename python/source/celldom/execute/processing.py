import tqdm
import celldom
from celldom.core import cytometry
from celldom.extract import NO_IMAGES
import logging

logger = logging.getLogger(__name__)

MAX_PROC_FAILURES = 10
MAX_FILES_IN_MEM_RES = 500


def run_cytometer(exp_config, output_dir, files, max_failures=MAX_PROC_FAILURES, return_results=False, dpf=NO_IMAGES):

    if return_results and len(files) > MAX_FILES_IN_MEM_RES:
        raise ValueError(
            'Processing cannot be run with `return_results=True` and a large number of input data files '
            '(given {} files and threshold for this error is {}).  Either set return_results=False '
            '(in which case data is saved on disk) or specify smaller batches of files yourself '
            'and deal with trimming results in memory'.format(len(files), MAX_FILES_IN_MEM_RES)
        )

    results = []
    with cytometry.Cytometer(exp_config, output_dir) as cytometer:
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
            except Exception:
                logger.exception(
                    'A failure occurred processing file %s (failure threshold = %s, current failure count = %s)',
                    f, max_failures, n_fail
                )
                n_fail += 1
            if n_fail >= max_failures:
                logger.error('Threshold for max number of failures exceeded; skipping any further processing')
                break

    return results if return_results else None
