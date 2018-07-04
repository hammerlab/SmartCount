import tqdm
import celldom
from celldom.core import cytometry
import logging

logger = logging.getLogger(__name__)

MAX_PROC_FAILURES = 10


def run_cytometer(exp_config, output_dir, files, max_failures=MAX_PROC_FAILURES):
    with cytometry.Cytometer(exp_config, output_dir) as cytometer:
        n_fail = 0
        for i, f in tqdm.tqdm(enumerate(files), total=len(files)):
            try:
                # Specify an "Acquisition", which only exists to also make it possible
                # to associate custom metadata with records
                acq = cytometry.Acquisition(f)

                # Analyze the image
                raw_data, apartment_data, cell_data = cytometer.analyze(acq)

                # Save the results
                cytometer.save(raw_data, apartment_data, cell_data)
            except Exception as e:
                n_fail += 1
                logger.error(
                    'A failure occurred processing file %s (failure threshold = %s, current failure count = %s)',
                    f, max_failures, n_fail
                )
                logger.error('Error: %s', e)
            if n_fail >= max_failures:
                logger.error('Threshold for max number of failures exceeded; skipping any further processing')
                break
