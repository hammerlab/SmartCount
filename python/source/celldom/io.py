import os
import os.path as osp
import celldom
from urllib import request
from retrying import retry
import logging

logger = logging.getLogger(__name__)

MAX_DOWNLOAD_ATTEMPTS = os.getenv(celldom.ENV_CELLDOM_MAX_DOWNLOAD_ATTEMPTS, 3)


@retry(stop_max_attempt_number=MAX_DOWNLOAD_ATTEMPTS)
def download(url, file):
    try:
        request.urlretrieve(url, file)
        return file
    except:
        logger.warning(
            'An error occurred attempting to download url "%s".  '
            'This will be tried again up to a maximum of %s times',
            url, MAX_DOWNLOAD_ATTEMPTS
        )
        raise


def cache(url, cache_path):
    path = osp.join(celldom.get_cache_dir(), cache_path)
    if not osp.exists(path):
        logger.info(
            'Downloading model file from "%s" to %s; this may take a while as these files are '
            'often large ...', url, path
        )
        os.makedirs(osp.dirname(path), exist_ok=True)
        download(url, path)
        logger.info('Download complete')
    return path
