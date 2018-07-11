""" Logging initialization for use in notebooks (%run -m celldom.nb.logging) """
import logging
import celldom
import os
console = logging.StreamHandler()
console.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:%(name)s: %(message)s'))
logger = logging.getLogger()
logger.setLevel(os.getenv(celldom.ENV_CELLDOM_NB_LOG_LEVEL, 'INFO'))
logger.addHandler(console)
