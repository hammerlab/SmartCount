from celldom.config import rcnn_config
from celldom.constant import CLASS_NAME_CELL
import os

CLASS_NAMES = [CLASS_NAME_CELL]
CLASS_INDEX_CELL = 1


class CellTrainingConfig(rcnn_config.CelldomTrainingConfig):
    """Configuration for cell identification Mask-RCNN model training"""

    # Give the configuration a recognizable name
    NAME = "celldom-cell"

    # Raw chip images seen so far all have shape (1024, 1376) and 1408 is
    # used here as it is already divisible by 2 at least 6 times (per MRCNN requirements).
    # Note that is crucial that IMAGE_MIN_DIM be 1024 here to ensure that the images
    # are not actually resized -- only padded in both directions to 1408x1408
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 320
    IMAGE_MAX_DIM = 320

    DETECTION_MAX_INSTANCES = 250

    # Suggestions per: https://github.com/matterport/Mask_RCNN/issues/587
    MAX_GT_INSTANCES = 250

    # Suggestions per: https://github.com/matterport/Mask_RCNN/issues/498
    TRAIN_ROIS_PER_IMAGE = 200  # 200

    # Number of classes (including background)
    NUM_CLASSES = 1 + len(CLASS_NAMES)


class CellInferenceConfig(CellTrainingConfig):
    """Configuration for marker Mask-RCNN model inference"""

    GPU_COUNT = rcnn_config.get_num_gpus_inference()
    IMAGES_PER_GPU = rcnn_config.get_num_images_per_gpu_inference()

    # Matching to https://github.com/matterport/Mask_RCNN/blob/4129a27275c48c672f6fd8c6303a88ba1eed643b/
    # samples/nucleus/nucleus.py
    POST_NMS_ROIS_INFERENCE = 2000
    DETECTION_MIN_CONFIDENCE = 0
