from celldom.config import rcnn_config
from celldom.constant import CLASS_NAME_MARKER
import os

CLASS_NAMES = [CLASS_NAME_MARKER]


class MarkerTrainingConfig(rcnn_config.CelldomTrainingConfig):
    """Configuration for marker Mask-RCNN model training"""

    # Give the configuration a recognizable name
    NAME = "celldom-marker"

    # Raw chip images seen so far all have shape (1024, 1376) and 1408 is
    # used here as it is already divisible by 2 at least 6 times (per MRCNN requirements).
    # Note that is crucial that IMAGE_MIN_DIM be 1024 here to ensure that the images
    # are not actually resized -- only padded in both directions to 1408x1408
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1408

    # IMAGE_RESIZE_MODE = "crop"
    # IMAGE_MIN_DIM = 64
    # IMAGE_MAX_DIM = 1024

    DETECTION_MAX_INSTANCES = 50  # 100

    # Suggestions per: https://github.com/matterport/Mask_RCNN/issues/587
    MAX_GT_INSTANCES = 50  # 100
    RPN_ANCHOR_SCALES = (8, 16, 32)  # (32, 64, 128, 256, 512)

    # WARNING: You cannot change the number of these (there must be 3) or there will be an error
    # when initializing models about incorrect shapes
    # RPN_ANCHOR_RATIOS = [0.5, 1, 2]

    # POST_NMS_ROIS_INFERENCE = 250
    # POST_NMS_ROIS_TRAINING = 250
    # RPN_TRAIN_ANCHORS_PER_IMAGE = 250

    # Suggestions per: https://github.com/matterport/Mask_RCNN/issues/498
    TRAIN_ROIS_PER_IMAGE = 200  # 200

    # Number of classes (including background)
    NUM_CLASSES = 1 + len(CLASS_NAMES)

    EXPECTED_CLASS_COUNTS = {
        CLASS_NAME_MARKER: 1
    }


class MarkerInferenceConfig(MarkerTrainingConfig):
    """Configuration for marker Mask-RCNN model inference"""

    GPU_COUNT = rcnn_config.get_num_gpus_inference()
    IMAGES_PER_GPU = rcnn_config.get_num_images_per_gpu_inference()

