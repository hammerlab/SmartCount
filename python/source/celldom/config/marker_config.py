from celldom.config import rcnn_config
from celldom.constant import CLASS_NAME_MARKER
import os

CLASS_NAMES = [CLASS_NAME_MARKER]
TARGET_SIZE = 512


class MarkerTrainingConfig(rcnn_config.CelldomTrainingConfig):
    """Configuration for marker Mask-RCNN model training"""

    # Give the configuration a recognizable name
    NAME = "celldom-marker"

    # Raw chip images seen so far all have shape (1024, 1376) and resizing
    # them to 512x512 still preserves enough information for segmentation
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = TARGET_SIZE
    IMAGE_MAX_DIM = TARGET_SIZE

    # IMAGE_RESIZE_MODE = "square"
    # IMAGE_MIN_DIM = 1024
    # IMAGE_MAX_DIM = 1408

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

    POST_NMS_ROIS_TRAINING = 1000  # 2000
    POST_NMS_ROIS_INFERENCE = 500  # 1000
    # RPN_TRAIN_ANCHORS_PER_IMAGE = 250

    # Suggestions per: https://github.com/matterport/Mask_RCNN/issues/498
    TRAIN_ROIS_PER_IMAGE = 100  # 200

    # Number of classes (including background)
    NUM_CLASSES = 1 + len(CLASS_NAMES)

    EXPECTED_CLASS_COUNTS = {
        CLASS_NAME_MARKER: 1
    }


class MarkerInferenceConfig(MarkerTrainingConfig):
    """Configuration for marker Mask-RCNN model inference"""

    DETECTION_MIN_CONFIDENCE = .5
    GPU_COUNT = rcnn_config.get_num_gpus_inference()
    IMAGES_PER_GPU = rcnn_config.get_num_images_per_gpu_inference()

