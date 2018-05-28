from mrcnn.config import Config
import os

CLASS_NAMES = ['Cell', 'StNum', 'AptNum', 'CellClump', 'Marker', 'Chamber']
ENV_NUM_GPUS_TRAIN = 'CONFIG_NUM_GPUS_TRAIN'
ENV_NUM_IMAGES_PER_GPU_TRAIN = 'CONFIG_IMAGES_PER_GPU_TRAIN'
ENV_NUM_GPUS_INFERENCE = 'CONFIG_NUM_GPUS_INFERENCE'
ENV_NUM_IMAGES_PER_GPU_INFERENCE = 'CONFIG_IMAGES_PER_GPU_INFERENCE'


class CelldomTrainingConfig(Config):
    """Configuration for training on the nucleus segmentation dataset."""
    # Give the configuration a recognizable name
    NAME = "celldom"

    # Adjust depending on your GPU memory
    GPU_COUNT = int(os.getenv(ENV_NUM_GPUS_TRAIN, 1))
    IMAGES_PER_GPU = int(os.getenv(ENV_NUM_IMAGES_PER_GPU_TRAIN, 1))

    # Number of classes (including background)
    NUM_CLASSES = 1 + len(CLASS_NAMES)

    # Number of training and validation steps per epoch
    STEPS_PER_EPOCH = 50
    VALIDATION_STEPS = 10

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet50"

    # Input image resizing
    # Square of size 512x512
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 384
    IMAGE_MAX_DIM = 384

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 200

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 250

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 250


class CelldomInferenceConfig(CelldomTrainingConfig):

    GPU_COUNT = int(os.getenv(ENV_NUM_GPUS_INFERENCE, 1))
    IMAGES_PER_GPU = int(os.getenv(ENV_NUM_IMAGES_PER_GPU_INFERENCE, 1))

