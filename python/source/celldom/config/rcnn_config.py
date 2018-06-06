from mrcnn.config import Config
import os

ENV_NUM_GPUS_TRAIN = 'CONFIG_NUM_GPUS_TRAIN'
ENV_NUM_IMAGES_PER_GPU_TRAIN = 'CONFIG_IMAGES_PER_GPU_TRAIN'
ENV_NUM_GPUS_INFERENCE = 'CONFIG_NUM_GPUS_INFERENCE'
ENV_NUM_IMAGES_PER_GPU_INFERENCE = 'CONFIG_IMAGES_PER_GPU_INFERENCE'


def get_num_gpus_train():
    return int(os.getenv(ENV_NUM_GPUS_TRAIN, 1))


def get_num_images_per_gpu_train():
    return int(os.getenv(ENV_NUM_IMAGES_PER_GPU_TRAIN, 1))


def get_num_gpus_inference():
    return int(os.getenv(ENV_NUM_GPUS_INFERENCE, 1))


def get_num_images_per_gpu_inference():
    return int(os.getenv(ENV_NUM_IMAGES_PER_GPU_INFERENCE, 1))


class CelldomTrainingConfig(Config):
    """Standard Celldom Mask-RCNN training configuration"""

    # Give the configuration a recognizable name
    NAME = "celldom"

    # Adjust depending on your GPU memory
    GPU_COUNT = get_num_gpus_train()
    IMAGES_PER_GPU = get_num_images_per_gpu_train()

    # Number of training and validation steps per epoch
    # *Note: In order to control the exact number of training iterations to, for
    # example, match the number of training images, STEPS_PER_EPOCH should be
    # set to the number of images / IMAGES_PER_GPU (see
    # https://github.com/matterport/Mask_RCNN/blob/4129a27275c48c672f6fd8c6303a88ba1eed643b/samples/nucleus/nucleus.py
    # for an example of this)
    STEPS_PER_EPOCH = 100
    VALIDATION_STEPS = 10

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet50"

    # Input image resizing
    # Square of size 512x512
    # IMAGE_RESIZE_MODE = "square"
    # IMAGE_MIN_DIM = 384
    # IMAGE_MAX_DIM = 384

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
