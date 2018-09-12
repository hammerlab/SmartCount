from skimage.color import gray2rgb
from skimage.transform import resize
from skimage.exposure import rescale_intensity
import numpy as np


def prep_digit_image(img):
    # Note that resize will take care of converting from uint8 to float in 0-1, and that at TOW
    # digit classifier always expects 32 x 32 images
    assert img.dtype == np.uint8, 'Expected image of type uint8 but got {}'.format(img.dtype)

    # Convert to 2D with target height/width
    img = gray2rgb(resize(img, (32, 32), mode='constant', anti_aliasing=True)).astype(np.float32)

    # Rescale by min/max
    img = rescale_intensity(img, out_range=(0, 1))

    assert np.all(img <= 1.) and np.all(img >= 0.)
    return img


def extract_single_digits(digit_imgs, digit_model):
    preds = digit_model.predict(np.stack([prep_digit_image(img) for img in digit_imgs]))
    digits = np.argmax(preds, axis=1)
    scores = ','.join(['{:.3f}'.format(v) for v in np.max(preds, axis=1)])
    return ''.join([str(d) for d in digits]), scores

