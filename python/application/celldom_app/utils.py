from skimage import exposure
from PIL import Image
import numpy as np
from io import BytesIO
import base64


def get_encoded_image(img):
    if img.dtype != np.uint8:
        img = exposure.rescale_intensity(img, out_range=np.uint8).astype(np.uint8)
    im = Image.fromarray(img)
    bio = BytesIO()
    im.save(bio, format='PNG')
    return base64.b64encode(bio.getvalue()).decode()
