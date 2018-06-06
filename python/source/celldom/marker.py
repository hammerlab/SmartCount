from skimage import morphology
from skimage import transform


def prep_marker_img(img):
    img = rank.median(img)
    disk = morphology.disk(radius=np.array(img.shape).min() // 3)
    disk = transform.resize(disk, img.shape, mode='constant', order=1)
    return img * disk


def get_weighted_centroid(img):
    if np.any(img < 0):
        raise ValueError('Image must be non-negative to use for weighted centroid calculation')
    row, col = np.indices(img.shape[:2])
    return np.average(row, weights=img), np.average(col, weights=img)


def get_bbox_for_mask(mask):
    ly, lx, uy, ux = list(np.argwhere(mask).min(axis=0)) + list(np.argwhere(mask).max(axis=0))
    return ly, lx, uy, ux