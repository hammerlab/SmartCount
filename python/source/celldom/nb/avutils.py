"""Video generation and embedding utilities"""
import os
import os.path as osp


def make_video(images, tmp_dir=None):
    """Create an mp4 video from a list of images

    Args:
        images: List of images (must have same size)
        tmp_dir: Working directory; will be created if not specified
    Returns:
        Path to generated video
    """
    from skimage import io as sk_io
    import tempfile

    if tmp_dir is None:
        tmp_dir = tempfile.mkdtemp()
    for i, img in enumerate(images):
        sk_io.imsave(osp.join(tmp_dir, 'img_{:03d}.png'.format(i)), img)
    cmd = 'avconv -framerate 1 -f image2 -i "{}/img_%3d.png" -y {}/video.mp4'.format(tmp_dir, tmp_dir)
    os.system(cmd)
    return osp.join(tmp_dir, 'video.mp4')


def embed_video(path):
    """Return HTML tag for embedding videos in notebooks

    Args:
        path: Path to mp4 file
    Returns:
        HTML element (from IPython.display)
    """
    import io as pyio
    import base64
    from IPython.display import HTML
    video = pyio.open(path, 'r+b').read()
    encoded = base64.b64encode(video)
    return HTML(data='''<video alt="test" controls>
                <source src="data:video/mp4;base64,{0}" type="video/mp4" /></video>'''
                .format(encoded.decode('ascii')))