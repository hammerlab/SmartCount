"""Video generation and embedding utilities"""
import os
import os.path as osp
from moviepy.editor import ImageClip, concatenate_videoclips


def make_video(images, duration=1):
    """Create an mp4 video from a list of images

    Args:
        images: List of images (must have same size)
        duration: Duration of each frame in secons
    """
    clips = [ImageClip(img).set_duration(duration) for img in images]
    clips = concatenate_videoclips(clips, method="compose")
    return clips


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