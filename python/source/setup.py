import os.path as osp
import logging
from setuptools import setup, find_packages

readme_path = osp.join(osp.dirname(__file__), 'README.md')

try:
    with open(readme_path, 'r') as f:
        readme_markdown = f.read()
except:
    logging.warning("Failed to load %s" % readme_path)
    readme_markdown = ""


if __name__ == '__main__':
    setup(
        name='celldom',
        version='0.0.1',
        description="Celldom Array Image Processing",
        author="Eric Czech",
        author_email="eric@hammerlab.org",
        url="",
        license="http://www.apache.org/licenses/LICENSE-2.0.html",
        classifiers=[
            'Environment :: Console',
            'Operating System :: OS Independent',
            'Intended Audience :: Science/Research',
            'Topic :: Scientific/Engineering :: Bio-Informatics',
            'License :: OSI Approved :: Apache Software License',
            'Programming Language :: Python :: 3.6'
        ],
        install_requires=[
            'keras==2.1.6',
            'scikit-image==0.14.0',
            'imageio==2.4.1',
            'jupyterlab',
            'plotly',
            'plotnine',
            'Cython',
            'papermill',
            'opencv-python',
            'tables',
            'retrying',
            'fire',
            'tifffile',
            'umap-learn',
            'moviepy',
            'Pillow',
            'h5py'
        ],
        dependency_links=[
            'git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI'
            'git+https://github.com/matterport/Mask_RCNN.git@v2.1#egg=maskrcnn-2.1'
        ],
        extras_require={
            'tf': ['tensorflow>=1.7.0'],
            'tf_gpu': ['tensorflow-gpu>=1.7.0'],
            'training': ['pycocotools', 'maskrcnn', 'imgaug'],
            'app': [
                'dash==0.36.0',
                'dash-html-components==0.13.5',
                'dash-core-components==0.43.0',
                'dash-table==3.1.11',
            ]
        },
        long_description=readme_markdown,
        long_description_content_type='text/markdown',
        packages=find_packages(exclude=('tests',)),
        package_data={},
        include_package_data=True,
        entry_points={'console_scripts': ['celldom = celldom.cli.celldom_cli:main']}
    )
