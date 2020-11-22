# Celldom

Repository for collaboration on Celldom computer vision solutions

### Screencasts

- [Basic Usage](https://drive.google.com/file/d/1Z5WBaoYv3USme-7F7lqpd-Hsw0fxQXM7/view?usp=sharing) - Overview of the UI and visualization app
- [Cross Filtering](https://drive.google.com/file/d/1w0X0GpKM2B2cv5STVIl1Yth_s8OwXUw4/view?usp=sharing) - Visualize growth across an entire array with drill down interaction to images and per-hour cell counts for single apartments
- [Heatmaps Over Time](https://drive.google.com/file/d/18fNeAVOHsj7K0bCn-UZSPzvb-QEoXCcE/view?usp=sharing) - How the time-indexed heatmaps can be used for QC (e.g. identifying apartments that were not counted) or visualizing cell growth rates over time
- [Apartment Time Lapse](https://drive.google.com/file/d/18fNeAVOHsj7K0bCn-UZSPzvb-QEoXCcE/view?usp=sharing) - Visualize cell counts for individual apartments over time as well as export video time lapses of segmented objects within those apartments
    
### Examples

- [Processing Raw Microscope Images](python/notebook/examples/processing_py.ipynb) - This example shows how an experiment producing raw images of cell apartments can be processed to accomplish the following:
    - Extract single apartment images from multi-apartment images
    - Extract individual cell images from apartment images
    - Quantify the cells in single apartments (counts, sizes, "roundness", etc.)
    - Interpret the database of information that results from processing (3 tables, one for raw images, apartments, and individual cells)
- [Processing CLI](python/notebook/examples/processing_cli.ipynb) - This example shows how to accomplish the above using the CLI instead of python, as well as how to run a growth rate analysis
- Generating Videos ([basic](python/notebook/examples/generating_videos_simple.ipynb) | [advanced](python/notebook/examples/generating_videos_detailed.ipynb)) - These examples show how to get details about specific apartments (like videos), after using the pre-computed cell counts to help select interesting ones

### Installation and Setup

To use the tools in this repo, you need
[nvidia-docker](https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)#installing-version-20) running
on Ubuntu (preferably 16.04 but other versions may work too).  Installing nvidia-docker will also involve installing
[NVIDIA Drivers](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#package-manager-installation)
as well as standard [Docker](https://docs.docker.com/install/).

After that, there isn't much to setup other than building and running the docker container.  All that requires is
to first clone this repository somewhere locally (e.g. ~/repos), and then run:

```bash
# Pull the latest docker image (can be run anywhere)
nvidia-docker pull eczech/celldom:latest

# Decide which locally directory you want to use within the container as
# the main storage directory (otherwise, everything you generate in the container is temporary)
export CELLDOM_DATA_DIR=/data/disk2/celldom

# Run the container, which will show a link to visit in your browser
# Port relationships: 8888 -> Jupyterlab, 6006 -> Tensorboard, 8050-8060 -> Dash App
nvidia-docker run --rm -ti -p 8888:8888 -p 6006:6006 -p 8050-8060:8050-8060 \
-v $CELLDOM_DATA_DIR:/lab/data/celldom \
--name celldom eczech/celldom:latest
```

The primary interface to the container is [JupyterLab](http://jupyterlab.readthedocs.io/en/stable/), which will be available on the localhost at port 8888.


### Training 

The training process for all 3 models types can be found in these notebooks:

- [Cell Model Training](python/notebook/training/cells/cell_model_training.ipynb) - This notebook shows how the Mask-RCNN cell model is trained to identify individual cell objects across several cell lines and chip form factor.
- [Marker Model Traiing](python/notebook/training/markers/marker_model_training.ipynb) - A "marker" can be any part of a chip apartment that is used to identify common pixel offsets.  This can be any visual feature of the apartment, though there is typically a feature printed on the chips specifically for doing this kind of key point identification.  See the notebook for example images and how large images containing many apartments are broken into individual apartment images.  This model is also based on the Mask-RCNN architecture.
- [Digit Model Training](python/notebook/training/digits/digit_model_training.ipynb) - Digit images are extracted from raw microscope images after the "marker" for each apartment has been identified (using fixed pixel offsets).  Exports of many of these images were annotated with the appropriate digit label and a 10-class classifier was trained in this notebook to recognize each digit.


## Development Notes

### Backups 

To sync local annotations to Google Storage:

```bash
cd /data/disk2/celldom/dataset
gsutil rsync -r training gs://celldom/dataset/training
```

### Models

Trained models stored as ```.h5``` files are available at [https://storage.googleapis.com/celldom/models](https://console.cloud.google.com/storage/browser/celldom/models/?project=hammerlab-chs&authuser=0).

Currently, both cell and digit recognition models (saved as ```cell_model.h5``` and ```single_digit_model.h5``` 
respectively) are agnostic to chip type which means that selecting a model to use
for a new experiment is as simple as finding the most recently trained one.  In other words, the model with
with the highest "rX.X" designation should be the most recently trained version.

Marker models on the other hand can have a different target outcome that is chip-dependent.  This relationship
between chips and the most recently trained marker models is as follows:

- **G1**: r0.7/marker_model.h5
- **G2**: r0.6/marker_model.h5
- **G3**: r0.6/marker_model.h5
- **ML**: r0.8/marker_model.h5 
