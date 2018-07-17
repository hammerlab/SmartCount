# Celldom

Repository for collaboration on Celldom computer vision solutions

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
as well as regular [Docker](https://docs.docker.com/install/) -- both of these links are from the first link on
installing nvidia-docker and all three worked fine for me on the first go at it.

After that, there isn't much to setup other than building and running the docker container.  All that requires is
to first clone this repository somewhere locally (say ~/repos), and then run:

```
cd ~/repos/celldom/docker

# Build the container (you only need to do this once -- or if there are important changes to it)
nvidia-docker build -t celldom -f Dockerfile.prd .

# Decide which locally directory you want to use within the container as
# the main storage directory (otherwise, everything you generate in the container is temporary)
export CELLDOM_DATA_DIR=/data/disk2/celldom

# Set this to be the location of this repository on the host system
export CELLDOM_REPO_DIR=$HOME/repos/celldom

# Run the container, which will show a link to visit in your browser
nvidia-docker run --rm -ti -p 8888:8888 -p 6006:6006 \
-v $CELLDOM_DATA_DIR:/lab/data/celldom \
-v $CELLDOM_REPO_DIR:/lab/repos/celldom \
celldom

```

After running the container, you will see a message like:

    Copy/paste this URL into your browser when you connect for the first time,
    to login with a token:
        http://localhost:8888/?token=405433c2da0457103c4727d96392x50945ce85bdee80e095


If a browser has already opened (which happens automatically on mac), then you can paste the portion
after "token=" into the text box to login.  After this you will be in [jupyterlab](http://jupyterlab.readthedocs.io/en/stable/)
and can use that to write/run code (either python or bash).
