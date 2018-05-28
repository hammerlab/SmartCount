# Docker Instructions

```
cd $REPOS/celldom

# Build image 
nvidia-docker build --no-cache -t celldom -f docker/Dockerfile.devel .
nvidia-docker build -t celldom -f docker/Dockerfile.devel .

# Set locations of local repos (on host)
export CELLDOM_DATA_DIR=/data/disk2/celldom
export CELLDOM_REPO_DIR=/home/eczech/repos/celldom
export CVUTILS_REPO_DIR=/home/eczech/repos/cvutils

# Run container and bind required volumes on host
nvidia-docker run --rm -ti -p 8888:8888 -p 6006:6006 \
-v $CELLDOM_DATA_DIR:/lab/data/celldom \
-v $CELLDOM_REPO_DIR:/lab/repos/celldom \
-v $CVUTILS_REPO_DIR:/lab/repos/cvutils \
celldom

```