# Docker Instructions

#### Development Container Instructions

```
cd $REPOS/celldom

nvidia-docker build -t celldom-dev -f docker/Dockerfile.dev .

export CELLDOM_DATA_DIR=/data/disk2/celldom
export CELLDOM_REPO_DIR=$HOME/repos/celldom
export CVUTILS_REPO_DIR=$HOME/repos/cvutils

nvidia-docker run --rm -ti -p 8888:8888 -p 6006:6006 \
-v $CELLDOM_DATA_DIR:/lab/data/celldom \
-v $CELLDOM_REPO_DIR:/lab/repos/celldom \
-v $CVUTILS_REPO_DIR:/lab/repos/cvutils \
celldom-dev
```

#### Production Container Instructions

```
cd $REPOS/celldom

nvidia-docker build -t celldom -f docker/Dockerfile.prd .

export CELLDOM_DATA_DIR=/data/disk2/celldom
export CELLDOM_REPO_DIR=$HOME/repos/celldom

nvidia-docker run --rm -ti -p 8888:8888 -p 6006:6006 \
-v $CELLDOM_DATA_DIR:/lab/data/celldom \
-v $CELLDOM_REPO_DIR:/lab/repos/celldom \
celldom
```