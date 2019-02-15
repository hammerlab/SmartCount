# Docker Instructions

#### Production Container Instructions

```bash
nvidia-docker run --rm celldom echo "hello"
ndocker run -d --rm --name celldom celldom
```

```bash
cd $REPOS/celldom/docker

nvidia-docker build -t celldom -f Dockerfile.prd .

export CELLDOM_DATA_DIR=/data/disk2/celldom
export CELLDOM_REPO_DIR=$HOME/repos/celldom

nvidia-docker run --rm -ti -p 8888:8888 -p 6006:6006 -p 8050-8060:8050-8060 \
-v $CELLDOM_DATA_DIR:/lab/data/celldom \
-v $CELLDOM_REPO_DIR:/lab/repos/celldom \
celldom
```

#### Development Container Instructions

This is only necessary when developing/testing some repos simultaneously and
when you intend to train models (as opposed to just use them):

```bash
cd $REPOS/celldom/docker

nvidia-docker build -t celldom-dev -f Dockerfile.dev .

export CELLDOM_DATA_DIR=/data/disk2/celldom
export CELLDOM_REPO_DIR=$HOME/repos/celldom
export CVUTILS_REPO_DIR=$HOME/repos/cvutils
export SVHN_REPO_DIR=$HOME/repos/misc/SVHN-Classifier
export CONFIG_NUM_GPUS_TRAIN=1

nvidia-docker run --rm -ti -p 8888:8888 -p 6006:6006 -p 8050-8060:8050-8060 \
-v $CELLDOM_DATA_DIR:/lab/data/celldom \
-v $CELLDOM_REPO_DIR:/lab/repos/celldom \
-v $CVUTILS_REPO_DIR:/lab/repos/cvutils \
-v $SVHN_REPO_DIR:/lab/repos/svhn \
-e CONFIG_NUM_GPUS_TRAIN=$CONFIG_NUM_GPUS_TRAIN \
-e MASK_RCNN_CACHE_DIR=/lab/data/celldom/model/pretrained \
celldom-dev
```