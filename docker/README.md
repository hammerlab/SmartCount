# Docker Instructions

nvidia-docker build --no-cache -t celldom -f docker/Dockerfile.devel .
nvidia-docker build -t celldom -f docker/Dockerfile.devel .

export CELLDOM_DATA_DIR=/data/disk2/celldom
export CELLDOM_REPO_DIR=/home/eczech/repos/celldom
nvidia-docker run -ti -p 8888:8888 \
-v $CELLDOM_DATA_DIR:/lab/data \
-v $CELLDOM_REPO_DIR:/lab/repos/celldom \
-e "CELLDOM_DATA_DIR=/lab/data" \
-e "CELLDOM_REPO_DIR=/lab/repos/celldom" \
celldom
