# Docker Instructions

sudo nvidia-docker build --no-cache -t celldom -f docker/Dockerfile.gpu .
sudo nvidia-docker build -t celldom -f docker/Dockerfile.gpu .

export CELLDOM_DATA_DIR=/data/disk2/celldom
export CELLDOM_REPO_DIR=/home/eczech/repos/celldom
sudo nvidia-docker run -ti -p 8888:8888 \
-v $CELLDOM_DATA_DIR:/notebooks/data \
-v $CELLDOM_REPO_DIR:/notebooks/repo \
-e "CELLDOM_DATA_DIR=/notebooks/data" \
-e "CELLDOM_REPO_DIR=/notebooks/repo" \
celldom
