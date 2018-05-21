# Celldom

Repository for collaboration on Celldom computer vision solutions


### Environment Initialization Notes

\* TODO: Move to dockerfile

    conda create -n celldom python=3.6
    python -m pip install --upgrade pip
    jupyterlab
    matplotlib
    python-opencv
    scikit-image
    tensorflow-gpu # Using TF 1.8.0
    keras
    pandas 
    plotnine

    cd Mask_RCNN
    python setup.py install

    # See: https://github.com/matterport/Mask_RCNN/issues/6
    pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"

