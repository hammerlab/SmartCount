# Chip Configuration

These configurations are necessary in order for the system to know where important components of a
chip are relative to the marker.

For example, in [chip_01.yaml](https://github.com/hammerlab/celldom/blob/39ab8d8c06dbdec67c9f1c7c19769882452d071e/config/chip/chip_01.yaml) there is a section like this:

```
apt_margins:
    left: -160
    right: 75
    bottom: 60
    top: -250
```

This identifies the bounding box around an apartment as offsets from the center of the marker on the chip.

Each of these properties can be specified manually but the simplest way to create a new configuration is to:

1. Open the [via.html](via.html) file in this directory (this is an image annotation tool)
2. Go to ```Annotation -> Import Annotations (from csv)``` and select the [annotations/chip_01.csv](annotations/chip_01.csv)
 file in this directory
    - This will load an image from a remote URL and show what it's annotations should look like for a chip
    - You should then see something like this:
   
![Screenshot](https://drive.google.com/uc?export=download&id=17xYyCNRA3uKBJK-6xSjUsQuQGmTAA22u)
    
3. Set the property  ```Update Project Settings (cog wheel in menu bar) -> Region Label``` to ```Value of Region
Attribute: type```; After doing this, the annotated regions should now have labels that make it more
clear what they are.
4. Load a new image of a chip that you would like to create a configuration for, and add similar bounding boxes with
exactly the same attributes (there is only one attribute called "type" and each of those has to be assigned for the
new chip).
    - **IMPORTANT**: All annotations should be bounding boxes except for the **marker_center**,
    which should be a point annotation
    - Here is a list of all the different chip components that must be annotated (you can view these via ```View -> Toggle annotation editor```:
    
    ![Attrs](https://drive.google.com/uc?export=download&id=17jGAxCrw9iNX7CXgcD2shBXp1vWkGOqg)
    
5. Export the annotations as a csv file in the [annotations](annotations) folder in this same directory
6. Run the ```chip_config_generator.py``` script to convert the annotations to a chip config.
    - To be able to run this, there are two options:
        1. Run this in the provided docker container via a terminal in JupyterLab (recommended)
        2. Run this on a local machine, after downloading and installing Anaconda
        ([Anaconda Download](https://www.anaconda.com/download/#macos)) and doing the following:

        ```
        > conda create -n celldom python=3.6  # If using anaconda
        > source activate celldom             # If using anaconda
        > echo "$CELLDOM_REPO_DIR/python/source" > $(python -m site --user-site)/local.pth # Add celldom code to pypath
        > pip install fire pyyaml pandas numpy   # Install necessary packages
        ```

    - Next, the script can be run like this:

    ```
    # This will convert the exported annotations to a .yaml file in the celldom code repo
    python chip_config_generator.py convert-annotations-to-config \
    --annotation-csv=annotations/chip_02.csv \ # This is the exported csv from step 5 above
    --chip-name='chip_02' \  # This name is used to create a yaml file in the config/chip folder
    --save-result=True
    ```
    
    For more context, when running this in JupyterLab the command and output should look like this:

    ![Execution Example](https://drive.google.com/uc?export=download&id=1BOk6h20PXVjYtZgPK3aC9x0KfrTCd01p)
