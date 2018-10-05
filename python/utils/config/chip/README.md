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

- Open the [via.html](via.html) file in this directory (this is an image annotation tool)
- Choose an image to annotate such as this extraction of a single apartment from a G3 chip: ![Apartment Image](https://drive.google.com/uc?export=download&id=1zsv1YX8uXj_eC14EpUSbbQstxfQnIkHI)
- Add annotations to the image with the goal being to eventually arrive at a result like this:

![Screenshot](https://drive.google.com/uc?export=download&id=17xYyCNRA3uKBJK-6xSjUsQuQGmTAA22u)
   
- To add these annotations, a good place to start is by drawing a bounding box and then expanding
the ```Attributes``` on the left panel to add a new attribute called "type".  Then for every annotation made,
you can go to ```View -> Toggle annotation editor``` to be able to set the "type" for each.  This should
appear like this when complete:

![Attrs](https://drive.google.com/uc?export=download&id=17jGAxCrw9iNX7CXgcD2shBXp1vWkGOqg)
 
- Note that in order for labels of annotations to show up strings instead of numbers, you have to 
set ```Update Project Settings (cog wheel in menu bar) -> Region Label``` to ```Value of Region
Attribute: type```.  After doing this, the annotated regions should now have labels that make it more
clear what they are.

- The ```chip_border``` (bounding box) and ```marker_center``` (point) annotations are required, but the others 
have a more generic behavior.  For example, ```st_num_1``` and ```st_num_2``` could be extended to a third digit
by just adding a ```st_num_3``` annotation.  Similarly, the annotations prefixed by ```component_``` are not
required but they should be added as necessary since cells identified will be associated with membership in those
components (**IMPORTANT**: Cells are automatically filtered to these components to make sure none are identified
in unwanted locations).  Also, these are the only annotations that can be polygons as well as bounding boxes.
    
- Export the annotations as a csv file using ```Annotations -> Export Annotations (as csv)```.  It is also a
good idea to then rename and move this file to somewhere memorable
- Run the ```celldom create_chip_configuration``` command to convert the annotations to a chip config.  This
should be in a JupyterLab terminal within the provided docker container; Example: 

    ```
    # This command will print the configuration to stdout as a yaml document
    celldom create_chip_configuration /lab/repos/celldom/config/chip/chip-G02/chip-G02.via.csv \
    --chip_name=G02 \
    --marker_spacing=[255,275] \
    --apt_num_range=[0,46] \
    --st_num_range=[0,31] 
    
    # For convenience, this can either be pasted or redirected, for example ("!!" means rerun the last command):
    !! > /lab/repos/celldom/config/chip/chip-G02.yaml
    ```

