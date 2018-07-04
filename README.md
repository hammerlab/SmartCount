# Celldom

Repository for collaboration on Celldom computer vision solutions

### Examples

- [Processing Raw Microscope Images](python/notebook/examples/processing_py.ipynb) - This example shows how an experiment producing raw images of cell apartments can be processed to accomplish the following:
    - Extract single apartment images from multi-apartment images
    - Extract individual cell images from apartment images
    - Quantify the cells in single apartments (counts, sizes, "roundness", etc.)
    - Interpret the database of information that results from processing (3 tables, one for raw images, apartments, and individual cells)
- [Processing CLI](python/notebook/examples/processing_cli.ipynb) - This example shows how to accomplish the above using the CLI instead of python, as well as how to run a growth rate analysis
- [Generating Videos](python/notebook/examples/generating_videos.ipynb) - This example shows how to get details about specific apartments (like videos), after using the pre-computed cell counts to help select interesting ones

