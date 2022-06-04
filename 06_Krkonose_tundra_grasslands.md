# 1D, 2D and 3D CNNs
> Classify hyperspectral data using different Convolutional Neural Networks (CNNs), which use convolution in the spectral, spatial or in both spectral and spatial domains.

Utilization of convolutional neural networks (CNN) has been growing steeply in many fields, including remote
sensing. At the same time, several textbooks and online learning materials have appeared. What is not so
frequent or missing, are easy-to-use tools enabling practical experimentation with different designs of CNNs.
The presented Classifier Comparison tool, implemented in Python, helps users understand 1D, 2D, and 3D
(spectral, spatial and spectro-spatial) CNN architectures for classification of hyper- or multispectral images,
while presenting a straightforward framework for building more complex networks.

Target audience for our tool are MSc and PhD students, researchers and practitioners from public sector and
industry in fields related to remote sensing and computer vision dealing with CNNs at a beginner level. We
expect only a basic knowledge of CNN fundamentals, Python and Jupyter notebooks.

The presented tool was developed within the ongoing project “E-learning course on Time Series Analysis in
Remote Sensing for Understanding Human-Environment Interactions” (E-TRAINEE, ID 2020-1-CZ01-KA203-
078308) funded by the Erasmus+ Strategic partnership programme. For details see the project homepage:
[https://web.natur.cuni.cz/gis/etrainee/](https://web.natur.cuni.cz/gis/etrainee/)


## Installation

The installation guide is created for Windows. If using MacOS/Linux, most of the process remains the same, except you may need to install GDAL Python API in a different way and it may not be possible use your GPU, given the CUDA toolkit may not be available on your system.
If you already have some experience with Python, you may likely skip to installing external libraries:

- Python installation
- Virtual environment setup (optional)
- Installing GDAL Python API
- Installing PyTorch
- Installing most external libraries
- Running jupyter notebook

### Python installation
Necessary only if you don't have Python 3 already. To try if you have Python 3 installed, open the command line and run either
```sh
python
```
or
```sh
python3
```
If Python 3 is available, it should look similar to this:

![Python running in the command line](img/cli_python3.png "Python running in the command line")

If you don't already have a Python 3 installation, we recommend installing miniconda with Python 3.9 from [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html).

### Virtual environment setup (optional)
We recommend creating a new Python virtual environment for this project, so your other environments don't become cluttered, which may even lead to conflicts between libraries. While this is not strictly necessary, virtual environments can be created using either [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands) or [venv](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/).

### Installing GDAL Python API
Based on your preferred Python package manager, install GDAL either through

pip - [https://opensourceoptions.com/blog/how-to-install-gdal-for-python-with-pip-on-windows/](https://opensourceoptions.com/blog/how-to-install-gdal-for-python-with-pip-on-windows/)

or

conda - [https://opensourceoptions.com/blog/how-to-install-gdal-with-anaconda/](https://opensourceoptions.com/blog/how-to-install-gdal-with-anaconda/)

### Installing PyTorch
Suitable command for PyTorch installation should be selected on the [PyTorch website](https://pytorch.org/get-started/locally/) based on if your computer has a GPU by Nvidia:
* If you have a Nvidia CUDA-capable GPU then you can install _CUDA toolkit_ and _CuDNN_ from the [Nvidia website](https://developer.nvidia.com/cuda-toolkit), you need to sign up for 'NVIDIA Developer Program' in order to download CuDNN (check the PyTorch website first, so you install an appropriate version of _CUDA_ and _CuDNN_). After successfully installing _CUDA toolkit_ and _CuDNN_, install PyTorch using the command from the [PyTorch website](https://pytorch.org/get-started/locally/). All models were tested with Python 3.9, CUDA version 10.1, CuDNN 7.6 and PyTorch 1.8.1.
* If you do not have a CUDA-capable Nvidia GPU, you can simply use PyTorch on the CPU, by selecting `CPU` in the _Compute Platform_ field on the [PyTorch website](https://pytorch.org/get-started/locally/). All models were tested with Python 3.9 and PyTorch 1.10.2.

### Installing most external libraries
Required Python libraries are:
- matplotlib
- sklearn
- torchnet
- notebook
- tqdm
- ipywidgets
- scipy

Most can be installed only using either

```sh
pip install <library name>
```
or
```sh
conda install <library name>
```

## Usage example

A few motivating and useful examples of how your product can be used. Spice this up with code blocks and potentially more screenshots.

Currently work in progress, sorry for the inconvenience.



---
title: "E-TRAINEE Case study: monitoring tundra grasslands in the Krkonoše Mountains"
description: "This is the sixth theme within the Airborne Imaging Spectroscopy Time Series Analysis module."
dateCreated: 2021-03-28
authors:
contributors:
estimatedTime:
---

# Case study: monitoring tundra grasslands in the Krkonoše Mountains

Utilization of convolutional neural networks (CNN) has been growing steeply in many fields, including remote
sensing. At the same time, several textbooks and online learning materials have appeared. What is not so
frequent or missing, are easy-to-use tools enabling practical experimentation with different designs of CNNs.
The presented Classifier Comparison tool, implemented in Python, helps users understand 1D, 2D, and 3D
(spectral, spatial and spectro-spatial) CNN architectures for classification of hyper- or multispectral images,
while presenting a straightforward framework for building more complex networks.

## Objectives

In this theme, you will learn about:
* 1D, 2D, 3D convolutional neural network structure
* data cubes
* examples of sensors, their spectral and spatial resolution – UAV, airborne, spaceborne, and laboratory
* comparison of hyperspectral and multispectral imaging
* spectral libraries

After finishing this theme you will be able to:
* Visualise spectral information
* Classify


## Methods

If your not


## Examples ---filler

e.g. from literature


## Exercise

### Classification of hyperspectral data using CNNs

This exercise aims to introduce three types of convolutional neural networks

The aim of this exercise is to get acquainted with the data in the EnMAP-Box environment, to understand the possibilities
of its visualization and concept and to compare the spectral properties of the selected classes.

- *Prerequisities*
   - Downloaded data ([folder](course/module4/01_spectroscopy_principles/data))
   - Installed EnMAP-Box plugin to QGIS ([manual](../../software/software_enmap_box.md))
- *Tasks*
   - First encounter with data
   - Comparison of spectral characteristics
   - Exploration of changes in spectral characteristics in time

#### 1. First encounter with data
Details about origin of data, equipment used for its capturing and scanned areas can be found [here - TO BE ADDED](LINK)

Multiple regions of interest (ROI) have been selected for this task from a scanned area - Luční hora mountain (LH).
In this exercise you will work with four classes of land cover:
- scree
- mountain pine
- grass
   - _Deschampsia cespitosa_ (tufted hairgrass - [wikipedia](https://en.wikipedia.org/wiki/Deschampsia_cespitosa))
   - _Nardus stricta_ (matgrass - [wikipedia](https://en.wikipedia.org/wiki/Nardus))
- shrub
   - _Vaccinium myrtillus_ (european blueberry - [wikipedia](https://en.wikipedia.org/wiki/Vaccinium_myrtillus))
   - _Calluna vulgaris_ (common heather - [wikipedia](https://en.wikipedia.org/wiki/Calluna))

<img src="media/lh_map_clip.png" alt="Map" title="Map of the area with numbered ROIs" width="800">

Names of files in [data folder](course/module4/01_spectroscopy_principles/data) mostly follow a pattern:
`area_class_subclass_number_year_month`,
e.g. `LH_wetland_2_2020_08` for the second ROI showing wetland in the image of Luční hora mountain captured
in August 2020. Subclass is used only for grass and shrub classes.

##### Load and visualize image

For the first task, choose any ROI you like from [folder with classified images](course/module4/01_spectroscopy_principles/data/task_1_2/classes).
Do not forget to mention a name of the file in your report. Visualize it in the EnMAP-Box plugin in QGIS. Follow the
steps in [manual - Getting Started](https://enmap-box.readthedocs.io/en/latest/usr_section/usr_gettingstarted.html#).
There are some screenshots that might also help you:

- Open EnMAP-Box in QGIS. <img src="media/enmap_box/01_open_enmap_box.png" alt="Open EnMAP-Box in QGIS" title="Open EnMAP-Box in QGIS" width="450">

- Open a map window to later show the raster in. <img src="media/enmap_box/02_open_map_window.png" alt="Open a map window" title="Open a map window" width="250">

- Through a dialog in add data source option, open the chosen image. Point at a file with an extension `dat`. <img src="media/enmap_box/03_add_data_source.png" alt="Add a data source" title="Add a data source" width="220">

- Drag a raster from the data sources and drop it to the map in data views. <img src="media/enmap_box/04_load_raster_to_map.png" alt="Load a raster to the map" title="Load a raster to the map" width="300">

- Adjust visualization of the chosen image. There is just a tip for visualization in true colors in the screenshot.
Minimal and maximal values depend on the chosen image. <img src="media/enmap_box/05_adjust_visualization.png" alt="Adjust visualization" title="Adjust visualization" width="750">

- Due to a short range of wavelength captured in each band and significant amount of noise in some bands, the final image
does not look like multispectral imageries in true colors. <img src="media/enmap_box/06_final_state.png" alt="Final state of the visualization" title="Final state of the visualization" width="750">

##### EnMAP-Box capabilities

Now you are ready to explore EnMAP-Box capabilities. Add screenshots of the following steps to your report.

- First of all, you might be interested in the location where the chosen image comes from. To inspect the surroundings,
load some of the available WMS layers. Maybe, you'll be surprised that for example Google Satellite Maps provides such
a bad resolution of the imagery in contrast to the images you work with in this exercise. <img src="media/enmap_box/07_wms_layer.png" alt="Add WMS Layer" title="Add WMS Layer" width="400"> <img src="media/enmap_box/08_google_satellite_maps.png" alt="Google satellite maps" title="Google satellite maps" width="750">

- Right click on the raster layer in map and choose image statistics or click on this option in menu tools. Select your
image and feel free to also choose actual (slower) accuracy as the image is very small so the computation of statistics
will be fast.  Write down to your report number of available bands, range of captured wavelength and enclose a few
screenshots of the histograms. Similar information except for the histograms can be found in metadata viewer in menu
tools. <img src="media/enmap_box/09_image_statistics.png" alt="Image statistics" title="Image statistics" width="650"> <img src="media/enmap_box/10_metadata_viewer.png" alt="Metadata viewer" title="Metadata viewer" width="450">

##### Spectral curves

As the last step of the first task, let's visualize a wavelength spectrum of individual pixels.

- Click on open a spectral library window next to the option for opening map window and activate a button with a cursor
(_Identify a cursor location and collect pixels values, spectral profiles and or vector feature attributes._)
and the one with a spectral curve (_Identify pixel profiles and show them in a Spectral Library._). <img
src="media/enmap_box/11_open_spectral_library.png" alt="Open a spectral library window" title="Open a spectral library window" width="450">

- Then just click anywhere in the image and the spectral curve should appear in the spectral library window. Spend a while
observing the curves of various pixels, play around with settings for x axis (wavelength in nanometers, band index) and
sampling in the spectral profile sources panel. <img src="media/enmap_box/12_x_axis.png" alt="Choices for x axis" title="Choices for x axis" width="500"> <img src="media/enmap_box/13_sampling.png" alt="Choices for sampling" title="Choices for sampling" width="400">

- It is possible to visualize more curves than for one pixel or its surrounding. Choose option add profiles or add profiles
automatically and curves will remain in the plot. If you want to remove some of them, activate toggle editing mode and
delete selected rows in the table. In the editing mode, profiles can be also renamed, another columns can be added, etc. <img src="media/enmap_box/14_add_profiles.png" alt="Add profiles to the plot" title="Add profiles to the plot" width="600"> <img src="media/enmap_box/15_toggle_editing_mode.png" alt="Toggle editing mode and delete features" title="Toggle editing mode and delete features" width="600">

- It is also possible to change colors of the plot, to export raw data to CSV or to export the plot as a raster image...

Enclose three screenshots of the whole EnMAP-Box window with different settings for spectral curves visualization in your report.

#### 2. Comparison of spectral characteristics
There are three examples for each class and subclass of land cover from August 2020 in
[folder with classified images](course/module4/01_spectroscopy_principles/data/task_1_2/classified).
Open them one by one or more at once in multiple map windows
in EnMAP-Box and visualize the spectral curves of particular pixels. It is recommended to use Sample5x5Mean sampling
because it reduces noise in the signal by averaging of signals from neighbouring pixels.

##### Spectral curves for classes and subclasses
Choose a representative curves for each class and subclass, enclose the plots to your report and
describe typical spectral behaviour of each class, highlight differences between them. Focus on values of pixels
in wavelength ranges of visible and near infrared part of the spectrum, and also on a shape of the curve. Try to answer
following questions:
- Which part of the spectrum can be used for distinguishing between the classes?
- Which ranges of wavelength are similar for two or more classes and can cause insufficient results of classification?
- Is it possible to distinguish between subclasses? Which ranges of wavelength might help you?

<img src="media/enmap_box/16_spectral_curves_classes.png" alt="Spectral curves for two classes" title="Spectral curves for two classes" width="750">

##### Differences in spectral curves within classes
During the previous task, you have probably noticed differences in the spectral curves even between pixels in one image.
Select a suitable pixels and plot multiple different spectral curves of pixels from one image in a one graph. To your
report, answer these questions:
- Are these differences significant from the classification point of view?
- Can the differences cause misclassification of some of the pixels to another class? If so, do you know methods that
are used for removing of noise in the classification results?

##### Classification of additional images
There are six images in the [folder with images of unknown class](course/module4/01_spectroscopy_principles/data/task_1_2/unknown) (`unknown_1-6`).
Go through these
images and try to classify them to one of the classes (and subclasses if possible). Describe your arguments for the
classification and enclose the spectral curves that helped you.

#### 3. Exploration of changes in spectral characteristics in time
For each class and subclass, one ROI is selected and images captured in June, July, August, and September 2020 are in the
[folder for task 3](course/module4/01_spectroscopy_principles/data/task_3).
Open all four images of one ROI in EnMAP-Box and explore differences in spectral curves.
In your report, describe the differences for individual classes. Try to answer following questions:
- For which classes are the changes in spectral curves most obvious?
- If there are differences, which month is the most suitable for separation of the class from the others? On the other hand,
in which month are the spectral curves of one class similar to spectral curves of the others?

<img src="media/enmap_box/17_visualizations_months_mountain_pine.png" alt="Mountain Pine in four months" title="Mountain Pine in four months" width="750">

### Next unit
Proceed with a case study on [seasonal dynamics of flood-plain forests](../07_flood_plain_forest/07_flood_plain_forest.md)
