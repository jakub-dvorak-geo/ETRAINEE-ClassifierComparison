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

spectral 1D - As the name suggests, this network performs convolution only along the spectral dimension. In other words it classifies one pixel at a time without taking spatial relationships into account. Convolutional layers may be followed by fully connected layers which perform the final classification.

__image here__

spatial 2D - Spatial CNNs are most commonly used for classification of multispectral data, as well as for general Computer Vision tasks. Convolution is performed along the spatial dimensions while the spectral bands get stacked into input feature maps. For hyperspectral data, spatial networks may be preceded by dimensionality reduction (often using PCA).

__image here__

spectro-spatial 3D - This network structure use two dimensions for spatial information, with the third dimension . ex...
Most complex of the presented networks, 3D CNNs require considerably higher computational power and more memory than the other two presented CNN arcitectures.

__image here__

## Examples

e.g. from literature
current code gets revierwed
Solid explanations and experimentation with the neural network code are provided by both Audebert et al. (2019) and Paoletti et al. (2019)

## Exercise

### Classification of hyperspectral data using CNNs

This exercise introduces three types of convolutional neural networks via three Jupyter notebooks. Each notebook represents a different approach to classification of hyperspectral imagery:
* spectral (1D)
* spatial (2D)
* spectro-spatial (3D)

The notebooks are currently setup to use the Pavia City Centre dataset, however any GDAL-readable rasters can be used instead. Function _image_preprocessing.read_gdal()_ takes paths to a training data raster and a reference data raster as parameters. Both need to have the same extent and pixel size.

- *Prerequisities*
   - Downloaded data ([folder](course/module4/01_spectroscopy_principles/data))
   - Installed EnMAP-Box plugin to QGIS ([manual](../../software/software_enmap_box.md))
- *Tasks*
   - Set up the environment
   - Go through the code
   - Train a CNN of your choice
   - Alter hyperparameters and observe

#### 1. Setting up the environment

All the code can be run either on local machines, or in the cloud, for example through Google Colab. We currently recommend using Google Colab as it is significantly easier to set up and faster to run. However it requires a Google account to use.

##### 1.1 Google Colab

Open the corresponding links to individual notebooks and copy them to your own google drive:


##### 1.2 Local machine
The installation guide is created for Windows. If using MacOS/Linux, most of the process remains the same, except you may need to install GDAL Python API in a different way and it may not be possible use your GPU, given the CUDA toolkit may not be available on your system.

Download the code from [https://github.com/YesPrimeMinister/ETRAINEE-ClassifierComparison](https://github.com/YesPrimeMinister/ETRAINEE-ClassifierComparison) and the Pavia Centre dataset from [https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes#Pavia_Centre_and_University](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes#Pavia_Centre_and_University)

If you already have some experience with Python, you may likely skip to installing external libraries:

* Python installation
* Virtual environment setup (optional)
* Installing GDAL Python API
* Installing PyTorch
* Installing most external libraries
* Running jupyter notebook

###### 1.2a Python installation
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

###### 1.2b Virtual environment setup (optional)
We recommend creating a new Python virtual environment for this project, so your other environments don't become cluttered, which may even lead to conflicts between libraries. While this is not strictly necessary, virtual environments can be created using either [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands) or [venv](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/).

###### 1.2c Installing GDAL Python API
Based on your preferred Python package manager, install GDAL either through

pip - [https://opensourceoptions.com/blog/how-to-install-gdal-for-python-with-pip-on-windows/](https://opensourceoptions.com/blog/how-to-install-gdal-for-python-with-pip-on-windows/)

or

conda - [https://opensourceoptions.com/blog/how-to-install-gdal-with-anaconda/](https://opensourceoptions.com/blog/how-to-install-gdal-with-anaconda/)

###### 1.2d Installing PyTorch
Suitable command for PyTorch installation should be selected on the [PyTorch website](https://pytorch.org/get-started/locally/) based on if your computer has a GPU by Nvidia:
* If you have a Nvidia CUDA-capable GPU then you can install _CUDA toolkit_ and _CuDNN_ from the [Nvidia website](https://developer.nvidia.com/cuda-toolkit), you need to sign up for 'NVIDIA Developer Program' in order to download CuDNN (check the PyTorch website first, so you install an appropriate version of _CUDA_ and _CuDNN_). After successfully installing _CUDA toolkit_ and _CuDNN_, install PyTorch using the command from the [PyTorch website](https://pytorch.org/get-started/locally/). All models were tested with Python 3.9, CUDA version 10.1, CuDNN 7.6 and PyTorch 1.8.1.
* If you do not have a CUDA-capable Nvidia GPU, you can simply use PyTorch on the CPU, by selecting `CPU` in the _Compute Platform_ field on the [PyTorch website](https://pytorch.org/get-started/locally/). All models were tested with Python 3.9 and PyTorch 1.10.2.

###### 1.2e Installing remaining external libraries
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

#### 2. Go through the code
Read the code explanation for at least one of the networks and try to understand it. You will hopefully get an idea of why perform the different preprocessing operations, what the network structure represents and how the training procedure works.

#### 3. Train a CNN of your choice
Train networks for classification


#### 4. Alter hyperparameters and observe
Try changing up some hyperparameters - primarily the number of training epochs, learning rate, batch size or class weights

### Next unit
Proceed with a case study on [seasonal dynamics of flood-plain forests](../07_flood_plain_forest/07_flood_plain_forest.md)
