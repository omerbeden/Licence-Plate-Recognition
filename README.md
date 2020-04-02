# Licence-Plate-Recognition
Licence Plate Recognition Using Tensorflow Object Detection API
+ Tensorflow version 1.15.2
+ Python version 3.6.9
![application](/app.png)
## Introduction
In this repository , I used Tensorflow Object Detection API on both train and inference time.It is very common API that could use.There is a few things you should do if you want to get this application as same as mine. I use the Tensorflow API on **Google COLAP**

## Installition
### Tensorflow API Installiton
There is 2 option to install Tensorflow API. You can follow [original repo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) or you can follow [this repo](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10) to learn how to **train** a model and how to make **prediction** out of trained model.

### Extra Libraries
+ PyQt5  to install `pip install PyQt5`

### Protobuff installition
I Used protobuf 3.4.0 in this project. You can see protobuff versions [over here](https://github.com/protocolbuffers/protobuf/releases)

### COCO API Installition
Go to [this repo](https://github.com/cocodataset/cocoapi) and download it. To install it you can [visit here](https://medium.com/@abinovarghese/installing-coco-api-in-windows-python-9b4dfc3812ef).

### Download This Repo
+ After the all things have been installled , clone or  download this repot
+ Move this repo files to .../models/research/object_detection folder

### Don't Forget
Before the run this project , dont forget you should have these files in .../models/research/object_detection folder
+ inference_graph
+ inference_graph2
+ training
+ training2
+ also don't forget to move images_inference file in this repo to ojbect_detection file because the application gets the image from this folder

inference graph files and training files  can be created following  [this repo](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10)

## Dataset
+ I collect the dataset from google images
+ [car dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)
+ [AOLP dataset](https://github.com/HaoRecog/AOLP)
+ Hsu, G.-S.; Chen, J.-C. and Chung, Y.-Z., "Application-Oriented License Plate Recognition," Vehicular Technology, IEEE Trans., vol.62, no.2, pp.552-561, Feb. 2013


### in the training folder these files should in it
+ labelmap.pbtxt
+ .... .config file

***if you want me to share my models graph please send me an e-mail***





