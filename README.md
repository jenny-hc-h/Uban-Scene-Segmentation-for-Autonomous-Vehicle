# ECE228_FinalProject

Uban Scene Segmentation for Autonomous Vehicle
ECE228 Fianal Project - Group 8

Introduction:
In this project, we are going to use fully convolutional networks(FCN) to perform pixel-wised classification in urban scene images. We apply two popular models, AlexNet and VGG net, into fully convolutional networks, and try to compare the results of these two different model with/without skip connection. The skip connection is a method to add back the resolution we loss during the convolution and pooling operation.

Source Code:
We provide four versions of code, which are
(1)sceneSeg_AlexNet_nonskip.py
(2)sceneSeg_AlexNet_skip.py
(3)sceneSeg_VGG_nonskip.py
(4)sceneSeg_VGG_skip.py

To execute this project, type the following command in terminal.
For checking the dataset format:

    python checkDataset.py --dataset DATASET_DIR
    
For training:

    python sceneSeg.py --mode train --dataset DATASET_DIR 
    
For visualization:

    python sceneSeg.py --mode visualize --image IMG_PATH
    python sceneSeg.py --mode visualize --imagedir IMG_FOLDER_DIR   
   
DATASET_DIR is the direction of the dataset folder.

IMG_PATH is the image path.

MODE is the mode "train" or "visualize".
