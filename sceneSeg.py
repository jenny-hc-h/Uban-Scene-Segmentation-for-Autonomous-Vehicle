from PIL import Image
import numpy as np
import glob 
import os
import tqdm
import matplotlib.pyplot as plt
import scipy.io as spio
import scipy.misc as spmi

LEARNING_RATE = 0.01
BATCH_SIZE = 10
MAX_STEP = 100
N_EXAMPLE = 1 # maximum 3475
N_TRAINING_DATA = 1
IMAGE_SCALE = 0.25


def load_dataset(dataset_path,N_examples,N_traingdata):
    """
    Cityscapes Dataset : https://www.cityscapes-dataset.com/
    """
    labelset = []
    imageset = []
    im_fpath = glob.glob(dataset_path+"/leftImg8bit/*.png")
    for n in tqdm.tqdm(range(0,N_examples)):
        # load labels --- N x H x W x C(19)
        lb_fn = os.path.splitext(im_fpath[n].split('/')[-1])[0][0:-12] + '_gtFine_color.mat'
        labelset.append(np.array(spio.loadmat(dataset_path+"/gtFine/"+lb_fn)['label']))
        # load images --- N x H x HW x 3
        image = Image.open(im_fpath[n])
        imageset.append(np.array(spmi.imresize(image, (int(image.size[1]*IMAGE_SCALE),int(image.size[0]*IMAGE_SCALE),3), interp='bilinear')))

    """................................test label.................................
    fig, ax = plt.subplots(1, 1)
    plt.axis('off')
    im =  np.array(spmi.imresize(image, (int(image.size[1]*IMAGE_SCALE),int(image.size[0]*IMAGE_SCALE),3), interp='bilinear'))
    ax.imshow(im)
    label = np.array(spio.loadmat(dataset_path+"/gtFine/"+lb_fn)['label'])
    ax.imshow(label[:,:,13], alpha=0.3)
    plt.show()
    ................................test label................................."""

    # Split the dataset into training and testing sets
    train_data = np.array(imageset[0:N_traingdata]) 
    test_data = np.array(imageset[N_traingdata:N_examples])
    train_labels = np.array(labelset[0:N_traingdata])
    test_labels = np.array(labelset[N_traingdata:N_examples])
    print train_data.shape, train_labels.shape
    return {"train_data":train_data, "test_data": test_data, "train_labels": train_labels, "test_labels": test_labels}
    
dataset_path = "/Users/JennyH/Desktop/Jenny/UCSD/HW/MachineLearning/CityscapesDataset"
load_dataset(dataset_path, N_EXAMPLE, N_TRAINING_DATA)


