from PIL import Image
import numpy as np
import glob 
import os
import tqdm

CLASSES = [33, 34, 35, 36, 38, 39, 40] #car, motorcycle, bicycle, pedestrian, truck, bus, tricycle
N_EXAMPLE = 10
N_TRAINING_DATA = 90


def load_dataset(dataset_path,N_examples,N_traingdata):
    labelset = []
    imageset = []
    im_fpath = glob.glob(dataset_path+"/train_color/*.jpg")
    for n in tqdm.tqdm(range(0,N_examples)):
        # load labels --- NxWxH
        lb_fn = im_fpath[n].split('/')
        lb_fn = os.path.splitext(lb_fn[-1])[0] + '_instanceIds.png'
        print lb_fn
        label = np.array(Image.open(dataset_path+"/train_label/"+lb_fn))
        print np.unique(label)
        class_numbers = np.unique(label/1000)
        class_based = np.zeros(label.shape)
        for i in class_numbers:
            if i in CLASSES:
                loc = (label/1000) == i
                class_based[loc] = (CLASSES.index(i) + 1)
        labelset.append(class_based)
        print np.unique(class_based) 
        # load images --- NxWxHx3
        image = np.array(Image.open(im_fpath[n]))
        imageset.append(image)

    # Split the dataset into training and testing sets
    train_data = np.array(imageset[0:N_traingdata]) # NxWxHx3
    test_data = np.array(imageset[N_traingdata:N_examples])
    train_labels = np.array(labelset[0:N_traingdata]) # NxWxH
    test_labels = np.array(labelset[N_traingdata:N_examples])
    return {"train_data":train_data, "test_data": test_data, "train_labels": train_labels, "test_labels": test_labels}
    

load_dataset("/Users/JennyH/Desktop/Jenny/UCSD/HW/MachineLearning/Dataset", N_EXAMPLE, N_TRAINING_DATA)


