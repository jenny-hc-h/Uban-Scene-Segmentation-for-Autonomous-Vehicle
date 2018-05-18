from __future__ import print_function
import BatchDatsetReader as batchreader
import TensorflowUtils as utils
from six.moves import xrange
from PIL import Image
import matplotlib.pyplot as plt
import scipy.io as spio
import tensorflow as tf
import numpy as np
import argparse
import scipy.misc as spmi
import datetime
import glob 
import tqdm
import os
from os.path import dirname

"""
Reference: https://github.com/shekkizh/FCN.tensorflow
Training:
    python sceneSeg.py --mode train --dataset DATASET_DIR
Visualization:
    python sceneSeg.py --mode visualize --image IMAGE_PATH

1. extract one of the convolution layer and add it to the final layer -> predict (refer the paper)
2. implement AlexNet & GoogleNet

"""
# ==========================================================================================
N_EXAMPLE = 1000 # maximum 3475
N_TRAINING_DATA = 900
LEARNING_RATE = 0.0001
BATCH_SIZE = 10
TRAIN_CLASSES = [11,13] # max: range(19)
NUM_OF_CLASSES = len(TRAIN_CLASSES) 
# ..........................................................................................
MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'
MODEL_DIR = dirname(__file__)+'/Model/'
LOG_DIR = dirname(__file__)+'/logs/VGGNet_c'+str(NUM_OF_CLASSES)+'/'
# ==========================================================================================

MAX_ITERATION = int(1e5 + 1)
IMSIZE_X = 256
IMSIZE_Y = 512
RGB_OF_CLASSES = {0:(128,54,128),1:(244,35,232),2:(70,70,70),3:(102,102,156),4:(190,153,153), \
5:(153,153,153),6:(250,170,30),7:(220,220,0),8:(107,142,35),9:(152,251,152), \
10:(70,130,180),11:(220,20,60),12:(255,0,0),13:(0,0,142),14:(0,0,70), \
15:(0,60,100),16:(0,80,100),17:(0,0,230),18:(119,11,32),19:(0,0,0)}


def load_dataset(dataset_path,N_examples,N_traingdata):
    """
    Cityscapes Dataset : https://www.cityscapes-dataset.com/
    0: road   1: sidewalk        2: building       3: wall         4: fence
    5: pole   6: traffic light   7: traffic sign   8: vegetation   9: terrain
    10: sky   11: person         12: rider         13: car         14: trunck   
    15: bus   16: train          17: motorcycle    18: bicycle     19: others
     """
    labelset = []
    imageset = []
    im_fpath = glob.glob(dataset_path+"/leftImg8bit/*.png")
    for n in tqdm.tqdm(range(0,N_examples)):
        # load labels --- N x H x W x (C+1)
        lb_fn = os.path.splitext(im_fpath[n].split('/')[-1])[0][0:-12] + '_gtFine_color.mat'
        lab = np.array(spio.loadmat(dataset_path+"/gtFine/"+lb_fn)['label'])
        lab_other = (np.sum(lab[:,:,np.array(TRAIN_CLASSES)], axis=2)==0).astype(int)
        labelset.append(np.concatenate((lab[:,:,np.array(TRAIN_CLASSES)],np.expand_dims(lab_other, axis=2)),axis=2))
        
        # load images --- N x H x W x 3
        image = Image.open(im_fpath[n])
        imageset.append(np.array(spmi.imresize(image, (int(image.size[1]/4),int(image.size[0]/4),3), interp='bilinear')))

    # Split the dataset into training and testing sets
    train_data = np.array(imageset[0:N_traingdata]) 
    test_data = np.array(imageset[N_traingdata:N_examples])
    train_labels = np.array(labelset[0:N_traingdata])
    test_labels = np.array(labelset[N_traingdata:N_examples])
    print(train_data.shape, train_labels.shape)
    return {"train_data":train_data, "test_data": test_data, "train_labels": train_labels, "test_labels": test_labels}

def vgg_net(weights, image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    net = {}
    current = image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
            bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
            current = utils.conv2d_basic(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
        elif kind == 'pool':
            current = utils.avg_pool_2x2(current)
        net[name] = current
    return net


def inference(image, keep_prob):
    """
    Semantic segmentation network definition
    :param image: input image. Should have values in range 0-255
    :param keep_prob:
    :return:
    """
    print("setting up vgg initialized conv layers ...")
    model_data = utils.get_model_data(MODEL_DIR, MODEL_URL)

    mean = model_data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))

    weights = np.squeeze(model_data['layers'])

    processed_image = utils.process_image(image, mean_pixel)

    with tf.variable_scope("inference"):
        image_net = vgg_net(weights, processed_image)
        conv_final_layer = image_net["conv5_3"]

        pool5 = utils.max_pool_2x2(conv_final_layer)

        W6 = utils.weight_variable([7, 7, 512, 4096], name="W6")
        b6 = utils.bias_variable([4096], name="b6")
        conv6 = utils.conv2d_basic(pool5, W6, b6)
        relu6 = tf.nn.relu(conv6, name="relu6")
        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)

        W7 = utils.weight_variable([1, 1, 4096, 4096], name="W7")
        b7 = utils.bias_variable([4096], name="b7")
        conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)
        relu7 = tf.nn.relu(conv7, name="relu7")
        relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)

        W8 = utils.weight_variable([1, 1, 4096, NUM_OF_CLASSES+1], name="W8")
        b8 = utils.bias_variable([NUM_OF_CLASSES+1], name="b8")
        conv8 = utils.conv2d_basic(relu_dropout7, W8, b8)
        # annotation_pred1 = tf.argmax(conv8, dimension=3, name="prediction1")

        # now to upscale to actual image size
        deconv_shape1 = image_net["pool4"].get_shape()
        W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, NUM_OF_CLASSES+1], name="W_t1")
        b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
        conv_t1 = utils.conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(image_net["pool4"]))
        fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")

        deconv_shape2 = image_net["pool3"].get_shape()
        W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
        b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
        conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(image_net["pool3"]))
        fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")

        shape = tf.shape(image)
        deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], NUM_OF_CLASSES+1])
        W_t3 = utils.weight_variable([16, 16, NUM_OF_CLASSES+1, deconv_shape2[3].value], name="W_t3")
        b_t3 = utils.bias_variable([NUM_OF_CLASSES+1], name="b_t3")
        conv_t3 = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)

        annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")

    return tf.expand_dims(annotation_pred, dim=3), conv_t3


def train(loss_val, var_list, g_step):
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    return optimizer.apply_gradients(grads, global_step=g_step)


def main(mode, data_dir, image_path):
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[None, IMSIZE_X, IMSIZE_Y, 3], name="input_image")
    if mode == 'train':
        annotation = tf.placeholder(tf.float32, shape=[None, IMSIZE_X, IMSIZE_Y, NUM_OF_CLASSES+1], name="annotation")

    pred_label, logits = inference(image, keep_probability)

    if mode == 'train':
        tf.summary.image("input_image", image, max_outputs=2)
        gt_label = tf.expand_dims(tf.argmax(annotation, axis=3),dim=3)
        tf.summary.image("ground_truth", tf.cast(gt_label*255/NUM_OF_CLASSES, tf.uint8), max_outputs=2)
        tf.summary.image("pred_label", tf.cast(pred_label*255/NUM_OF_CLASSES, tf.uint8), max_outputs=2)
        
        # Compute loss
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=annotation,name="entropy"))
        tf.summary.scalar("entropy_loss", loss)

        # Compute accuracy
        mask = tf.cast(tf.not_equal(gt_label,NUM_OF_CLASSES), tf.float32)
        pixel_acc = tf.div(tf.reduce_sum(tf.multiply(tf.cast(tf.equal(gt_label, pred_label), tf.float32), mask)), tf.cast(tf.reduce_sum(mask), tf.float32))
        #pixel_acc = tf.div(tf.reduce_sum(tf.cast(tf.equal(gt_label,pred_label), tf.float32)),tf.cast(tf.reduce_sum(annotation), tf.float32))
        tf.summary.scalar("pixel_accuracy", pixel_acc)

        trainable_var = tf.trainable_variables()

        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = train(loss, trainable_var, global_step)

        print("Setting up summary op...")
        summary_op = tf.summary.merge_all()

        print("Setting up image reader...")
        data = load_dataset(data_dir,N_EXAMPLE,N_TRAINING_DATA)

        print("Setting up dataset reader")
        train_records = {'images':data['train_data'], 'annotations':data['train_labels']}
        valid_records = {'images':data['test_data'], 'annotations':data['test_labels']}
    
        train_dataset_reader = batchreader.BatchDatset(dataset=train_records)
        validation_dataset_reader = batchreader.BatchDatset(dataset=valid_records)

    sess = tf.Session()
    print("Setting up Saver...")
    saver = tf.train.Saver()
    writer_valid = tf.summary.FileWriter(LOG_DIR+'valid', sess.graph)
    writer_train = tf.summary.FileWriter(LOG_DIR+'train', sess.graph)

    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(LOG_DIR)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")

    if mode == "train":
        for itr in xrange(MAX_ITERATION):
            train_images, train_annotations = train_dataset_reader.next_batch(BATCH_SIZE)
            feed_dict = {image: train_images, annotation: train_annotations, keep_probability: 0.85}

            sess.run(train_op, feed_dict=feed_dict)
            step = tf.train.global_step(sess, global_step) - 1
            if step % 10 == 0:
                train_loss, train_acc, summary_str = sess.run([loss, pixel_acc, summary_op], feed_dict=feed_dict)
                print("Step: %d, Train_loss:%g, Train_acc:%g" % (step, train_loss, train_acc))
                writer_train.add_summary(summary_str, step)

            if step % 100 == 0:
                valid_images, valid_annotations = validation_dataset_reader.next_batch(BATCH_SIZE)
                valid_feed_dict = {image: valid_images, annotation: valid_annotations, keep_probability: 1.0}
                valid_loss, valid_acc, summary_str = sess.run([loss, pixel_acc, summary_op], feed_dict=valid_feed_dict)
                print("%s ---> Validation_loss:%g, Validation_acc:%g" % (datetime.datetime.now(), valid_loss, valid_acc))
                writer_valid.add_summary(summary_str, step)
                saver.save(sess, LOG_DIR + "model.ckpt", global_step=global_step)

    elif mode == "visualize":
        org_image = np.array(spmi.imresize(Image.open(image_path),(IMSIZE_X,IMSIZE_Y,3), interp='bilinear'))
        pred = sess.run(pred_label, feed_dict={image: np.expand_dims(org_image, axis=0), keep_probability: 1.0})
        pred = np.squeeze(np.squeeze(pred, axis=3), axis=0)
        lab_image = np.zeros((IMSIZE_X,IMSIZE_Y,3))
        for i in range(NUM_OF_CLASSES):
            lab_image[pred==i] = RGB_OF_CLASSES[TRAIN_CLASSES[i]]

        fig, ax = plt.subplots(1, 1)
        plt.axis('off')
        ax.imshow(org_image)
        ax.imshow(lab_image, alpha=0.5)
        dir_path = dirname(__file__)+'/Results/'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        fig.savefig(dir_path + os.path.splitext(image_path.split('/')[-1])[0] + '_seg.png')
        print("Saved image : " + os.path.splitext(image_path.split('/')[-1])[0] + '_seg.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Scene Segmentation')
    parser.add_argument('--mode',type=str,required=True,help='Specify the mode (train, visualize)')
    parser.add_argument('--dataset',type=str,help='Specify the directory of dataset')
    parser.add_argument('--image',type=str,help='Path to the image file')
    args = parser.parse_args()
    if (args.mode == 'train') and (args.dataset is None):
        parser.error('--train requires --dataset')
    if (args.mode == 'visualize') and (args.image is None):
        parser.error('--visualize requires --image')


    main(mode=args.mode, data_dir=args.dataset, image_path=args.image)
    #tf.app.run()


    

