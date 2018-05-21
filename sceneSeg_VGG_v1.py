from __future__ import print_function
import BatchDatsetReader as batchreader
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
N_EXAMPLE = 1200 # maximum 3475
N_TRAINING_DATA = 1000
LEARNING_RATE = 0.0001
BATCH_SIZE = 5
TRAIN_CLASSES = [0,13] # max: range(19)
NUM_OF_CLASSES = len(TRAIN_CLASSES) 
# ..........................................................................................
LOG_DIR = dirname(__file__)+'/logs/VGG_c'+str(NUM_OF_CLASSES)+'/'
RESULT_DIR = dirname(__file__)+'/Results/VGG_c'+str(NUM_OF_CLASSES)+'/'
# ==========================================================================================

MAX_ITERATION = int(1e5 + 1)
IMSIZE_X = 256
IMSIZE_Y = 512
RGB_OF_CLASSES = {0:(128,54,128),1:(244,35,232),2:(70,70,70),3:(102,102,156),4:(190,153,153),
                5:(153,153,153),6:(250,170,30),7:(220,220,0),8:(107,142,35),9:(152,251,152),
                10:(70,130,180),11:(220,20,60),12:(255,0,0),13:(0,0,142),14:(0,0,70),
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

def inference(image, keep_prob):
    """
    Semantic segmentation network definition
    :param image: input image. Should have values in range 0-255
    :param keep_prob:
    :return:
    """
    print("setting up VGG net ...")

    with tf.variable_scope("inference"):
        # ---------------------------------------- DOWNSAMPLING ---------------------------------------- 
        # Convolutional Layer 1
        W1_1 = tf.get_variable(name='W1_1', initializer=tf.truncated_normal(shape=[3,3,3,64], stddev=0.02))
        b1_1 = tf.get_variable(name='b1_1', initializer=tf.constant(0.0, shape=[64]))
        conv1_1 = tf.nn.bias_add(tf.nn.conv2d(image, W1_1, strides=[1, 1, 1, 1], padding="SAME"), b1_1)
        relu_dropout1_1 = tf.nn.dropout(tf.nn.relu(conv1_1, name="relu1_1"), keep_prob=keep_prob)
        W1_2 = tf.get_variable(name='W1_2', initializer=tf.truncated_normal(shape=[3,3,64,64], stddev=0.02))
        b1_2 = tf.get_variable(name='b1_2', initializer=tf.constant(0.0, shape=[64]))
        conv1_2 = tf.nn.bias_add(tf.nn.conv2d(relu_dropout1_1, W1_2, strides=[1, 1, 1, 1], padding="SAME"), b1_2)
        relu_dropout1_2 = tf.nn.dropout(tf.nn.relu(conv1_2, name="relu1_2"), keep_prob=keep_prob)
        # Pooling Layer 1
        pool1 = tf.nn.max_pool(relu_dropout1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name='pool1')

        # Convolutional Layer 2
        W2_1 = tf.get_variable(name='W2_1', initializer=tf.truncated_normal(shape=[3,3,64,128], stddev=0.02))
        b2_1 = tf.get_variable(name='b2_1', initializer=tf.constant(0.0, shape=[128]))
        conv2_1 = tf.nn.bias_add(tf.nn.conv2d(pool1, W2_1, strides=[1, 1, 1, 1], padding="SAME"), b2_1)
        relu_dropout2_1 = tf.nn.dropout(tf.nn.relu(conv2_1, name="relu2_1"), keep_prob=keep_prob)
        W2_2 = tf.get_variable(name='W2_2', initializer=tf.truncated_normal(shape=[3,3,128,128], stddev=0.02))
        b2_2 = tf.get_variable(name='b2_2', initializer=tf.constant(0.0, shape=[128]))
        conv2_2 = tf.nn.bias_add(tf.nn.conv2d(relu_dropout2_1, W2_2, strides=[1, 1, 1, 1], padding="SAME"), b2_2)
        relu_dropout2_2 = tf.nn.dropout(tf.nn.relu(conv2_2, name="relu2_2"), keep_prob=keep_prob)
        # Pooling Layer 2
        pool2 = tf.nn.max_pool(relu_dropout2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name='pool2')

        # Convolutional Layer 3
        W3_1 = tf.get_variable(name='W3_1', initializer=tf.truncated_normal(shape=[3,3,128,256], stddev=0.02))
        b3_1 = tf.get_variable(name='b3_1', initializer=tf.constant(0.0, shape=[256]))
        conv3_1 = tf.nn.bias_add(tf.nn.conv2d(pool2, W3_1, strides=[1, 1, 1, 1], padding="SAME"), b3_1)
        relu_dropout3_1 = tf.nn.dropout(tf.nn.relu(conv3_1, name="relu3_1"), keep_prob=keep_prob)
        W3_2 = tf.get_variable(name='W3_2', initializer=tf.truncated_normal(shape=[3,3,256,256], stddev=0.02))
        b3_2 = tf.get_variable(name='b3_2', initializer=tf.constant(0.0, shape=[256]))
        conv3_2 = tf.nn.bias_add(tf.nn.conv2d(relu_dropout3_1, W3_2, strides=[1, 1, 1, 1], padding="SAME"), b3_2)
        relu_dropout3_2 = tf.nn.dropout(tf.nn.relu(conv3_2, name="relu3_2"), keep_prob=keep_prob)
        W3_3 = tf.get_variable(name='W3_3', initializer=tf.truncated_normal(shape=[3,3,256,256], stddev=0.02))
        b3_3 = tf.get_variable(name='b3_3', initializer=tf.constant(0.0, shape=[256]))
        conv3_3 = tf.nn.bias_add(tf.nn.conv2d(relu_dropout3_2, W3_3, strides=[1, 1, 1, 1], padding="SAME"), b3_3)
        relu_dropout3_3 = tf.nn.dropout(tf.nn.relu(conv3_3, name="relu3_3"), keep_prob=keep_prob)
        W3_4 = tf.get_variable(name='W3_4', initializer=tf.truncated_normal(shape=[3,3,256,256], stddev=0.02))
        b3_4 = tf.get_variable(name='b3_4', initializer=tf.constant(0.0, shape=[256]))
        conv3_4 = tf.nn.bias_add(tf.nn.conv2d(relu_dropout3_3, W3_4, strides=[1, 1, 1, 1], padding="SAME"), b3_4)
        relu_dropout3_4 = tf.nn.dropout(tf.nn.relu(conv3_4, name="relu3_4"), keep_prob=keep_prob)
        # Pooling Layer 3
        pool3 = tf.nn.max_pool(relu_dropout3_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name='pool3')

        # Convolutional Layer 4
        W4_1 = tf.get_variable(name='W4_1', initializer=tf.truncated_normal(shape=[3,3,256,512], stddev=0.02))
        b4_1 = tf.get_variable(name='b4_1', initializer=tf.constant(0.0, shape=[512]))
        conv4_1 = tf.nn.bias_add(tf.nn.conv2d(pool3, W4_1, strides=[1, 1, 1, 1], padding="SAME"), b4_1)
        relu_dropout4_1 = tf.nn.dropout(tf.nn.relu(conv4_1, name="relu4_1"), keep_prob=keep_prob)
        W4_2 = tf.get_variable(name='W4_2', initializer=tf.truncated_normal(shape=[3,3,512,512], stddev=0.02))
        b4_2 = tf.get_variable(name='b4_2', initializer=tf.constant(0.0, shape=[512]))
        conv4_2 = tf.nn.bias_add(tf.nn.conv2d(relu_dropout4_1, W4_2, strides=[1, 1, 1, 1], padding="SAME"), b4_2)
        relu_dropout4_2 = tf.nn.dropout(tf.nn.relu(conv4_2, name="relu4_2"), keep_prob=keep_prob)
        W4_3 = tf.get_variable(name='W4_3', initializer=tf.truncated_normal(shape=[3,3,512,512], stddev=0.02))
        b4_3 = tf.get_variable(name='b4_3', initializer=tf.constant(0.0, shape=[512]))
        conv4_3 = tf.nn.bias_add(tf.nn.conv2d(relu_dropout4_2, W4_3, strides=[1, 1, 1, 1], padding="SAME"), b4_3)
        relu_dropout4_3 = tf.nn.dropout(tf.nn.relu(conv4_3, name="relu4_3"), keep_prob=keep_prob)
        W4_4 = tf.get_variable(name='W4_4', initializer=tf.truncated_normal(shape=[3,3,512,512], stddev=0.02))
        b4_4 = tf.get_variable(name='b4_4', initializer=tf.constant(0.0, shape=[512]))
        conv4_4 = tf.nn.bias_add(tf.nn.conv2d(relu_dropout4_3, W4_4, strides=[1, 1, 1, 1], padding="SAME"), b4_4)
        relu_dropout4_4 = tf.nn.dropout(tf.nn.relu(conv4_4, name="relu4_4"), keep_prob=keep_prob)
        # Pooling Layer 4
        pool4 = tf.nn.max_pool(relu_dropout4_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name='pool4')

        # Convolutional Layer 5
        W5_1 = tf.get_variable(name='W5_1', initializer=tf.truncated_normal(shape=[3,3,512,512], stddev=0.02))
        b5_1 = tf.get_variable(name='b5_1', initializer=tf.constant(0.0, shape=[512]))
        conv5_1 = tf.nn.bias_add(tf.nn.conv2d(pool4, W5_1, strides=[1, 1, 1, 1], padding="SAME"), b5_1)
        relu_dropout5_1 = tf.nn.dropout(tf.nn.relu(conv5_1, name="relu5_1"), keep_prob=keep_prob)
        W5_2 = tf.get_variable(name='W5_2', initializer=tf.truncated_normal(shape=[3,3,512,512], stddev=0.02))
        b5_2 = tf.get_variable(name='b5_2', initializer=tf.constant(0.0, shape=[512]))
        conv5_2 = tf.nn.bias_add(tf.nn.conv2d(relu_dropout5_1, W5_2, strides=[1, 1, 1, 1], padding="SAME"), b5_2)
        relu_dropout5_2 = tf.nn.dropout(tf.nn.relu(conv5_2, name="relu5_2"), keep_prob=keep_prob)
        W5_3 = tf.get_variable(name='W5_3', initializer=tf.truncated_normal(shape=[3,3,512,512], stddev=0.02))
        b5_3 = tf.get_variable(name='b5_3', initializer=tf.constant(0.0, shape=[512]))
        conv5_3 = tf.nn.bias_add(tf.nn.conv2d(relu_dropout5_2, W5_3, strides=[1, 1, 1, 1], padding="SAME"), b5_3)
        relu_dropout5_3 = tf.nn.dropout(tf.nn.relu(conv5_3, name="relu5_3"), keep_prob=keep_prob)
        W5_4 = tf.get_variable(name='W5_4', initializer=tf.truncated_normal(shape=[3,3,512,512], stddev=0.02))
        b5_4 = tf.get_variable(name='b5_4', initializer=tf.constant(0.0, shape=[512]))
        conv5_4 = tf.nn.bias_add(tf.nn.conv2d(relu_dropout5_3, W5_4, strides=[1, 1, 1, 1], padding="SAME"), b5_4)
        relu_dropout5_4 = tf.nn.dropout(tf.nn.relu(conv5_4, name="relu4_4"), keep_prob=keep_prob)
        # Pooling Layer 5
        pool5 = tf.nn.max_pool(relu_dropout5_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name='pool5')

        # Convolutional Layer 6
        W6 = tf.get_variable(name='W6', initializer=tf.truncated_normal(shape=[7,7,512,4096], stddev=0.02))
        b6 = tf.get_variable(name='b6', initializer=tf.constant(0.0, shape=[4096]))
        conv6 = tf.nn.bias_add(tf.nn.conv2d(pool5, W6, strides=[1, 1, 1, 1], padding="SAME"), b6)
        relu_dropout6 = tf.nn.dropout(tf.nn.relu(conv6, name="relu6"), keep_prob=keep_prob)
        
        # Convolutional Layer 7
        W7 = tf.get_variable(name='W7', initializer=tf.truncated_normal(shape=[1,1,4096,4096], stddev=0.02))
        b7 = tf.get_variable(name='b7', initializer=tf.constant(0.0, shape=[4096]))
        conv7 = tf.nn.bias_add(tf.nn.conv2d(relu_dropout6, W7, strides=[1, 1, 1, 1], padding="SAME"), b7)
        relu_dropout7 = tf.nn.dropout(tf.nn.relu(conv7, name="relu7"), keep_prob=keep_prob)
        
        # Convolutional Layer 8
        W8 = tf.get_variable(name='W8', initializer=tf.truncated_normal(shape=[1,1,4096,NUM_OF_CLASSES+1], stddev=0.02))
        b8 = tf.get_variable(name='b8', initializer=tf.constant(0.0, shape=[NUM_OF_CLASSES+1]))
        conv8 = tf.nn.bias_add(tf.nn.conv2d(relu_dropout7, W8, strides=[1, 1, 1, 1], padding="SAME"), b8)
        
        # ---------------------------------------- UPSAMPLING ---------------------------------------- 
        # Deconvolution Layer 1
        deconv_shape1 = pool4.get_shape()
        W_t1 = tf.get_variable(name='W_t1', initializer=tf.truncated_normal(shape=[4, 4, deconv_shape1[3].value, NUM_OF_CLASSES+1], stddev=0.02))
        b_t1 = tf.get_variable(name='b_t1', initializer=tf.constant(0.0, shape=[deconv_shape1[3].value]))
        conv_t1 = tf.nn.bias_add(tf.nn.conv2d_transpose(conv8, W_t1, output_shape=tf.shape(pool4), strides=[1, 2, 2, 1], padding="SAME"), b_t1)
        skip_1 = tf.add(conv_t1, pool4, name="skip_1")

        # Deconvolution Layer 2
        deconv_shape2 = pool3.get_shape()
        W_t2 = tf.get_variable(name='W_t2', initializer=tf.truncated_normal(shape=[4, 4, deconv_shape2[3].value, deconv_shape1[3].value], stddev=0.02))
        b_t2 = tf.get_variable(name='b_t2', initializer=tf.constant(0.0, shape=[deconv_shape2[3].value]))
        conv_t2 = tf.nn.bias_add(tf.nn.conv2d_transpose(skip_1, W_t2, output_shape=tf.shape(pool3), strides=[1, 2, 2, 1], padding="SAME"), b_t2)
        skip_2 = tf.add(conv_t2, pool3, name="skip_2")

        # Deconvolution Layer 3
        shape = tf.shape(image)
        deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], NUM_OF_CLASSES+1])
        W_t3 = tf.get_variable(name='W_t3', initializer=tf.truncated_normal(shape=[16, 16, NUM_OF_CLASSES+1, deconv_shape2[3].value], stddev=0.02))
        b_t3 = tf.get_variable(name='b_t3', initializer=tf.constant(0.0, shape=[NUM_OF_CLASSES+1]))
        conv_t3 = tf.nn.bias_add(tf.nn.conv2d_transpose(skip_2, W_t3, output_shape=deconv_shape3, strides=[1, 8, 8, 1], padding="SAME"), b_t3)

        annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")

    return tf.expand_dims(annotation_pred, dim=3), conv_t3


def train(loss_val, var_list, g_step):
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    return optimizer.apply_gradients(grads, global_step=g_step)


def main(mode, data_dir, image_path, image_dir):
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
        if image_path is not None:
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
            fig.savefig(RESULT_DIR + os.path.splitext(image_path.split('/')[-1])[0] + '_seg.png',dpi=200, transparent=True)
            print("Saved image : " + RESULT_DIR + os.path.splitext(image_path.split('/')[-1])[0] + '_seg.png')

        if image_dir is not None:
            if not os.path.exists(image_dir+'/Results/'):
                os.makedirs(image_dir+'/Results/')
            for fname in os.listdir(image_dir):
                if os.path.splitext(fname)[-1]=='.jpg':
                    f = os.path.join(image_dir,fname)
                    org_image = np.array(spmi.imresize(Image.open(f),(IMSIZE_X,IMSIZE_Y,3), interp='bilinear'))
                    pred = sess.run(pred_label, feed_dict={image: np.expand_dims(org_image, axis=0), keep_probability: 1.0})
                    pred = np.squeeze(np.squeeze(pred, axis=3), axis=0)
                    lab_image = np.zeros((IMSIZE_X,IMSIZE_Y,3))
                    for i in range(NUM_OF_CLASSES):
                        lab_image[pred==i] = RGB_OF_CLASSES[TRAIN_CLASSES[i]]

                    fig, ax = plt.subplots(1, 1)
                    plt.axis('off')
                    ax.imshow(org_image)
                    ax.imshow(lab_image, alpha=0.5)
                    fig.savefig(image_dir+'/Results/' + os.path.splitext(fname.split('/')[-1])[0] + '_seg.png',dpi=200, transparent=True)
                    print("Saved image : " + args.imagedir + '/Results/' + os.path.splitext(fname.split('/')[-1])[0] + '_seg.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Scene Segmentation')
    parser.add_argument('--mode',type=str,required=True,help='Specify the mode (train, visualize)')
    parser.add_argument('--dataset',type=str,help='Specify the directory of dataset')
    parser.add_argument('--image',type=str,help='Path to the image file')
    parser.add_argument('--imagedir',type=str,help='Directory to the image folder')
    args = parser.parse_args()
    if (args.mode == 'train') and (args.dataset is None):
        parser.error('--train requires --dataset')
    if (args.mode == 'visualize') and ((args.image is None) and (args.imagedir is None)):
        parser.error('--visualize requires --image/--imagedir')

    # Create folders
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    main(mode=args.mode, data_dir=args.dataset, image_path=args.image, image_dir=args.imagedir)
    #tf.app.run()


    

