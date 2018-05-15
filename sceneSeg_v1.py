from __future__ import print_function
import BatchDatsetReader as batchreader
import TensorflowUtils as utils
from six.moves import xrange
from PIL import Image
import scipy.misc as spmi
import scipy.io as spio
import tensorflow as tf
import numpy as np
import argparse
import datetime
import glob 
import tqdm
import os
from os.path import dirname

"""
Reference: https://github.com/shekkizh/FCN.tensorflow
"""

#FLAGS = tf.flags.FLAGS
#tf.flags.DEFINE_string("data_dir", "CityscapesDataset", "path to dataset")
#tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")
#tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
#tf.flags.DEFINE_string('mode', "train", "Mode train/ visualize")

#MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'
MODEL_DIR = dirname(__file__)+'/Model/imagenet-vgg-verydeep-19.mat'
LOG_DIR = dirname(__file__)+'/logs/'
DEBUG = False

MAX_ITERATION = int(1e5 + 1)
LEARNING_RATE = 0.0001
BATCH_SIZE = 2
NUM_OF_CLASSESS = 2 #maximum 18
N_EXAMPLE = 50 # maximum 3475
N_TRAINING_DATA = 40
IMSIZE_X = 256
IMSIZE_Y = 512

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
        # load labels --- N x H x W x C(20)
        lb_fn = os.path.splitext(im_fpath[n].split('/')[-1])[0][0:-12] + '_gtFine_color.mat'
        lab = np.array(spio.loadmat(dataset_path+"/gtFine/"+lb_fn)['label'])
        lab_other = (np.sum(lab[:,:,0:NUM_OF_CLASSESS], axis=2)==0).astype(int)
        labelset.append(np.concatenate((lab[:,:,0:NUM_OF_CLASSESS],np.expand_dims(lab_other, axis=2)),axis=2))
        # load images --- N x H x W x 3
        image = Image.open(im_fpath[n])
        imageset.append(np.array(spmi.imresize(image, (image.size[1]/4,image.size[0]/4,3), interp='bilinear')))

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
            if DEBUG:
                utils.add_activation_summary(current)
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
    model_data = utils.get_model_data(MODEL_DIR)#, MODEL_URL)

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
        if DEBUG:
            utils.add_activation_summary(relu6)
        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)

        W7 = utils.weight_variable([1, 1, 4096, 4096], name="W7")
        b7 = utils.bias_variable([4096], name="b7")
        conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)
        relu7 = tf.nn.relu(conv7, name="relu7")
        if DEBUG:
            utils.add_activation_summary(relu7)
        relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)

        W8 = utils.weight_variable([1, 1, 4096, NUM_OF_CLASSESS+1], name="W8")
        b8 = utils.bias_variable([NUM_OF_CLASSESS+1], name="b8")
        conv8 = utils.conv2d_basic(relu_dropout7, W8, b8)
        # annotation_pred1 = tf.argmax(conv8, dimension=3, name="prediction1")

        # now to upscale to actual image size
        deconv_shape1 = image_net["pool4"].get_shape()
        W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, NUM_OF_CLASSESS+1], name="W_t1")
        b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
        conv_t1 = utils.conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(image_net["pool4"]))
        fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")

        deconv_shape2 = image_net["pool3"].get_shape()
        W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
        b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
        conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(image_net["pool3"]))
        fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")

        shape = tf.shape(image)
        deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], NUM_OF_CLASSESS+1])
        W_t3 = utils.weight_variable([16, 16, NUM_OF_CLASSESS+1, deconv_shape2[3].value], name="W_t3")
        b_t3 = utils.bias_variable([NUM_OF_CLASSESS+1], name="b_t3")
        conv_t3 = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)

        annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")

    return tf.expand_dims(annotation_pred+1, dim=3), conv_t3


def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    if DEBUG:
        # print(len(var_list))
        for grad, var in grads:
            utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads)


def main(mode, data_dir):
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[None, IMSIZE_X, IMSIZE_Y, 3], name="input_image")
    annotation = tf.placeholder(tf.float32, shape=[None, IMSIZE_X, IMSIZE_Y, NUM_OF_CLASSESS+1], name="annotation")

    pred_annotation, logits = inference(image, keep_probability)
    tf.summary.image("input_image", image, max_outputs=2)
    tf.summary.image("ground_truth", tf.cast(tf.expand_dims(tf.argmax(annotation, axis=3)+1,dim=3)*255/19, tf.uint8), max_outputs=2)
    tf.summary.image("pred_annotation", tf.cast(pred_annotation*255/19, tf.uint8), max_outputs=2)
    #loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=tf.squeeze(annotation, squeeze_dims=[3]),name="entropy")))
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=annotation,name="entropy"))
    tf.summary.scalar("entropy", loss)

    trainable_var = tf.trainable_variables()
    if DEBUG:
        for var in trainable_var:
            utils.add_to_regularization_and_summary(var)
    train_op = train(loss, trainable_var)

    print("Setting up summary op...")
    summary_op = tf.summary.merge_all()

    print("Setting up image reader...")
    data = load_dataset(data_dir,N_EXAMPLE,N_TRAINING_DATA)

    print("Setting up dataset reader")
    train_records = {'images':data['train_data'], 'annotations':data['train_labels']}
    valid_records = {'images':data['test_data'], 'annotations':data['test_labels']}
    if mode == 'train':
        train_dataset_reader = batchreader.BatchDatset(dataset=train_records)
    validation_dataset_reader = batchreader.BatchDatset(dataset=valid_records)

    sess = tf.Session()

    print("Setting up Saver...")
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)

    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(LOG_DIR)
    if ckpt and ckpt.model_checkpoint_path:
        print(ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")

    if mode == "train":
        for itr in xrange(MAX_ITERATION):
            train_images, train_annotations = train_dataset_reader.next_batch(BATCH_SIZE)
            feed_dict = {image: train_images, annotation: train_annotations, keep_probability: 0.85}

            sess.run(train_op, feed_dict=feed_dict)

            if itr % 10 == 0:
                train_loss, summary_str = sess.run([loss, summary_op], feed_dict=feed_dict)
                print("Step: %d, Train_loss:%g" % (itr, train_loss))
                summary_writer.add_summary(summary_str, itr)

            if itr % 500 == 0:
                valid_images, valid_annotations = validation_dataset_reader.next_batch(BATCH_SIZE)
                valid_loss = sess.run(loss, feed_dict={image: valid_images, annotation: valid_annotations,
                                                       keep_probability: 1.0})
                print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))
                saver.save(sess, LOG_DIR + "model.ckpt", itr)

    elif mode == "visualize":
        valid_images, valid_annotations = validation_dataset_reader.get_random_batch(BATCH_SIZE)
        pred = sess.run(pred_annotation, feed_dict={image: valid_images, annotation: valid_annotations,
                                                    keep_probability: 1.0})
        valid_annotations = np.squeeze(valid_annotations, axis=3)
        pred = np.squeeze(pred, axis=3)

        for itr in range(BATCH_SIZE):
            utils.save_image(valid_images[itr].astype(np.uint8), LOG_DIR, name="inp_" + str(5+itr))
            utils.save_image(valid_annotations[itr].astype(np.uint8), LOG_DIR, name="gt_" + str(5+itr))
            utils.save_image(pred[itr].astype(np.uint8), LOG_DIR, name="pred_" + str(5+itr))
            print("Saved image: %d" % itr)


if __name__ == "__main__":
    print(os.getcwd()+"/Model_zoo/")
    parser = argparse.ArgumentParser(description='Scene Segmentation')
    parser.add_argument('--dataset',type=str,help='Specify the directory of dataset')
    parser.add_argument('--mode',type=str,required=True,help='Specify the mode (train, visualize)')
    args = parser.parse_args()
    if (args.dataset is None):
        parser.error('requires --dataset')
    if (args.mode is None):
        parser.error('requires --mode (train/visualize)')


    main(mode=args.mode, data_dir=args.dataset)
    #tf.app.run()


    

