import os.path
from glob import glob
import numpy as np
import random
import re
import scipy.misc
import skimage.exposure
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    vgg = tf.get_default_graph()
    return (vgg.get_tensor_by_name(vgg_input_tensor_name),
        vgg.get_tensor_by_name(vgg_keep_prob_tensor_name),
        vgg.get_tensor_by_name(vgg_layer3_out_tensor_name),
        vgg.get_tensor_by_name(vgg_layer4_out_tensor_name),
        vgg.get_tensor_by_name(vgg_layer7_out_tensor_name))
tests.test_load_vgg(load_vgg, tf)

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes,
    kernel_size_3=8, kernel_size_4=4, kernel_size_7=4, conv_1x1_depth=0):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    if conv_1x1_depth > 0:
        input_7 = tf.layers.conv2d(vgg_layer7_out, conv_1x1_depth, 1, 1)
    else:
        input_7 = vgg_layer7_out

    vgg_layer4_depth = vgg_layer4_out.get_shape()[3]
    transpose_7 = tf.layers.conv2d_transpose(
      vgg_layer7_out, vgg_layer4_depth, kernel_size_7, 2, padding='SAME')
    out_7 = transpose_7 + vgg_layer4_out

    vgg_layer3_depth = vgg_layer3_out.get_shape()[3]
    transpose_4 = tf.layers.conv2d_transpose(
      out_7, vgg_layer3_depth, kernel_size_4, 2, padding='SAME')
    out_4 = transpose_4 + vgg_layer3_out

    transpose_3 = tf.layers.conv2d_transpose(
      out_4, num_classes, kernel_size_3, 8, padding='SAME')

    return transpose_3
tests.test_layers(layers)

def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFlow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    logits = tf.reshape(nn_last_layer, [-1, num_classes])
    onehot_labels = tf.reshape(correct_label, [-1, num_classes])
    cross_entropy_loss = tf.losses.softmax_cross_entropy(onehot_labels, logits)

    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)

def train_nn(sess, epochs, batch_size, get_batches_fn, train_op,
    cross_entropy_loss, input_image, correct_label, keep_prob, learning_rate,
    keep_prob_value=0.5, learning_rate_value=0.001):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    for epoch in range(epochs):
        batch = 0
        for inputs, labels in get_batches_fn(batch_size):
            batch += 1
            loss, _ = sess.run(
                [cross_entropy_loss, train_op], {
                    input_image: inputs,
                    correct_label: labels,
                    keep_prob: keep_prob_value,
                    learning_rate: learning_rate_value
                }
            )
            print(epoch, batch, loss)
tests.test_train_nn(train_nn)

DATA_DIR = './data'
VGG_PATH = os.path.join(DATA_DIR, 'vgg')
RUNS_DIR = './runs'
NUM_CLASSES = 2
IMAGE_SHAPE = (160, 576)

#
# Build NN using load_vgg, layers, and optimize functions.
#
def build(sess, layer_params={}):
    image_input, keep_prob, \
        vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = \
        load_vgg(sess, VGG_PATH)

    output_layer = layers(
        vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, NUM_CLASSES,
        **layer_params)

    correct_label = tf.placeholder(
        tf.float32,
        (None, IMAGE_SHAPE[0], IMAGE_SHAPE[1], NUM_CLASSES)
    )
    learning_rate = tf.placeholder(tf.float32, ())
    logits, train_op, cross_entropy_loss = optimize(
        output_layer, correct_label, learning_rate, NUM_CLASSES)

    return train_op, cross_entropy_loss, image_input, correct_label, \
        keep_prob, learning_rate, logits

def identity(image, gt_image):
    return image, gt_image

def reflect(image, gt_image):
    return image[:,::-1,:], gt_image[:,::-1,:]

def darken(image, gt_image):
    return skimage.exposure.adjust_gamma(image, 2), gt_image

def darken_reflect(image, gt_image):
    image, gt_image = darken(image, gt_image)
    return reflect(image, gt_image)

def lighten(image, gt_image):
    return skimage.exposure.adjust_gamma(image, 0.5), gt_image

def lighten_reflect(image, gt_image):
    image, gt_image = lighten(image, gt_image)
    return reflect(image, gt_image)

AUGMENTATION_TRANSFORMS = [
    identity, reflect, darken, darken_reflect, lighten, lighten_reflect]

#
# Return a generator that yields augmented training data. See the notebook
# for examples. Based on helper.gen_batch_function.
#
def gen_augmented_batch_function(data_folder):
    def get_augmented_batches_fn(batch_size):
        image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
        label_paths = {
            re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
            for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}
        background_color = np.array([255, 0, 0])

        augmented_image_paths = [
            (image_path, augmentation_transform)
            for image_path in image_paths
            for augmentation_transform in AUGMENTATION_TRANSFORMS
        ]

        random.shuffle(augmented_image_paths)
        for batch_i in range(0, len(augmented_image_paths), batch_size):
            images = []
            gt_images = []
            for image_file, transform in augmented_image_paths[batch_i:batch_i+batch_size]:
                gt_image_file = label_paths[os.path.basename(image_file)]

                image = scipy.misc.imresize(scipy.misc.imread(image_file), IMAGE_SHAPE)
                gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), IMAGE_SHAPE)

                image, gt_image = transform(image, gt_image)

                gt_bg = np.all(gt_image == background_color, axis=2)
                gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
                gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)

                images.append(image)
                gt_images.append(gt_image)

            yield np.array(images), np.array(gt_images)
    return get_augmented_batches_fn

def run(num_epochs=25, batch_size=13, augment=True):
    tests.test_for_kitti_dataset(DATA_DIR)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(DATA_DIR)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    config = tf.ConfigProto()
    config.graph_options.optimizer_options.global_jit_level = \
        tf.OptimizerOptions.ON_1
    with tf.Session(config=config) as sess:
        # Create function to get batches
        data_folder = os.path.join(DATA_DIR, 'data_road/training')
        if augment:
            get_batches_fn = gen_augmented_batch_function(data_folder)
        else:
            get_batches_fn = helper.gen_batch_function(data_folder, IMAGE_SHAPE)

        train_op, cross_entropy_loss, image_input, correct_label, \
            keep_prob, learning_rate, logits = build(sess)

        sess.run(tf.global_variables_initializer())
        train_nn(
            sess,
            num_epochs,
            batch_size,
            get_batches_fn,
            train_op,
            cross_entropy_loss,
            image_input,
            correct_label,
            keep_prob,
            learning_rate
        )

        helper.save_inference_samples(
            RUNS_DIR, DATA_DIR, sess,
            IMAGE_SHAPE, logits, keep_prob, image_input)

        # OPTIONAL: Apply the trained model to a video

if __name__ == '__main__':
    run()
