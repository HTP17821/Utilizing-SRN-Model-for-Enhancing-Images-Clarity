import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
import logging
logging.disable(logging.WARNING)
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
import tf_slim as slim
import numpy as np
import cv2


def perceptual_loss(img_gt, img_pred):
    # Load pre-trained VGG19 model
    vgg = VGG19(weights='imagenet', include_top=False)
    # Extract feature maps from specific layers
    content_layers = ['block1_conv2', 'block2_conv2', 'block3_conv2', 'block4_conv2', 'block5_conv2']
    outputs = [vgg.get_layer(name).output for name in content_layers]
    model = Model(inputs=vgg.input, outputs=outputs)
    # Normalize the images to the same range as the VGG19 input
    img_gt = tf.keras.applications.vgg19.preprocess_input(img_gt * 255.0)
    img_pred = tf.keras.applications.vgg19.preprocess_input(img_pred * 255.0)
    # Compute the feature maps
    gt_features = model(img_gt)
    pred_features = model(img_pred)
    # Compute the perceptual loss as the L2 distance between feature maps
    perceptual_loss_value = tf.reduce_mean(
        [tf.reduce_mean((gt_feat - pred_feat) ** 2) for gt_feat, pred_feat in zip(gt_features, pred_features)])
    return perceptual_loss_value

def total_variation_loss(image):
    # Calculate the differences between adjacent pixel values in the vertical direction
    vertical_diff = tf.reduce_sum(tf.abs(image[:, :-1, :, :] - image[:, 1:, :, :]))
    # Calculate the differences between adjacent pixel values in the horizontal direction
    horizontal_diff = tf.reduce_sum(tf.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
    # Sum the vertical and horizontal differences
    total_variation = vertical_diff + horizontal_diff
    return total_variation

def im2uint8(x):
    if x.__class__ == tf.compat.v1.Tensor:
        return tf.compat.v1.cast(tf.clip_by_value(x, 0.0, 1.0) * 255.0, tf.compat.v1.uint8)
    else:
        #print(x.__class__)
        t = np.clip(x, 0.0, 1.0) * 255.0
        return t.astype(np.uint8)


def lap_sharp(img, sharpness):
    kernel = np.array([[0, -1, 0],
                       [-1, 4, -1],
                       [0, -1, 0]])
    #print("Sharpness = " + str(sharpness))
    laplacianImage = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
    g = img + ((laplacianImage * sharpness) + (0.00205952 * 255))
    result = np.clip(g, 0.0, 255.0)
    return result


def resnet_block(x, dim, ksize, scope='rb'):
    with tf.compat.v1.variable_scope(scope):
        net = slim.conv2d(x, dim, [ksize, ksize], scope='conv1')
        net = slim.conv2d(net, dim, [ksize, ksize], activation_fn=None, scope='conv2')
        return net + x