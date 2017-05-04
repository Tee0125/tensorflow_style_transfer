#/usr/bin/env python

import tensorflow as tf

import scipy.io
import numpy as np
import PIL.Image

"""Helper-functions for image manipulation"""
# This function loads an image and returns it as a numpy array of floating-points.
# The image can be automatically resized so the largest of the height or width equals max_size.
# or resized to the given shape
def load_image(filename, shape=None, max_size=None):
    image = PIL.Image.open(filename)

    if max_size is not None:
        # Calculate the appropriate rescale-factor for
        # ensuring a max height and width, while keeping
        # the proportion between them.
        factor = float(max_size) / np.max(image.size)

        # Scale the image's height and width.
        size = np.array(image.size) * factor

        # The size is now floating-point because it was scaled.
        # But PIL requires the size to be integers.
        size = size.astype(int)

        # Resize the image.
        image = image.resize(size, PIL.Image.LANCZOS) # PIL.Image.LANCZOS is one of resampling filter

    if shape is not None:
        image = image.resize(shape, PIL.Image.LANCZOS) # PIL.Image.LANCZOS is one of resampling filter

    # Convert to numpy floating-point array.
    return np.float32(image)

VGGNET_FILE = 'imagenet-vgg-verydeep-19.mat'
IMAGE_FILE  = 'starry-night.jpg'

def _conv_layer(input, param):
    weight = param[0]
    bias   = param[1]

    # matconvnet: weights are [width, height, in_channels, out_channels]
    # tensorflow: weights are [height, width, in_channels, out_channels]
    weight = np.transpose(weight, (1, 0, 2, 3))
    bias   = bias.reshape(-1)

    conv = tf.nn.conv2d(input, tf.constant(weight), strides=(1, 1, 1, 1),
            padding='SAME')

    return tf.nn.bias_add(conv, bias)

# load image
image = load_image(IMAGE_FILE)
image = image - np.array([123.68, 116.779, 103.939])
shape = (1,) + image.shape
image = np.reshape(image, shape)

# load matrix
vggnet = scipy.io.loadmat(VGGNET_FILE)

# build graph
vgg_layers = vggnet['layers'][0]
num_vgg_layer = vgg_layers.shape[0]

# build graph
layers = []
conv_layers = []

inn = tf.placeholder(tf.float32, shape=image.shape)
layers.append(inn)

for i in range(0, num_vgg_layer):
    layer_type = vgg_layers[i][0][0][1]
    layer      = ''

    # extend layer
    if layer_type == 'conv':
        layer = _conv_layer(layers[i], vgg_layers[i][0][0][2][0])
    elif layer_type == 'relu':
        layer = tf.nn.relu(layers[i])
    elif layer_type == 'pool':
        layer = tf.nn.max_pool(layers[i], ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')

    print(layer_type, layer)

    # met unknown layer type 
    assert layer != ''

    layers.append(layer)

    # epilog
    if layer_type == 'conv':
        conv_layers.append(layer)

        if len(conv_layers) >= 5:
            break

assert len(conv_layers) == 5

sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(conv_layers[-1], feed_dict={inn: image}))
