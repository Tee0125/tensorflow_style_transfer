#/usr/bin/env python3

import tensorflow as tf

import scipy.io
import numpy as np
import PIL.Image

VGGNET_FILE = 'imagenet-vgg-verydeep-19.mat'

CONTENT_IMAGE_FILE = 'starry-night.jpg'
STYLE_IMAGE_FILE   = 'starry-night.jpg'

def load_vggnet(filename):
    return scipy.io.loadmat(filename)

def load_images(content_filename, style_filename):
    print(content_filename, 1)
    try:
        content_image = PIL.Image.open(content_filename)
        style_image   = PIL.Image.open(style_filename)
    except:
        print("image open error")
        exit(-1)

    width  = min(content_image.size[0], style_image.size[0])
    height = min(content_image.size[1], style_image.size[1])

    size = (width, height)

    if content_image.size != size:
        image = content_image.resize(size, PIL.Image.LANCZOS) 

    if style_image.size != size:
        style_image = style_image.resize(size, PIL.Image.LANCZOS) 

    return content_image, style_image

def preprocess_image(image):
    # from vggnet paper
    mean_pixel = np.array([123.68, 116.779, 103.939])

    image = np.float32(image) - mean_pixel

    # add extra dimension to meet tensorflow conv api
    shape = (1,) + image.shape

    return np.reshape(image, shape)

def conv_layer(input, param):
    weight = param[0]
    bias   = param[1]

	# code in below is copieded from:
	#	https://github.com/hwalsuklee/tensorflow-style-transfer
	#
    # matconvnet: weights are [width, height, in_channels, out_channels]
    # tensorflow: weights are [height, width, in_channels, out_channels]
    weight = np.transpose(weight, (1, 0, 2, 3))
    bias   = bias.reshape(-1)

    conv = tf.nn.conv2d(input, tf.constant(weight), strides=(1, 1, 1, 1),
            padding='SAME')

    return tf.nn.bias_add(conv, bias)

def pool_layer(input):
    return tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')

def relu_layer(input):
    return tf.nn.relu(input)

def build_vggnet_graph(input, vggnet):
    vgg_layers = vggnet['layers'][0]
    num_vgg_layer = vgg_layers.shape[0]

    layers = [input]
    conv_layers = []

    for i in range(0, num_vgg_layer):
        layer_type = vgg_layers[i][0][0][1]
        layer      = ''

        # extend layer
        if layer_type == 'conv':
            layer = conv_layer(layers[i], vgg_layers[i][0][0][2][0])
        elif layer_type == 'relu':
            layer = relu_layer(layers[i])
        elif layer_type == 'pool':
            layer = pool_layer(layers[i])

        # met unknown layer type 
        assert layer != ''

        layers.append(layer)

        # build enough?
        if layer_type == 'conv':
            conv_layers.append(layer)
 
            if len(conv_layers) >= 5:
                break

    assert len(conv_layers) == 5

    return conv_layers

# load vggnet matrix
vggnet = load_vggnet(VGGNET_FILE)

# load images
content_image, style_image = load_images(CONTENT_IMAGE_FILE, STYLE_IMAGE_FILE)

# preprocess iamges
content_image = preprocess_image(content_image)
style_image   = preprocess_image(style_image)

# build graph
vgg_layers = vggnet['layers'][0]
num_vgg_layer = vgg_layers.shape[0]

shape = content_image.shape

# build graph
input = tf.placeholder(tf.float32, shape=shape)

conv_layers = build_vggnet_graph(input, vggnet)

# prepare tensorflow
sess = tf.Session()

# generate content representation
init = tf.global_variables_initializer()
sess.run(init)

Lcontent = sess.run(conv_layers[3], feed_dict={input: content_image})
#print(np.array(Lcontent).shape)

# generate style representation
Lstyle   = sess.run(conv_layers, feed_dict={input: style_image})

#print(np.array(Lstyle[0]).shape)
#print(np.array(Lstyle[1]).shape)
#print(np.array(Lstyle[2]).shape)
#print(np.array(Lstyle[3]).shape)
#print(np.array(Lstyle[4]).shape)