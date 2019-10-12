import sys
sys.path.append('../') #root path
from VDSR.modules.ops import *
from VDSR.utils.other_utils import *
import tensorflow as tf


def Concatenation(layers):
    return tf.concat(layers, axis=3)


def SkipConnect(conv):
    skipconv = list()
    for i in conv:
        x = Concatenation(i)
        skipconv.append(x)
    return skipconv



def SRDense(input, scope_name, num_channels, reuse=tf.AUTO_REUSE, is_training=False, scale = 2):
    #initializer = tf.initializers.VarianceScaling()
    initializer = tf.contrib.layers.xavier_initializer()

    def desBlock(desBlock_layer, outlayer, name=None, filter_size=3):
        with tf.variable_scope(name):
            nextlayer = low_conv
            conv = list()
            for i in range(1, outlayer + 1):
                conv_in = list()
                for j in range(1, desBlock_layer + 1):
                    # The first conv need connect with low level layer
                    #print(i, j)
                    if j is 1:
                        x = conv2D(nextlayer, name="conv{}{}".format(i, j), ksize=filter_size, strides=1, fsize=64,
                                   is_training=is_training, padding="SAME", initializer=initializer)
                        x = tf.nn.relu(x)
                        conv_in.append(x)
                    else:
                        x = Concatenation(conv_in)
                        x = conv2D(x, name="conv{}{}".format(i, j), ksize=filter_size, strides=1, fsize=64,
                                   is_training=is_training, padding="SAME", initializer=initializer)
                        x = tf.nn.relu(x)
                        conv_in.append(x)

                nextlayer = conv_in[-1]
                #print(conv_in[-1])
                conv.append(conv_in)
            #print(conv)
            return conv

    def bot_layer(input_layer,name):
        x = conv2D(input_layer, name="conv_{}".format(name), ksize=3, strides=1, fsize=64, is_training=is_training, padding="SAME",
                   initializer=initializer)
        x = tf.nn.relu(x)
        return x

    def resize(x, scale=2.0):
        return tf.image.resize_images(
            x, [tf.cast(scale * tf.cast(tf.shape(x)[1], tf.float32), tf.int32),
                tf.cast(scale * tf.cast(tf.shape(x)[2], tf.float32), tf.int32)],
            align_corners=True,
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    def deconv_layer(input_layer, name, scale = 2):
        x = resize(input_layer, scale=scale)
        x = conv2D(x, name="conv_{}".format(name), ksize=3, strides=1, fsize=64, is_training=is_training, padding="SAME", initializer=initializer)
        x = conv2D(x, name="conv_{}".format(name), ksize=3, strides=1, fsize=64, is_training=is_training, padding="SAME", initializer=initializer)
        x = tf.nn.relu(x)
        return x

    def reconv_layer(input_layer, name):
        x = conv2D(input_layer, name="conv_{}".format(name), ksize=3, strides=1, fsize=num_channels, is_training=is_training, padding="SAME",
                   initializer=initializer)
        return x


    with tf.variable_scope(scope_name, reuse=reuse):
        low_conv = tf.nn.relu(conv2D(input, name="conv_low", ksize=3, strides=1, fsize=16, is_training=is_training, padding="SAME", initializer=initializer))

        x = desBlock(8, 8, filter_size=3, name="dense")
        # NOTE: Cocate all dense block

        x = SkipConnect(x)
        x.append(low_conv)
        x = Concatenation(x)
        x = bot_layer(x, name = "bot1")
        x = deconv_layer(x, name = "deconv1", scale = scale)
        x = reconv_layer(x, name = "recon1")

    return x


