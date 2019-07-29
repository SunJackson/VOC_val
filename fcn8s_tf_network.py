import tensorflow as tf

__weights_dict = dict()

is_train = False


def load_weights(weight_file):
    import numpy as np

    if weight_file == None:
        return

    try:
        weights_dict = np.load(weight_file).item()
    except:
        weights_dict = np.load(weight_file, encoding='bytes', allow_pickle=True).item()

    return weights_dict


def KitModel(weight_file=None):
    global __weights_dict
    __weights_dict = load_weights(weight_file)

    input = tf.placeholder(tf.float32, shape=(None, 500, 500, 3), name='input')
    conv1_1_pad = tf.pad(input, paddings=[[0L, 0L], [100L, 100L], [100L, 100L], [0L, 0L]])
    conv1_1 = convolution(conv1_1_pad, group=1, strides=[1, 1], padding='VALID', name='conv1_1')
    relu1_1 = tf.nn.relu(conv1_1, name='relu1_1')
    conv1_2_pad = tf.pad(relu1_1, paddings=[[0L, 0L], [1L, 1L], [1L, 1L], [0L, 0L]])
    conv1_2 = convolution(conv1_2_pad, group=1, strides=[1, 1], padding='VALID', name='conv1_2')
    relu1_2 = tf.nn.relu(conv1_2, name='relu1_2')
    pool1_pad = tf.pad(relu1_2, paddings=[[0L, 0L], [0L, 1L], [0L, 1L], [0L, 0L]], constant_values=float('-Inf'))
    pool1 = tf.nn.max_pool(pool1_pad, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID', name='pool1')
    conv2_1_pad = tf.pad(pool1, paddings=[[0L, 0L], [1L, 1L], [1L, 1L], [0L, 0L]])
    conv2_1 = convolution(conv2_1_pad, group=1, strides=[1, 1], padding='VALID', name='conv2_1')
    relu2_1 = tf.nn.relu(conv2_1, name='relu2_1')
    conv2_2_pad = tf.pad(relu2_1, paddings=[[0L, 0L], [1L, 1L], [1L, 1L], [0L, 0L]])
    conv2_2 = convolution(conv2_2_pad, group=1, strides=[1, 1], padding='VALID', name='conv2_2')
    relu2_2 = tf.nn.relu(conv2_2, name='relu2_2')
    pool2_pad = tf.pad(relu2_2, paddings=[[0L, 0L], [0L, 1L], [0L, 1L], [0L, 0L]], constant_values=float('-Inf'))
    pool2 = tf.nn.max_pool(pool2_pad, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID', name='pool2')
    conv3_1_pad = tf.pad(pool2, paddings=[[0L, 0L], [1L, 1L], [1L, 1L], [0L, 0L]])
    conv3_1 = convolution(conv3_1_pad, group=1, strides=[1, 1], padding='VALID', name='conv3_1')
    relu3_1 = tf.nn.relu(conv3_1, name='relu3_1')
    conv3_2_pad = tf.pad(relu3_1, paddings=[[0L, 0L], [1L, 1L], [1L, 1L], [0L, 0L]])
    conv3_2 = convolution(conv3_2_pad, group=1, strides=[1, 1], padding='VALID', name='conv3_2')
    relu3_2 = tf.nn.relu(conv3_2, name='relu3_2')
    conv3_3_pad = tf.pad(relu3_2, paddings=[[0L, 0L], [1L, 1L], [1L, 1L], [0L, 0L]])
    conv3_3 = convolution(conv3_3_pad, group=1, strides=[1, 1], padding='VALID', name='conv3_3')
    relu3_3 = tf.nn.relu(conv3_3, name='relu3_3')
    pool3_pad = tf.pad(relu3_3, paddings=[[0L, 0L], [0L, 1L], [0L, 1L], [0L, 0L]], constant_values=float('-Inf'))
    pool3 = tf.nn.max_pool(pool3_pad, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID', name='pool3')
    conv4_1_pad = tf.pad(pool3, paddings=[[0L, 0L], [1L, 1L], [1L, 1L], [0L, 0L]])
    conv4_1 = convolution(conv4_1_pad, group=1, strides=[1, 1], padding='VALID', name='conv4_1')
    score_pool3 = convolution(pool3, group=1, strides=[1, 1], padding='VALID', name='score_pool3')
    relu4_1 = tf.nn.relu(conv4_1, name='relu4_1')
    conv4_2_pad = tf.pad(relu4_1, paddings=[[0L, 0L], [1L, 1L], [1L, 1L], [0L, 0L]])
    conv4_2 = convolution(conv4_2_pad, group=1, strides=[1, 1], padding='VALID', name='conv4_2')
    relu4_2 = tf.nn.relu(conv4_2, name='relu4_2')
    conv4_3_pad = tf.pad(relu4_2, paddings=[[0L, 0L], [1L, 1L], [1L, 1L], [0L, 0L]])
    conv4_3 = convolution(conv4_3_pad, group=1, strides=[1, 1], padding='VALID', name='conv4_3')
    relu4_3 = tf.nn.relu(conv4_3, name='relu4_3')
    pool4_pad = tf.pad(relu4_3, paddings=[[0L, 0L], [0L, 1L], [0L, 1L], [0L, 0L]], constant_values=float('-Inf'))
    pool4 = tf.nn.max_pool(pool4_pad, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID', name='pool4')
    conv5_1_pad = tf.pad(pool4, paddings=[[0L, 0L], [1L, 1L], [1L, 1L], [0L, 0L]])
    conv5_1 = convolution(conv5_1_pad, group=1, strides=[1, 1], padding='VALID', name='conv5_1')
    score_pool4 = convolution(pool4, group=1, strides=[1, 1], padding='VALID', name='score_pool4')
    relu5_1 = tf.nn.relu(conv5_1, name='relu5_1')
    conv5_2_pad = tf.pad(relu5_1, paddings=[[0L, 0L], [1L, 1L], [1L, 1L], [0L, 0L]])
    conv5_2 = convolution(conv5_2_pad, group=1, strides=[1, 1], padding='VALID', name='conv5_2')
    relu5_2 = tf.nn.relu(conv5_2, name='relu5_2')
    conv5_3_pad = tf.pad(relu5_2, paddings=[[0L, 0L], [1L, 1L], [1L, 1L], [0L, 0L]])
    conv5_3 = convolution(conv5_3_pad, group=1, strides=[1, 1], padding='VALID', name='conv5_3')
    relu5_3 = tf.nn.relu(conv5_3, name='relu5_3')
    pool5_pad = tf.pad(relu5_3, paddings=[[0L, 0L], [0L, 1L], [0L, 1L], [0L, 0L]], constant_values=float('-Inf'))
    pool5 = tf.nn.max_pool(pool5_pad, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID', name='pool5')
    fc6 = convolution(pool5, group=1, strides=[1, 1], padding='VALID', name='fc6')
    relu6 = tf.nn.relu(fc6, name='relu6')
    fc7 = convolution(relu6, group=1, strides=[1, 1], padding='VALID', name='fc7')
    relu7 = tf.nn.relu(fc7, name='relu7')
    score_fr = convolution(relu7, group=1, strides=[1, 1], padding='VALID', name='score_fr')
    upscore2 = convolution_transpose(score_fr, output_shape=[1, 34L, 34L, 21L], strides=[1L, 2L, 2L, 1L],
                                     padding='VALID', name='upscore2')
    score_pool4c = tf.image.crop_to_bounding_box(score_pool4, offset_height=5, offset_width=5, target_height=34,
                                                 target_width=34)
    fuse_pool4 = upscore2 + score_pool4c
    upscore_pool4 = convolution_transpose(fuse_pool4, output_shape=[1, 70L, 70L, 21L], strides=[1L, 2L, 2L, 1L],
                                          padding='VALID', name='upscore_pool4')
    score_pool3c = tf.image.crop_to_bounding_box(score_pool3, offset_height=9, offset_width=9, target_height=70,
                                                 target_width=70)
    fuse_pool3 = upscore_pool4 + score_pool3c
    upscore8 = convolution_transpose(fuse_pool3, output_shape=[1, 568L, 568L, 21L], strides=[1L, 8L, 8L, 1L],
                                     padding='VALID', name='upscore8')
    score = tf.image.crop_to_bounding_box(upscore8, offset_height=31, offset_width=31, target_height=500,
                                          target_width=500)
    return input, score


def convolution_transpose(input, name, **kwargs):
    w = tf.Variable(__weights_dict[name]['weights'], trainable=is_train, name=name + "_weight")
    dim = __weights_dict[name]['weights'].ndim - 2
    if dim == 2:
        layer = tf.nn.conv2d_transpose(input, w, **kwargs)
    elif dim == 3:
        layer = tf.nn.conv3d_transpose(input, w, **kwargs)
    else:
        raise ValueError("Error dim number {} in ConvTranspose".format(dim))

    if 'bias' in __weights_dict[name]:
        b = tf.Variable(__weights_dict[name]['bias'], trainable=is_train, name=name + "_bias")
        layer = layer + b
    return layer


def convolution(input, name, group, **kwargs):
    w = tf.Variable(__weights_dict[name]['weights'], trainable=is_train, name=name + "_weight")
    if group == 1:
        layer = tf.nn.convolution(input, w, name=name, **kwargs)
    else:
        weight_groups = tf.split(w, num_or_size_splits=group, axis=-1)
        xs = tf.split(input, num_or_size_splits=group, axis=-1)
        convolved = [tf.nn.convolution(x, weight, name=name, **kwargs) for
                     (x, weight) in zip(xs, weight_groups)]
        layer = tf.concat(convolved, axis=-1)

    if 'bias' in __weights_dict[name]:
        b = tf.Variable(__weights_dict[name]['bias'], trainable=is_train, name=name + "_bias")
        layer = layer + b
    return layer
