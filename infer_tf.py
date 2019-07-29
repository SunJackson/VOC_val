# encoding: utf-8
'''
tensorflow模型推断
'''
import tensorflow as tf
import numpy as np
from PIL import Image
import datetime
import os

import vis


def check_dir(check_path):
    if not os.path.exists(check_path):
        os.makedirs(check_path)


devkitroot = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), 'VOCdevkit')
classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
           'dog', 'horse', 'motorbike', 'person', 'potted plant',
           'sheep', 'sofa', 'train', 'tv/monitor']
colormap = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
            [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
            [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
            [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
            [0, 192, 0], [128, 192, 0], [0, 64, 128]]
VOCopts = {
    'datadir': devkitroot,
    'dataset': 'VOC2012',
    'testset': 'val',
    'resdir': os.path.join(devkitroot, 'results', 'VOC2012'),
    'nclasses': len(classes),
}
id = 'comp3'
pb_file_path = '../caffe2tensorflow/caffe_fcn8s/'



with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, ['train'], pb_file_path)
    sess.run(tf.global_variables_initializer())

    input_x = sess.graph.get_tensor_by_name('input:0')
    output_y = sess.graph.get_tensor_by_name('crop_to_bounding_box_2/Slice:0')

    VOCopts_seg_imgsetpath = os.path.join(VOCopts['datadir'], VOCopts['dataset'],
                                          'ImageSets/Segmentation/{}.txt'.format(VOCopts['testset']))

    converter = tf.lite.TFLiteConverter.from_session(sess, [input_x], [output_y])
    tflite_model = converter.convert()
    open("converted_model.tflite", "wb").write(tflite_model)

    with open(VOCopts_seg_imgsetpath, 'r') as rf:
        gtids = [i.replace('\n', '') for i in rf.readlines()]
    num = 1
    total_time = 0
    for imname in gtids:
        start_time = datetime.datetime.now()
        if not imname:
            continue
        imgpath = os.path.join(VOCopts['datadir'], VOCopts['dataset'], 'JPEGImages/{}.jpg'.format(imname))
        print('读取图片: {}'.format(imgpath))

        im = Image.open(imgpath)

        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:, :, ::-1]
        in_ -= np.array((104.00698793, 116.66876762, 122.67891434))
        im_shape = in_.shape
        new_im = np.pad(in_,
                        ((0, 500 - int(im_shape[0])), (0, 500 - int(im_shape[1])), (0, 3 - int(im_shape[2]))),
                        'constant', constant_values=(0, 0))  # constant_values表示填充值，且(before，after)的填充值等于（0,0）

        out = sess.run(output_y, feed_dict={input_x: np.expand_dims(new_im, axis=0)})
        im_output = np.array( tf.argmax(out[0], 2).eval())
        result = im_output[:im_shape[0], :im_shape[1]]
        out_im = Image.fromarray(result.astype('uint8'))

        end_time = datetime.datetime.now()
        process_time = end_time - start_time
        total_time += process_time.total_seconds()
        print '处理图片第{}张图片{} 耗时{}'.format(num, imname, process_time)
        print '总耗时{}, 平均耗时{}'.format(total_time, total_time / num)
        num += 1
        save_path = os.path.join(VOCopts['resdir'],
                                 'Segmentation/{}_{}_cls/'.format(id, VOCopts['testset']))
        check_dir(save_path)
        resfile = os.path.join(VOCopts['resdir'],
                               'Segmentation/{}_{}_cls/{}.png'.format(id, VOCopts['testset'], imname))

        out_im.putpalette(np.array(colormap).reshape(-1))
        out_im.save(resfile)
        # visualize segmentation in PASCAL VOC colors
        voc_palette = vis.make_palette(21)
        out_palette_im = Image.fromarray(vis.color_seg(result, voc_palette))

        check_dir('./valresult/')
        out_palette_im.save('./valresult/out_{}.png'.format(imname))

        masked_im = Image.fromarray(vis.vis_seg(im, result, voc_palette))
        masked_im.save('./valresult/out_{}_visualization.png'.format(imname))
        print '存储结果图片: {}'.format(resfile)
