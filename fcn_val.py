# encoding: utf-8
import numpy as np
from PIL import Image
import os
import datetime

import caffe
import vis

# load net
net = caffe.Net('voc-fcn8s/deploy.prototxt', 'voc-fcn8s/fcn8s-heavy-pascal.caffemodel', caffe.TEST)
devkitroot = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), 'VOCdevkit')


def predict_image(id):
    # load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe

    VOCopts_seg_imgsetpath = os.path.join(VOCopts['datadir'], VOCopts['dataset'],
                                          'ImageSets/Segmentation/{}.txt'.format(VOCopts['testset']))

    with open(VOCopts_seg_imgsetpath, 'r') as rf:
        gtids = rf.read()
        gtids = gtids.split('\n')
    num = 1
    total_time = 0
    for imname in gtids:
        start_time = datetime.datetime.now()
        if not imname:
            continue
        imgpath = os.path.join(VOCopts['datadir'], VOCopts['dataset'], 'JPEGImages/{}.jpg'.format(imname))
        print '读取图片: {}'.format(imgpath)

        im = Image.open(imgpath)
        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:, :, ::-1]
        in_ -= np.array((104.00698793, 116.66876762, 122.67891434))
        in_ = in_.transpose((2, 0, 1))

        # shape for input (data blob is N x C x H x W), set data
        net.blobs['data'].reshape(1, *in_.shape)
        net.blobs['data'].data[...] = in_
        # run net and take argmax for prediction
        net.forward()
        out = net.blobs['score'].data[0].argmax(axis=0)

        out_im = Image.fromarray(out.astype('uint8'))

        end_time = datetime.datetime.now()
        process_time = end_time - start_time
        total_time += process_time.total_seconds()
        print '处理图片第{}张图片{} 耗时{}'.format(num, imname, process_time)
        print '总耗时{}, 平均耗时{}'.format(total_time , total_time/num)
        num += 1
        save_path = os.path.join(VOCopts['resdir'],
                                 'Segmentation/{}_{}_cls/'.format(id, VOCopts['testset']))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        resfile = os.path.join(VOCopts['resdir'],
                               'Segmentation/{}_{}_cls/{}.png'.format(id, VOCopts['testset'], imname))
        out_im.putpalette(np.array(colormap).reshape(-1))
        out_im.save(resfile)
        # visualize segmentation in PASCAL VOC colors
        voc_palette = vis.make_palette(21)
        out_palette_im = Image.fromarray(vis.color_seg(out, voc_palette))



        out_palette_im.save('./valresult/out_palette_{}.png'.format(imname))

        masked_im = Image.fromarray(vis.vis_seg(im, out, voc_palette))
        masked_im.save('./valresult/out_palette_{}_visualization.png'.format(imname))
        print '存储结果图片: {}'.format(resfile)



if __name__ == '__main__':
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

    predict_image('comp5')
