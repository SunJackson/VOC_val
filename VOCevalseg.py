# encoding: utf-8
import os
import numpy as np
from numpy.matlib import repmat


from matlab2python import matlab_imread


def VOCevalseg(VOCopts, id):
    VOCopts_seg_imgsetpath = os.path.join(VOCopts['datadir'], VOCopts['dataset'],
                                          'ImageSets/Segmentation/{}.txt'.format(VOCopts['testset']))

    with open(VOCopts_seg_imgsetpath, 'r') as rf:
        gtids = rf.read()
        gtids = gtids.split('\n')

    '''
    number of labels = number of classes plus one for the background
    '''
    num = VOCopts['nclasses']
    confcounts = np.zeros(num)
    count = 0
    for imname in gtids:
        if not imname:
            continue
        gtfile = os.path.join(VOCopts['datadir'], VOCopts['dataset'], 'SegmentationClass/{}.png'.format(imname))
        print '读取标签图片: {}'.format(gtfile)
        gtim, gtimap = matlab_imread(gtfile)
        gtim = np.array(gtim).astype(float)

        resfile = os.path.join(VOCopts['resdir'], 'VOC2012', 'Segmentation/{}_{}_cls/{}.png'.format(id, VOCopts['testset'], imname))
        print '读取推断图片: {}'.format(resfile)

        resim, resmap = matlab_imread(resfile)
        resim = np.array(resim).astype(float)

        maxlabel = resim.max()
        if (maxlabel > VOCopts['nclasses']):
            raise Exception('Results image "{}" has out of range value {} (the value should be <= {})'.format(imname,maxlabel,VOCopts['nclasses']))
        szgtim = gtim.shape

        szresim = resim.shape

        if szgtim != szresim:
            raise Exception(
                'Results image "{}" is the wrong size, was {} x {}, should be {} x {}.'.format(imname,
                                                                                               szresim[0],
                                                                                               szresim[1],
                                                                                               szgtim[0],
                                                                                               szgtim[1]))
        locs = gtim < 255

        sumim = 1 + gtim + resim * num
        hs, edges  = np.histogram(sumim[np.array(locs)], range(1, num * num+2))
        count += locs[locs==True].size
        confcounts = confcounts + hs.reshape(21,21).T

    conf = 100 * confcounts / repmat(1E-20  + np.asmatrix(confcounts).sum(axis=1), 1, confcounts.shape[1])
    rawcounts = confcounts

    accuracies = np.zeros((num, 1))
    print 'Accuracy for each class (intersection/union measure)'
    for j in range(num):
        gtj = np.sum(confcounts[j,:])
        resj = np.sum(confcounts[:, j])
        gtjresj = confcounts[j, j]
        '''
        The accuracy is: true positive / (true positive + false positive + false negative) 
        which is equivalent to the following percentage:
        '''
        accuracies[j] = 100 * gtjresj / (1E-20 + (gtj + resj - gtjresj))
        clname = classes[j]
        print '{} : {}'.format(clname,accuracies[j])
    avacc = np.mean(accuracies)
    print 'Average accuracy: {}'.format(avacc)

if __name__ == '__main__':
    devkitroot = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), 'VOCdevkit')
    classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
               'dog', 'horse', 'motorbike', 'person', 'potted plant',
               'sheep', 'sofa', 'train', 'tv/monitor']
    VOCopts = {
        'datadir': devkitroot,
        'dataset': 'VOC2012',
        'testset': 'val',
        'resdir': os.path.join(devkitroot, 'results'),
        'nclasses': len(classes),
    }
    VOCevalseg(VOCopts, 'comp5')
