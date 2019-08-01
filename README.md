# VOC_val

## 项目说明

使用VOC数据集对FCN模型推断结果进行性能评估，性能评估代码根据官方matlab程序VOCevalseg.m开发成相应python代码

## 环境说明

1. 环境：python2.7
2. 框架：caffe

## 环境搭建 

1. 安装docker
2. 从docker拉取caffe 镜像（https://github.com/BVLC/caffe/tree/master/docker）
    1. CPU版：`sudo docker run -ti bvlc/caffe:cpu caffe --version`
    2. GPU版：`nvidia-docker run -ti bvlc/caffe:gpu caffe --version` 
3. 挂载项目目录到docker `sudo docker run -it -v /home/sun/AIT:/workspace bvlc/caffe:cpu`
4. 在docker中的/workspace目录下载VOC数据集 `https://pjreddie.com/projects/pascal-voc-dataset-mirror/`
5. 在docker中的/workspace目录下载该项目 `git clone https://github.com/SunJackson/VOC_val.git`
6. 在docker中下载FCN模型到/workspace/VOC_val/voc-fcn8s目录下`wget http://dl.caffe.berkeleyvision.org/fcn8s-heavy-pascal.caffemodel`

## 模型测试

1. 进入项目
2. 执行 `python infer.py` 读取/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt文件获取验证集对应的图片名，到/VOCdevkit/VOC2012/JPEGImages/文件夹下获取相应验证集图片（注意该图片为索引图片）。
3. 检查docker中 /workspace/VOCdevkit/results/VOC2012/Segmentation/comp5_val_cls文件夹下模型推断结果图片，结果图与原图合并图片在项目 valresult 目录下

## 模型评估

1. 确认caffe模型推断完相应的图像
2. 执行 `python VOCevalseg.py`

## 模型转换

1. 将 caffe 模型转换成 tensorflow模型 `https://github.com/microsoft/MMdnn`
2. 将tensorflow模型转换成 tflite模型 `python tensorflow2tflite.py`
