# encoding: utf-8
'''
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/convert/python_api.md
'''
import tensorflow as tf

pb_file_path = '../caffe2tensorflow/caffe_fcn8s/'


with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, ['train'], pb_file_path)
    sess.run(tf.global_variables_initializer())

    input_x = sess.graph.get_tensor_by_name('input:0')
    output_y = sess.graph.get_tensor_by_name('crop_to_bounding_box_2/Slice:0')

    converter = tf.lite.TFLiteConverter.from_session(sess, [input_x], [output_y])
    tflite_model = converter.convert()
    open("./model/converted_model.tflite", "wb").write(tflite_model)