#-*-coding:utf8-*-
'''
CIFAR10_input:主要是对数据的预处理部分
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange
import tensorflow as tf

#原始数据参数
IMAGE_SIZE = 24#原始数据大小为24*24
NUM_CLASSES = 10#图像的种类
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000#用于训练的数据量
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000#用于评估的数据量


def read_cifar10(filename_queue):
    """读取文件，获取数据
    Args:
        filename_queue:文件序列
    returns:
        height:长度（32）
        width:宽度（32）
        depth:色彩的信道（3）
        key:记录数据
        label:图像的种类0..9
        uint8image:[height,width,depth],图像张量tensor
    """
    class CIFAR10Record(object):
        pass#定义一个空类
    result = CIFAR10Record()#整个函数返回一个对象

    #定义图像的维度
    label_bytes = 1#标签的维度为1
    result.height = 32
    result.width = 32
    result.depth = 3
    image_bytes = result.height * result.width * result.depth
    record_bytes = label_bytes + image_bytes#每个图像所包含的数据量
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)#按照固定长度读取数据
    result.key,value = reader.read(filename_queue)
    record_bytes = tf.decode_raw(value,tf.uint8)#将字符串，换为uint8类型
    result.label = tf.cast(tf.slice(record_bytes,[0],[label_bytes]),tf.int32)#将第一个bytes转换为标签
    depth_major = tf.reshape(tf.slice(record_bytes,[label_bytes],[image_bytes]),[result.depth,result.height,result.width])
    #将[depth,height,width] 转换成 [height,width,depth]
    result.uint8image = tf.transpose(depth_major,[1,2,0])#转换维度
    return result

def _generate_image_and_label_batch(iamge,label,min_queue_examples,batch_size,shuffle):
    """生成一系列的图像和标签
    Args:
        image:3-d tensor, [height,width,3]
        label:1-d tensor, int32
        min_queue_examples:挑选batch的最小样本数
        batch_size:每个batch包含的images数目
        shuffle:是否将数据打乱
    returns:
        images:4D tensor [batch_size,height,width,3]
        labels：labels 个数是[batch_size]
    """
    num_preprocess_threads = 16
    if shuffle:
        images,label_batch = tf.train.shuffle_batch(
            [iamge,label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size
            min_after_dequeue=min_queue_examples)#?
    else:
        ipdb.set_trace()  ######### Break Point ###########

        image,label_batch = tf.train.batch(
            [image,label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3*batch_size)#
    tf.image_summary('images',images)
    ipdb.set_trace()  ######### Break Point ###########
return images,tf.reshape(label_batch,[batch_size])

def distorted_inputs(data_dir,batch_size):
    """构建输入数据，模糊处理数据，增加数据样本
    Args:
        data_dir:输入数据的文件地址
        batch_size:每个批次图像的数目
    Returns:
        images:4D tensor [batch_size,IMAGE_SIZE,IMAGE_SIZE,3]
        labels:1D tensor [batch_size]
    """
    filenames = [os.path.join(data_dir,'data_batch_%d.bin'%i) for i in xrange(1,6)]
    for f in filenames:
        if not tf.gfile.Exists(f):#?
            raise ValueError('Failed to find file:' + f)

    #创建一个队列，生成读取的文件
    filename_queue = tf.train.string_input_producer(filenames)#将字符串输出
    read_input = read_cifar10(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image,tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    distorted_image = tf.random_crop(reshaped_image,[height,width,3])
    distorted_image = tf.image.random_flip_left_right(distorted_image)#随机将图像进行左右翻转
    distorted_image = tf.image.random_brightness(distorted_image,max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image，lower=0.2,upper=1.8)
    float_image = tf.image.per_image_whitening(distorted_image)
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_EVAL * min_fraction_of_examples_in_queue)
    print('Filling queue with %d CIFAR images before starting to train,this will take a few minutes.'%min_queue_examples)

    #生成一个批次的图像和标签
    return _generate_image_and_label_batch(float_image,read_input.label,min_queue_examples,batch_size,shuffle=True)

def inputs(eval_data,data_dir,batch_size):
    if not eval_data:
        filenames = [os.path.join(data_dir,'data_batch_%d.bin'%i) for i in xrange(1,6)]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:
        filenames = [os.path.join(data_dir,'test_batch.bin')]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file:' + f)

    #制造一个文件读取序列
    filename_queue = tf.train.string_input_producer(filenames)
    #读取信息
    read_input = read_cifar10(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image,tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    #进行评估
    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,width,height)
    float_image = tf.image.per_image_whitening(resized_image)

    #打乱图像
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch * min_fraction_of_examples_in_queue)

    #生成一系列的图像和标签
    return _generate_image_and_label_batch(float_image,read_input.label,min_queue_examples,batch_size,shuffle=False)
