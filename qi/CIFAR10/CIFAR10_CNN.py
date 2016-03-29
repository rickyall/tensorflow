s#-*-coding:utf8-*-
'''
CIFAR10_CNN:采用cnn对图像进行分类,该部分为模型层
decay:常用的惩罚项是所有权重的平方乘以一个衰减常量之和.
LRN:全称为Local Response Normalization，即局部响应归一化层，局部响应归一化层完成一种“临近抑制”
    操作，对局部输入区域进行归一化。
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os3
import re
import sys
import tarfile

from six.moves import urllib
import tensorflow as tf

import CIFAR10_input#载入输入的模块


#基本的参数
batch_size = 128#一次迭代输入的训练图像的数据
data_dir = 'data/CIFAR10_data'#保存的数据目录
DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'#数据的下载地址
TOWER_NAME = 'tower'

#关于数据的参数
IMAGE_SIZE = CIFAR10_input.IMAGE_SIZE#输入的图片大小
NUM_CLASSES = CIFAR10_input.NUM_CLASSES#分类的数目
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = CIFAR10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN#总的训练数据
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = CIFAR10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL#总的测试数据

#训练的参数
MOVING_AVERAGE_DECAY = 0.9999 #？
NUM_EPOCHS_PER_DECAY = 350.0 #?
LEARNING_RATE_DECAY_FACTOR = 0.1#？
INITIAL_LEARNING_RATE = 0.1#初始的学习率


def _activation_summary(x):
    """主要用于总结每轮迭代的状态
    Args:
        x:Tensor
    Returns:
        nothing
    """
    tensor_name = re.sub('%s_[0-9]*/'%TOWER_NAME,'',x.op.name)
    tf.histogram_summary(tensor_name+'/activations',x)
    tf.scalar_summary(tensor_name+'/sparsity',tf.nn.zero_fraction(x))

def _variable_on_cpu(name,shape,initializer):
    """
    创建一个变量
    Args:
        name:name of the variable
        shape:list of ints
        initializer:初始化变量
    Return:
        变量张量
    """
    with tf.device('/cpu:0'):
        var = tf.get_variable(name,shape,initializer=initializer)
    return var

def _variable_with_weight_decay(name,shape,stddev,wd):
    """权重衰减的变量
    args:
        name:变量名字
        shape:大小
        stddev:搞死分布
        wd:加入权重衰减系数
    returns:
        变量名字
    """
    var = _variable_on_cpu(name,shape,tf.truncated_normal_initializer(stddev=stddev))
    if wd:
        weight_decay = tf.mul(tf.nn.l2_loss(var),wd,name='weight_loss')
        tf.add_to_collection('losses',weight_decay)#?
    return var

def distorted_inputs():
    """训练数据
    """
    if not data_dir:
        raise ValueError('Please supply a data_dir')
        data_dir = os.path.join(data_dir,'cifar-10-batches-bin')
        return CIFAR10_input.distorted_inputs(data_dir=data_dir,batch_size=batch_size)

def inputs(eval_data):
    """测试数据
    """
    if not data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(data_dir,'cifar-10-batches-bin')
    return CIFAR10_input.inputs(eval_data=eval_data,data_dir=data_dir,batch_size=batch_size)

#建立模型
def inference(images):
    """建立Model
    """
    #conv1第一层卷积
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay('weights',shape=[5,5,3,64],stddev=1e-4,wd=0.0)
        conv = tf.nn.conv2d(images,kernel,[1,1,1,1],padding='SAME')
        biases = _variable_on_cpu('biases',[64],tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv,biases)
        conv1 = tf.nn.relu(bias,name=scope.name)
        _activation_summary(conv1)

    #pool1池化层
    pool1 = tf.nn.max_pool(conv1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME',name='pool1')

    #norm1
    norm1 = tf.nn.lrn(pool1,4,bias=1.0,alpha=0.001/9.0,beta=0.75,name='norm1')

    #conv2
    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights',shape=[5,5,64,64],stddev=1e-4,wd=0.0)
        conv = tf.nn.conv2d(norm1,kernel,[1,1,1,1],padding='SAME')
        biases = _variable_on_cpu('biases',[64],tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv,biases)
        conv2 = tf.nn.relu(bias,name = scope.name)
        _activation_summary(conv2)

    #norm2
    norm2 = tf.nn.lrn(conv2,4,bias=1.0,alpha=0.001/9.0,beta=0.75,name='norm2')

    #pool2
    pool2 = tf.nn.max_pool(norm2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME',name='pool2')

    #local3
    #基于修正线性激活的全连接层
    with tf.variable_scope('local3') as scope:
        dim = 1
        for d in pool2.get_shape()[1:].as_list():
            dim *= d
        reshape = tf.reshape(pool2,[batch_size,dim])
        weights = _variable_with_weight_decay('weights',shape=[dim,384],stddev=0.04,wd=0.004)
        biases = _variable_on_cpu('biases',[384],tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape,weights)+biases,name=scope.name)
        _activation_summary(local3)

    #local4
    with tf.variable_scope('local4') as scope:
        weights = _variable_with_weight_decay('weights',shape=[384,192],stddev=0.04,wd=0.004)
        biases = _variable_on_cpu('biases',[192],tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3,weights)+biases,name=scope.name)
        _activation_summary(local4)

    #softmax层
    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay('weights',[192,NUM_CLASSES],stddev=1/192.0,wd=0.0)
        biases = _variable_on_cpu('biases',[NUM_CLASSES],tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local3,weights),biases,name=scope.name)
        _activation_summary(softmax_linear)

    return softmax_linear


def loss(logits,labels):
    """
    计算村是函数
    Args:
        logits:评定模型
        labels:数据的真实值
    return:
        loss tensor
    """
    labels = tf.cast(labels,tf.int64)#转换数据类型
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits,labels,name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy,name='cross_entropy')
    tf.add_to_collection('losses',cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'),name='total_loss')

def _add_loss_summaries(total_loss):
    """将总的损失纳入到summary中Generates moving average
    Args:
        total_loss:loss()
    returns:
        loss_averages_op:获得节点
    """
    loss_averages = tf.train.ExponentialMovingAverage(0.9,name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses+[total_loss])
    for l in losses + [total_loss]:
        tf.scalar_summary(l.op.name+'(raw)',l)
        tf.scalar_summary(l.op.name,loss_averages.average(l))
    return loss_averages_op

def train(total_loss,global_step):
    """训练模型
    """
