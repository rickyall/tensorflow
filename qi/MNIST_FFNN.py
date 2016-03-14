#-*-coding:utf8-*-

'''
feed-forward neural network：采用前馈神经网络训练MNIST分类
cross_entropy：交叉熵？
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
import math
import time
from six.moves import xrange#?
import tensorflow as tf

#载入数据
import input_data
mnist = input_data.read_data_sets("qi/data/MNIST_data/",False)

#原始数据参数
NUM_CLASSES = 10#mnist的类别，总共包括10类
IMAGE_SIZE = 28#图像的大小
IMAGE_PIXELS = IMAGE_SIZE *IMAGE_SIZE#图像的总的像素数

#神经网络参数初始化
learning_rate = 0.01#学习率
max_steps = 2000#最大的迭代次数
hidden1 = 128#第一个隐藏层的数目
hidden2 = 32#第二个隐藏层的数目
batch_size = 100#每次迭代抽取的样本数目
fake_data = False#使用伪造数据


#在TF图中加入输入数据占位符
def placeholder_inputs(batch_size):
    """
    args:
        batch_size:每次迭代输入的样本数目
    returns:
        images_placeholder:图像占位符
        labels_placeholder:标签占位符
    """
    images_placeholder = tf.placeholder(tf.float32,shape=(batch_size,IMAGE_PIXELS))
    labels_placeholder = tf.placeholder(tf.int32,shape=(batch_size))
    return images_placeholder,labels_placeholder

#定义输入数据集
def fill_feed_dict(data_set,images_pl,labels_pl):
    #每次迭代抽取数据
    images_feed,labels_feed = data_set.next_batch(batch_size,fake_data)
    feed_dict = {
        images_pl: images_feed,
        labels_pl: labels_feed
    }
    return feed_dict


#创建一个模型
#定义模型图
def inference(images,hidden1_units,hidden2_units):
    """建立前馈神经网络模型
    Args:
        images:输入图像数据
        hidden1_units:第一个隐藏层的神经元数目
        hidden2_units:第二个隐藏层 的神经元数目
    returns:
        softmax_linear:输出张量为计算后的结果
    """
    #隐藏层1
    with tf.name_scope('hidden1'):
        weights = tf.Variable(tf.truncated_normal([IMAGE_PIXELS,hidden1_units],stddev=1.0/math.sqrt(float(IMAGE_PIXELS))),name='weights')#?
        biases = tf.Variable(tf.zeros([hidden1_units]),name='biases')
        hidden1 = tf.nn.relu(tf.matmul(images,weights)+biases)

    #隐藏层2
    with tf.name_scope('hidden2'):
        weights = tf.Variable(tf.truncated_normal([hidden1_units,hidden2_units],stddev=1.0/math.sqrt(float(hidden1_units))),name='weights')
        biases = tf.Variable(tf.zeros([hidden2_units]),name='biases')
        hidden2 = tf.nn.relu(tf.matmul(hidden1,weights)+biases)
    #线性输出层
    with tf.name_scope('softmax_linear'):
        weights = tf.Variable(tf.truncated_normal([hidden2_units,NUM_CLASSES]),name='biases')
        biases = tf.Variable(tf.zeros([NUM_CLASSES]),name='biases')
        logits = tf.matmul(hidden2,weights) + biases
    return logits

#定义损失函数
def loss(logits,labels):
    """衡量真实值和预测值之间的差距
    Args:
        logits:预测值，Logits tensor
        labels:真实值，labels tensor
    returns:
        loss:损失函数的值
    """
    labels = tf.to_int64(labels)#将标签值编码为一个含有1-hot values的Tensor.
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits,labels,name='xentropy')#计算交叉熵
    loss = tf.reduce_mean(cross_entropy,name = 'xentropy_mean')#交叉熵的平均值
    return loss

#定义训练函数
def training(loss,learning_rate):
    """建立一个训练的节点
    Args:
        loss:损失张量，来自loss()
        learning_rate:学习速率
    Returns:
        train_op:返回训练的节点
    """
    # 添加一个监听损失函数
    # 首先,该函数从 loss() 函数中获取损失Tensor,将其交给 tf.scalar_summary ,
    # 后者在与 SummaryWriter (见下 文)配合使用时,可以向事件文件(events file)中生成汇总值(summary values)。
    # 在本篇教程中,每次写入汇总值时,它都会释放损失Tensor的当前值(snapshot value)。
    tf.scalar_summary(loss.op.name,loss)
    #建立一个梯度下降的optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    #创建一个变量跟踪迭代的步数global step
    global_step = tf.Variable(0,name='global_step',trainable=False)
    #创建一个训练的节点
    train_op = optimizer.minimize(loss,global_step=global_step)
    return train_op


#定义评估函数
def evaluation(logits,labels):
    """评估训练的准确率
    Args:
        logits:预测的结果 logits tensor
        labels:真实的结果 labels tensor
    returns:
        正确的数目
    """
    #对分类的模型，我们可以使用 in_top_k Op
    #返回某一类的数据判断结果，比如正确的数据
    correct = tf.nn.in_top_k(logits,labels,1)
    return tf.reduce_sum(tf.cast(correct,tf.int32))#范围正确的数目,tf.cast将数据转换成指定类型

#定义评估动作
def do_eval(sess,eval_correct,images_placeholder,labels_placeholder,data_set):
    """在一次迭代过程中基于全部的数据进行预测，判断准确率
    Args:
        sess:模型已经被训练的过程
        eval_correct:预测正确的数据
        images_placeholder:图像占位符
        labels_placeholder:标签占位符
        data_set:数据集
    """
    true_count = 0
    steps_per_epoch = data_set.num_examples // batch_size #每次大的迭代需要的步数
    num_examples = steps_per_epoch * batch_size#总的参与训练的样本数量
    for step in xrange(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set,images_placeholder,labels_placeholder)
        true_count += sess.run(eval_correct,feed_dict=feed_dict)#正确的样本增加
    precision = true_count / num_examples #准确率
    print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %(num_examples, true_count, precision))




#定义训练的过程
def run_training():
    """
    训练过程
    """
    # 告诉TF，模型采用默认Graph
    with tf.Graph().as_default():
        #创建输入的图像和标签的占位符
        images_placeholder,labels_placeholder = placeholder_inputs(batch_size)
        #建立一个模型图，预测结果
        logits = inference(images_placeholder,hidden1,hidden2)
        #添加一个损失函数ops
        loss_op = loss(logits,labels_placeholder)
        #添加一个训练ops
        train_op = training(loss_op,learning_rate)
        #添加评估OP
        eval_correct = evaluation(logits,labels_placeholder)
        #建立一个会话
        sess = tf.Session()
        #初始化所有变量
        init = tf.initialize_all_variables()
        sess.run(init)

        #开始训练
        for step in xrange(max_steps):
            start_time = time.time()
            #建立feed dictionary
            feed_dict = fill_feed_dict(mnist.train,images_placeholder,labels_placeholder)
            _,loss_value = sess.run([train_op,loss_op],feed_dict=feed_dict)#训练并获取训练结果
            duration = time.time() - start_time #计算消耗的时间

            #输出每步训练的结果
            if step % 100 == 0:
                print('Step %d:loss = %.2f(%.3f sec)'%(step,loss_value,duration))#打印损失函数值和消耗的时间
                #评估模型训练集
                print('Training Data Eval:')
                do_eval(sess,eval_correct,images_placeholder,labels_placeholder,mnist.train)
                #评估模型模型测试集
                print('Test Data Eval:')
                do_eval(sess,eval_correct,images_placeholder,labels_placeholder,mnist.test)

if __name__ == '__main__':
    run_training()
