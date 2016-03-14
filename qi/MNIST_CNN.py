#-*-coding:utf8-*-

'''
CNN：采用卷积神经网络训练MNIST分类
dropout：指在模型训练时随机让网络某些隐含层节点的权重不工作，不工作的那些节点可以
        暂时认为不是网络结构的一部分，但是它的权重得保留下来（只是暂时不更新而已），
        因为下次样本输入时它可能又得工作了（有点抽象，具体实现看后面的实验部分）。
'''

#载入数据
import input_data
mnist = input_data.read_data_sets("qi/data/MNIST_data/",one_hot=True)
import tensorflow as tf

#参数初始化
learning_rate = 0.001#学习率
training_iters = 100000#迭代次数
batch_size = 128#一个训练批次
display_step = 10

#神经网络的参数
n_input = 784#MNIS数据输入（一个图片的大小：28*28）
n_classes = 10 #MNIS数据标签的种类（0-9digits）
dropout = 0.75 #MNIST，每次训练保留神经元的概率

#TF计算图的输入数据
x = tf.placeholder(tf.float32,[None,n_input])#输入的变量
y = tf.placeholder(tf.float32,[None,n_classes])#输入的结果
keep_prob = tf.placeholder(tf.float32)#在计算图中提供dropout占位符


#神经网络的权重和偏执量的配置
weights = {
    'wc1':tf.Variable(tf.random_normal([5,5,1,32])),#5x5的卷积核，1输入，32输出
    'wc2':tf.Variable(tf.random_normal([5,5,32,64])),#5x5的卷积核，32个输入，64个输出
    'wd1':tf.Variable(tf.random_normal([7*7*64,1024])),#7*7*64输入，1024输出
    'out':tf.Variable(tf.random_normal([1024,n_classes])),#1024输入，10输出
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

#创建一个模型
#卷积层
def conv2d(img,w,b):
    '''
    conv2d:第一个卷积层,输入图像序列，设置一个步长为1
    img:输入的图像序列
    w:神经元的权重
    b:偏移量
    '''
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(img,w,strides=[1,1,1,1],padding='SAME'),b))

#池化层
def max_pool(img,k):
    '''
    pool层：
    img:输入的图像序列
    k:池化的方格
    '''
    return tf.nn.max_pool(img,ksize=[1,k,k,1],strides=[1,k,k,1],padding='SAME')

#建立网络
def conv_net(_X,_weights,_biases,_dropout):
    '''
    _X:输入的像素序列
    _weights:权重
    _biases:偏移量
    _dropout:保留量
    '''
    #重新切分输入图片的像素，形成可以处理的数据类型
    _X = tf.reshape(_X,shape=[-1,28,28,1])
    #加入第一层卷积
    conv1 = conv2d(_X,_weights['wc1'],_biases['bc1'])
    conv1 = max_pool(conv1,k=2) #池化层
    conv1 = tf.nn.dropout(conv1,_dropout)#使用dropout

    #加入第二层卷积
    conv2 = conv2d(conv1,_weights['wc2'],_biases['bc2'])
    conv2 = max_pool(conv2,k=2)
    conv2 = tf.nn.dropout(conv2,_dropout)

    #加入一个全连接层
    dense1 = tf.reshape(conv2,[-1,_weights['wd1'].get_shape().as_list()[0]])#将输出重新整合适应全连接层的输入
    dense1 = tf.nn.relu(tf.add(tf.matmul(dense1,_weights['wd1']),_biases['bd1']))#输出激活函数
    dense1 = tf.nn.dropout(dense1,_dropout)#使用dropout

    #输出层，分类预测
    out = tf.add(tf.matmul(dense1,_weights['out']),_biases['out'])
    return out

#生成模型
pred = conv_net(x,weights,biases,keep_prob)

#定义损失函数和优化函数
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred,y))#输入预测和真实值
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)#训练模型，梯度下降法

#评估模型
correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))#?
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))#?

# 初始化所有变量
init = tf.initialize_all_variables()

#运算图
def run():
    #载入模型
    with tf.Session() as sess:
        sess.run(init)#初始化
        step = 1
        #开始训练，知道最大的迭代次数
        while step * batch_size < training_iters:
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            #使用切片数据进行训练
            sess.run(optimizer,feed_dict={x:batch_xs,y:batch_ys,keep_prob:dropout})
            if step % display_step == 0:#每隔10次输出数据
                #评估模型
                acc = sess.run(accuracy,feed_dict={x:batch_xs,y:batch_ys,keep_prob:1.})
                #评估损失
                loss = sess.run(cost,feed_dict={x:batch_xs,y:batch_ys,keep_prob:1.})
                print "Iter" + str(step*batch_size) +",Minibatch Loss"+"{:.6f}".format(loss)+",Training accuracy="+"{:.5f}".format(acc)
            step += 1
        print "Optimization Finished!"
        #测试
        print "Testing Accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images[:256], y: mnist.test.labels[:256], keep_prob: 1.})



if __name__ == '__main__':
    run()
