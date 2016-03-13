#-*-coding:utf8-*-
'''
MNIST:softmax()
'''
import tensorflow as tf
import input_data

#输入数据
mnist = input_data.read_data_sets("qi/data/MNIST_data/",one_hot=True)
x = tf.placeholder("float",[None,784])#None表示第一个维度可以是任意长度
W = tf.Variable(tf.zeros([784,10]))#变量初始化为0
b = tf.Variable(tf.zeros([10]))#b为偏执量


#建立模型
y = tf.nn.softmax(tf.matmul(x,W)+b)
y_ = tf.placeholder("float",[None,10])#计算交叉熵
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)


#初始化变量
init = tf.initialize_all_variables()
sess = tf.Session()#启动模型
sess.run(init)


#训练1000次
for i in range(1000):
    #随机抓取训练100个批处理数据点，然后替换占位符运行train_step
    batch_xs,batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})

#评估模型
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
print sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels})
