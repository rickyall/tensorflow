#-*-coding:utf8-*-
'''
tensorFlow：
1）使用图（graph）来表示计算任务
2）在被称之为会话（Session）的上下文（context）中执行图
3）使用tensor表示数据，数据流入图中
4）通过变量（Variable）维护状态
5）使用feed和fetch为任意操作输入和输出数据
构建图：
1）第一步：创建源op(source op)，不需要输入，如常量，输出传递给其他的op
2）有一个默认的graph
'''
import tensorflow as tf
#创建一个常量op,产生一个1*2矩阵，这个Op被作为一个节点
#加到默认图中
# matrix1 = tf.constant([[3.,3.]])
# matrix2 = tf.constant([[2.],[2.]])
# product = tf.matmul(matrix1,matrix2)


#变量 Variables
state = tf.Variable(0,name='counter')#定义一个变量，初始化为标量0
one = tf.constant(1)
new_value = tf.add(state,one)
update = tf.assign(state,new_value)#更新变量值

#初始化的op
init_op = tf.initialize_all_variables()

#使用with代码隐式的调用close
with tf.Session() as sess:
    sess.run(init_op)
    #打印state的初始值
    print sess.run(state)
    for _ in range(3):
        sess.run(update)
        print sess.run(state)
