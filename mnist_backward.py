#coding:utf-8
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward
import os
import matplotlib.pyplot as plt

BATCH_SIZE = 200 #每轮喂入神经网络的图片
LEARNING_RATE_BASE = 0.1  #最开始的学习率
LEARNING_RATE_DECAY = 0.99 #学习率衰减率
REGULARIZER = 0.0001 #正则化系数
STEPS = 20000 #训练轮数
MOVING_AVERAGE_DECAY = 0.99 #滑动平均 衰减率
MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "mnist_model"
xxx = []
yyy = []
def backward(mnist):
    x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
    y_ = tf.placeholder(tf.float32, [None, mnist_forward.OUTPUT_NODE])
    y = mnist_forward.forward(x, REGULARIZER)
    global_step = tf.Variable(0, trainable = False) #轮数计数器

    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y, labels = tf.argmax(y_, 1))
    cem = tf.reduce_mean(ce)
    loss = cem + tf.add_n(tf.get_collection('losses'))  #包含正则化的损失函数loss

    learning_rate = tf.train.exponential_decay(  #指数衰减学习率 
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True
    )
    
    #训练过程
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name = 'train')
    
    saver = tf.train.Saver()

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        #恢复所有w b 模型继续从断点处训练
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path :
            saver.restore(sess, ckpt.model_checkpoint_path)


        for i in range(STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x:xs, y_:ys})
            if i %1000 == 0:
                xxx.append(step)
                yyy.append(loss_value)
                print("After %d training, loss is %g" % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
    plt.xlabel('Number of iterations')         
    plt.ylabel('loss')
    plt.plot(xxx,yyy)
    plt.show()
def main():
    mnist = input_data.read_data_sets("./data/", one_hot = True)
    backward(mnist)
if __name__ == "__main__":
    main()
