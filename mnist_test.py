# ecoding:utf-8
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward
import mnist_backward

TEST_INTERVAL_SECS = 5  # 程序循环间隔时间


def test(mnist):
    with tf.Graph().as_default() as g:  # 复现计算图
        x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
        y_ = tf.placeholder(tf.float32, [None, mnist_forward.OUTPUT_NODE])
        y = mnist_forward.forward(x, None)  # 前向传播 算出y

        ema = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        while True:
            
            with tf.Session() as sess:  # 加载训练好的模型
                ckpt = tf.train.get_checkpoint_state(
                    mnist_backward.MODEL_SAVE_PATH)  # 如果已有 ckpt 模型则恢复
                if ckpt and ckpt.model_checkpoint_path:  # 恢复会话
                    saver.restore(sess, ckpt.model_checkpoint_path)  # 恢复轮数
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    # 计算准确率
                    accuracy_score = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
                    # 打印提示   
                    print("After %s training step(s), test accuracy= %g" % (global_step, accuracy_score))
                    # 如果没有模型
                else:
                    
                    print('No checkpoint file found')  # 模型不存在提示
                    return

            time.sleep(TEST_INTERVAL_SECS)
def main():
    mnist = input_data.read_data_sets("./data/", one_hot=True)
    
    test(mnist)
if __name__ == "__main__":
    main()
