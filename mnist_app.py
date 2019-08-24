import tensorflow as tf 
import numpy as np 
import mnist_forward
import mnist_backward
import cv2

def restore_model(testPicArr):
    with tf.Graph().as_default() as tg:
        x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
        y = mnist_forward.forward(x, None)
        preValue = tf.argmax(y,1)

        variable_averages = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

                preValue = sess.run(preValue, feed_dict={x:testPicArr})
                return preValue
            else:
                print("Not found checkpoint")
                return -1

def pre_pic(picName): #处理图片使之成为28*28的图像 像素的范围在0-1之间
    img = cv2.imread(picName,0)
    #reImg = img.reshape((28,28), img.ANTIALIAS)
    img = cv2.resize(img, (28,28))
    #im_arr = np.array(reImg.convert('L')) #灰度图
    threshold = 50
    #二值化
    for i in range(28):
        for j in range(28):
            img[i][j] = 255-img[i][j] #黑底白字
            if(img[i][j] < threshold):
                img[i][j] = 0
            else:
                img[i][j] = 255
    nm_arr = img.reshape([1, 784])
    nm_arr = nm_arr.astype(np.float32)  #
    img_res = np.multiply(nm_arr, 1.0/255.0)
    return img_res

def application():
    testNum = int(input("input test image number :"))
    for i in range(testNum):
        testPic = input("the path of  test image")
        testPicArr = pre_pic(testPic) 
        preValue = restore_model(testPicArr)
        print("The prediction number is: ", preValue)

def main():
    application()
if __name__ == "__main__":
    main()
