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
def wrap_digit(rect):
    x, y, w, h = rect
    padding = 5
    hcenter = x + w / 2
    vcenter = y + h / 2
    if (h > w):
        w = h
        x = hcenter - (w / 2)
    else:
        h = w
        y = vcenter - (h / 2)
    return (int(x - padding), int(y - padding), int(w + padding), int(h + padding))
def application():
    #预处理图片
    path = input("input the path of  test image in this dir: ")
    img = cv2.imread(path)
    ref = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ref = cv2.GaussianBlur(ref, (5, 5), 0)
    ref = cv2.threshold(ref,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]  #二值化
    ref = cv2.dilate(ref, np.ones((3, 3), np.uint8)) 
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, ))  # 定义结构元素
    #ref = cv2.morphologyEx(ref, cv2.MORPH_OPEN, kernel)
    image, cntrs, hier = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
    #testNum = int(input("input test image number :"))
    #rectangles = []
    
    for c in cntrs:
        r = x, y, w, h = cv2.boundingRect(c)
        x, y, w, h = wrap_digit(r)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi = ref[y-2:y+h+2, x:x+w]
        #cv2.imshow("roi", roi)
        roi = cv2.resize(roi, (28,28))
        top_size,bottom_size,left_size,right_size=(5,5,5,5)
        #扩展边界
        roi=cv2.copyMakeBorder(roi,top_size,bottom_size,left_size,right_size,borderType=cv2.BORDER_CONSTANT,value=0)
        roi = cv2.resize(roi, (28,28))
        #cv2.imshow("ref", roi)
        #使数据符合神经网络的数据格式
        nm_arr = roi.reshape([1, 784])
        nm_arr = nm_arr.astype(np.float32)  #
        img_res = np.multiply(nm_arr, 1.0/255.0)
        #预测
        preValue = restore_model(img_res)[0]
        cv2.putText(img, "%d" % preValue, (x, y - 1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
    cv2.imshow("res", img)
    #cv2.imshow("ref", ref)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def main():
    application()
if __name__ == "__main__":
    main()
