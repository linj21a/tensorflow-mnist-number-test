#coding:utf-8
import tensorflow as tf
from PIL import Image
import numpy as np
import forward
import mnist_backward
import mnist_test
from cv2 import *
#这是修改图片的格式，使其能够直接输入神经网络，执行预判

#定义一个函数，用于处理输入的图片，转化为值是在[0,1]的1×784的一维数组
def pre_picture(Picture_Path):
	#打开图片
	
	img = imread(Picture_Path)
	img = resize(img,(28,28),INTER_AREA)
  
    	#imshow("img",img);#
    	#cvtColor the picture
    	gray = cvtColor(img,COLOR_BGR2GRAY)
    	#imshow("gray",gray);
    	#bitwrise_not
    	gray_not = bitwise_not(gray)
    	#imshow("gray_not",gray_not)#
    	#erode the picture
	kernel = np.array([[3,3,3],
			[3,3,3],[3,3,3]])
			
    	dst = dilate(gray_not,kernel);
    	#imshow("dilate",dst);
    	#threshold picture
    	_,dst = threshold(dst,120,255,THRESH_BINARY)
    	#imshow("THRESHOLD",dst)#
	img_array = np.array(dst)
    	
	pre_array = img_array.reshape((1,784))
	#转化为浮点型
	pre_array = pre_array.astype(np.float32)
	#按比例缩放值到[0,1]
	img_ready = np.multiply(pre_array,1.0/255.0)
	
	return img_ready
			
#定义函数，批量加载图片并喂入神经网络,并返回预测的值
def restore_model(testPicArray):
	#创建默认的计算图，在该图中执行预测步骤
	with tf.Graph().as_default() as g:
		#输入的x占位
		x = tf.placeholder(tf.float32,shape=[None,forward.INPUT_NODE])
		#获取输出y
		y = forward.forward(x,None)
		#返回最大值为1的索引，并赋值给preValue
		preValue = tf.argmax(y,1)

		#实现滑动平均，参数MOVING_AVERAGE_DECAY用于控制模型更新的速度
		#训练过程中会对每个变量维护一个影子，这个影子的初始值就是相应变量的初始值
		variable_averages = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
		#加载滑动平均
		variables_to_restore = variable_averages.variables_to_restore()
		#实例化saver对象
		saver = tf.train.Saver(variables_to_restore)
		
			
		with tf.Session() as sess:
			#加载模型文件
			ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
			#判断
			if ckpt and ckpt.model_checkpoint_path:
				#加载训练的模型
				saver.restore(sess,ckpt.model_checkpoint_path)
				#计算预测值
				preValue = sess.run(preValue,feed_dict={x:testPicArray})

				return preValue
			else:
				print "No checkpoint file found"
				return -1

#定义一个应用的函数
def application():
	test_Number = int(input("input the number of the test pictures:"))
	#循环输入图片，输入一张，预测一张
	for i in range(test_Number):
		Path =raw_input("input the image's path:")
		#图片预处理
		
		testPicArray = pre_picture(Path)

		preValue = restore_model(testPicArray)
		print "the preValue number is ",preValue
	#waitKey(0);
		

def main():
	application()
	

if __name__ == '__main__':
	main()











