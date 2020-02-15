#coding:utf-8
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import forward
import mnist_backward

#设置延时
TEST_INTERVAL_SECS = 5

def test(mnist):
	#设置默认图作为计算的模型、
	with tf.Graph().as_default() as g:
		x =  tf.placeholder(tf.float32,shape=[None,forward.INPUT_NODE])
		y_ = tf.placeholder(tf.float32,shape=[None,forward.OUTPUT_NODE])
		#获取输出结果，且不使用正则化
		y = forward.forward(x,None)

		#设置滑动平均
		ema = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
		#加载滑动平均
		ema_restore = ema.variables_to_restore()
		saver = tf.train.Saver(ema_restore)

		#设置正确的标签
		correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
		#计算准确率
		accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
	
		while True:
			with tf.Session() as sess:
				#判断checkpoint是否存在
				ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
				if ckpt and ckpt.model_checkpoint_path:
					saver.restore(sess,ckpt.model_checkpoint_path)
					#提取当前保存模型的轮数
					global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
					#计算准确率
					accuracy_score = sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels})
					#打印准确率
					print "After %s training step(s),test accuracy = %g"%(global_step,accuracy_score)
				else:
					print "No checkpoint file found"
					return
			#延时
			time.sleep(TEST_INTERVAL_SECS)

def main():
	mnist = input_data.read_data_sets("./data/",one_hot=True)
	test(mnist)

if __name__ == '__main__':
	main()


















