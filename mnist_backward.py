#coding:utf-8
import tensorflow as tf 
import forward
import os
from tensorflow.examples.tutorials.mnist import input_data

#定义喂入数据的size
BATCH_SIZE = 200
#定义指数衰减学习率与滑动平均
LEARNING_RATE_DECAY = 0.99
LEARNING_RATE_BASE = 0.1
REGULARIZER = 0.0001
#轮数
STEPS = 50000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "/home/linhu/tf_file/fifth/model_data"
MODEL_NAME = "mnist_model"


def backward(mnist):
	x = tf.placeholder(tf.float32,shape=[None,forward.INPUT_NODE])
	y_ = tf.placeholder(tf.float32,shape=[None,forward.OUTPUT_NODE])
	y = forward.forward(x,REGULARIZER)
	global_step = tf.Variable(0,trainable=False)
	#使用交叉熵计算损失。配合softmax
	ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
	cem =  tf.reduce_mean(ce)
	loss = cem + tf.add_n(tf.get_collection('losses'))

	#设置指数衰减学习率，随机梯度下降调整学习率（其他）
	learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,mnist.train.num_examples/BATCH_SIZE,LEARNING_RATE_DECAY,staircase=True)
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
	
	#train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step)
	#train_step = tf.train.MomentumOptimizer(learning_rate,Momentum,name='Momentum').minimize(loss,global_step=global_step)

	#设置滑动平均
	ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
	ema_op = ema.apply(tf.trainable_variables())
	with tf.control_dependencies([train_step,ema_op]):
		train_op = tf.no_op(name='train')

	#保存模型先实例化saver对象
	saver = tf.train.Saver()
	input_ = input("请输入一个数选择是否实现断点续训，0表示是：")
	
	with tf.Session() as sess:
		init_op = tf.global_variables_initializer()
		sess.run(init_op)
		#加入断点续训功能
		if input_==0:
			ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
			if ckpt and ckpt.model_checkpoint_path:
				saver.restore(sess,ckpt.model_checkpoint_path)
		
		for i in range(STEPS):
		#设置输入的数据，根据BATCH_SIZE定
			xs,ys = mnist.train.next_batch(BATCH_SIZE)
			#print "xs shape:",xs.shape
			#print "ys shape:",ys.shape
			_,loss_value,step =   sess.run([train_op,loss,global_step],feed_dict={x:xs,y_:ys})
			#每隔1000次输出一次损失值
			if i%1000==0:
				print "After %d training steps , the loss on training batch is %g."%(step,loss_value)
				#保存模型
				saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)

def main():
	#下载mnist数据集	
	mnist = input_data.read_data_sets("./data/",one_hot=True)
	backward(mnist)

if __name__ == '__main__':
	main()


































