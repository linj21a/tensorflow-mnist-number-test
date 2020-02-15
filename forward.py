#coding:utf-8
import tensorflow as tf

#定义输入节点与输出节，隐藏层节点
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500

def get_weight(shape,regularizer):
	w = tf.Variable(tf.truncated_normal(shape,stddev=0.1))
	if regularizer != None: tf.add_to_collection("losses",tf.contrib.layers.l2_regularizer(regularizer)(w))
	return w

def get_bias(shape):
	b = tf.Variable(tf.zeros(shape))
	return b

def forward(x,regularizer):
	w1 = get_weight([INPUT_NODE,LAYER1_NODE],regularizer)
	b1 = get_bias([LAYER1_NODE])
	y1 = tf.nn.relu(tf.matmul(x,w1)+b1)
	#print y1.shape,x.shape

	w2 = get_weight([LAYER1_NODE,OUTPUT_NODE],regularizer)
	b2 = get_bias([OUTPUT_NODE])
	#注意y1必须写在w2前面，保证第一个矩阵的行=第二个矩阵的列。才可以相乘。
	y = tf.matmul(y1,w2)+b2
	return y















