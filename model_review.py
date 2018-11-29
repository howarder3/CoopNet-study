# model.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time
from six.moves import xrange

from model.utils.interpolate import *
from model.utils.custom_ops import *
from model.utils.data_io import DataSet, saveSampleResults
from model.utils.vis_util import Visualizer

class CoopNets(object):
    def __init__(self, num_epochs=200, image_size=64, batch_size=100, nTileRow=12, nTileCol=12, net_type='object',
                 d_lr=0.001, g_lr=0.0001, beta1=0.5,
                 des_step_size=0.002, des_sample_steps=10, des_refsig=0.016,
                 gen_step_size=0.1, gen_sample_steps=0, gen_refsig=0.3,
                 data_path='/tmp/data/', log_step=10, category='rock',
                 sample_dir='./synthesis', model_dir='./checkpoints', log_dir='./log', test_dir='./test'):
				 
		self.type = net_type #底下會用net_type決定z_size
		self.num_epochs = num_epochs # 200 epochs
		self.batch_size = batch_size # 100 batches
		self.image_size = image_size # 64 image size
		self.nTileRow = nTileRow # 12 
		self.nTileCol = nTileCol # 12
		self.num_chain = nTileRow * nTileCol # 144 12*12
		self.beta1 = beta1 # 0.5

		self.d_lr = d_lr # 0.001
		self.g_lr = g_lr # 0.0001
		self.delta1 = des_step_size # 0.002
		self.sigma1 = des_refsig #0.016
		self.delta2 = gen_step_size # 0.1
		self.sigma2 = gen_refsig # 0.3
		self.t1 = des_sample_steps # 10
		self.t2 = gen_sample_steps # 0

		self.data_path = os.path.join(data_path, category) #檔案路徑
		self.log_step = log_step # 10

		self.log_dir = log_dir # log位置
		self.sample_dir = sample_dir # sample 位置 (synthesis)
		self.model_dir = model_dir # model 位置 (checkpoints)
		self.test_dir = test_dir # test 位置 (test)

		if self.type == 'texture':
            self.z_size = 49
        elif self.type == 'object':
            self.z_size = 100
        elif self.type == 'object_small':
            self.z_size = 2
			
		self.syn = tf.placeholder(shape=[None, self.image_size, self.image_size, 3], dtype=tf.float32, name='syn') # placeholder 放syn size*size*3
        self.obs = tf.placeholder(shape=[None, self.image_size, self.image_size, 3], dtype=tf.float32, name='obs') # placeholder 放obs size*size*3
        self.z = tf.placeholder(shape=[None, self.z_size], dtype=tf.float32, name='z') # placeholder 放z size
		
		self.debug = False
		
	def build_model(self):
        self.gen_res = self.generator(self.z, reuse=False) # 拿出placeholder中的size(第一層conv的初始size)	
		'''
		def generator(self, inputs, reuse=False, is_training=True):
			with tf.variable_scope('gen', reuse=reuse):
				if self.type == 'object':
					inputs = tf.reshape(inputs, [-1, 1, 1, self.z_size])
					convt1 = convt2d(inputs, (None, self.image_size // 16, self.image_size // 16, 512), kernal=(4, 4)
									 , strides=(1, 1), padding="VALID", name="convt1")
					convt1 = tf.contrib.layers.batch_norm(convt1, is_training=is_training)
					convt1 = leaky_relu(convt1)

					convt2 = convt2d(convt1, (None, self.image_size // 8, self.image_size // 8, 256), kernal=(5, 5)
									 , strides=(2, 2), padding="SAME", name="convt2")
					convt2 = tf.contrib.layers.batch_norm(convt2, is_training=is_training)
					convt2 = leaky_relu(convt2)

					convt3 = convt2d(convt2, (None, self.image_size // 4, self.image_size // 4, 128), kernal=(5, 5)
									 , strides=(2, 2), padding="SAME", name="convt3")
					convt3 = tf.contrib.layers.batch_norm(convt3, is_training=is_training)
					convt3 = leaky_relu(convt3)

					convt4 = convt2d(convt3, (None, self.image_size // 2, self.image_size // 2, 64), kernal=(5, 5)
									 , strides=(2, 2), padding="SAME", name="convt4")
					convt4 = tf.contrib.layers.batch_norm(convt4, is_training=is_training)
					convt4 = leaky_relu(convt4)

					convt5 = convt2d(convt4, (None, self.image_size, self.image_size, 3), kernal=(5, 5)
									 , strides=(2, 2), padding="SAME", name="convt5")
					convt5 = tf.nn.tanh(convt5)

					return convt5
				else:
					return NotImplementedError		
		'''
		obs_res = self.descriptor(self.obs, reuse=False)
        syn_res = self.descriptor(self.syn, reuse=True)
		'''
		def descriptor(self, inputs, reuse=False):
			with tf.variable_scope('des', reuse=reuse):
				if self.type == 'object':
					conv1 = conv2d(inputs, 64, kernal=(5, 5), strides=(2, 2), padding="SAME", activate_fn=leaky_relu,
								   name="conv1")

					conv2 = conv2d(conv1, 128, kernal=(3, 3), strides=(2, 2), padding="SAME", activate_fn=leaky_relu,
								   name="conv2")

					conv3 = conv2d(conv2, 256, kernal=(3, 3), strides=(1, 1), padding="SAME", activate_fn=leaky_relu,
								   name="conv3")

					fc = fully_connected(conv3, 100, name="fc")

					return fc
				else:
					return NotImplementedError
		'''
		
