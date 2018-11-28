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
