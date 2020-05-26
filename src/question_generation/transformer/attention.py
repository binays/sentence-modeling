import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
from keras.preprocessing.text import text_to_word_sequence
from sklearn.model_selection import train_test_split
import gensim
from gensim.models.word2vec import Word2Vec
from tqdm import tqdm
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale
from keras.models import Sequential,Model,load_model,model_from_json
from keras.layers import Dense,Activation,Lambda,add,Input,LSTM,Embedding,Dropout,CuDNNLSTM
from keras.preprocessing.sequence import pad_sequences
from keras.utils.training_utils import multi_gpu_model
import tensorflow as tf
from keras.layers.merge import concatenate
from keras.preprocessing.text import Tokenizer
from keras import backend as K 
from keras.layers import Layer,Concatenate,Multiply,Dot,Flatten
from keras.layers.normalization import BatchNormalization
import os 
from keras.callbacks import EarlyStopping,ModelCheckpoint
import math
from keras.callbacks import Callback


class MatchingLayer(Layer):
	def __init__(self,output_dim,**kwargs):
		self.output_dim=output_dim
		super(MatchingLayer,self).__init__(**kwargs)

	def build(self,input_shape):
		assert isinstance(input_shape,list)
		self.kernel=self.add_weight(name='kernel',
									shape=(input_shape[0][2],1),
									initializer='uniform',
									trainable=True)
		super(MatchingLayer,self).build(input_shape)
	def call(self,x):
		assert isinstance(x,list)
		a,b=x 
		#print("input shape")
		#print(a.shape)
		#print(b.shape)
		#print("weight shape")
		temp_kernel=self.kernel 
		#print(temp_kernel.shape)


		temp_kernel=K.reshape(temp_kernel,(temp_kernel.shape[1],temp_kernel.shape[0]))
		#print(temp_kernel.shape)
		temp_kernel=K.repeat(temp_kernel,max_length)
		#print(temp_kernel.shape)
		ext_kernel=temp_kernel

		#multiplying each time steps of first input with weight
		res1=Multiply()([a,ext_kernel])
		
		#multiplying each time steps of second input with weight
		res2=Multiply()([b,ext_kernel])
		
		#print(res1.shape)
		#print(res2.shape)

		#computing cosine similarity between each time steps of first input to
		## each time steps of second input
		out=Dot(axes=2,normalize=True)([res1,res2])
		#print(out.shape)
		return (out)

	
	def compute_output_shape(self,input_shape):
		assert isinstance(input_shape,list)
		shape_a,shape_b=input_shape
		return (shape_a[0],self.output_dim[0],self.output_dim[1])