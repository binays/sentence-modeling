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

def load_model_from_disk(model_name,weight_file_name):
	json_file=open(model_name,'r')
	loaded_model_json=json_file.read()
	json_file.close()
	loaded_model=model_from_json(loaded_model_json)
	loaded_model.load_weights(weight_file_name)
	return loaded_model

def get_data(filename):
	file=open(filename,"r").readlines()
	question1=[]
	question2=[]
	y=[]
	for line in file:
		linearr=line.split("\t")
		q1=linearr[0]
		q2=linearr[1]
		l=int(linearr[2])
		question1.append(q1)
		question2.append(q2)
		y.append(l)
	return (question1,question2,np.array(y))

n_dim=300
x=[]
y=[]
BASE_DIR=''
GLOVE_DIR=os.path.join(BASE_DIR,'glove.6B')
max_length=30
batch_size=64
no_gpu=2

(training_q1,training_q2,training_y)=get_data("training.txt")
(test_q1,test_q2,test_y)=get_data("test.txt")
(valid_q1,valid_q2,valid_y)=get_data("validation.txt")

t=Tokenizer(lower=True,char_level=False)
t.fit_on_texts(training_q1)
t.fit_on_texts(training_q2)
t.fit_on_texts(test_q1)
t.fit_on_texts(test_q2)
t.fit_on_texts(valid_q1)
t.fit_on_texts(valid_q2)
training_encoded_q1=t.texts_to_sequences(training_q1)
training_encoded_q2=t.texts_to_sequences(training_q2)

training_padded_q1=pad_sequences(training_encoded_q1,maxlen=max_length,padding='post')
training_padded_q2=pad_sequences(training_encoded_q2,maxlen=max_length,padding='post')
training_padded_q1_back=np.flip(training_padded_q1,1)
training_padded_q2_back=np.flip(training_padded_q2,1)

test_encoded_q1=t.texts_to_sequences(test_q1)
test_encoded_q2=t.texts_to_sequences(test_q2)

test_padded_q1=pad_sequences(test_encoded_q1,maxlen=max_length,padding='post')
test_padded_q2=pad_sequences(test_encoded_q2,maxlen=max_length,padding='post')
test_padded_q1_back=np.flip(test_padded_q1,1)
test_padded_q2_back=np.flip(test_padded_q2,1)

valid_encoded_q1=t.texts_to_sequences(valid_q1)
valid_encoded_q2=t.texts_to_sequences(valid_q2)
valid_padded_q1=pad_sequences(valid_encoded_q1,maxlen=max_length,padding='post')
valid_padded_q2=pad_sequences(valid_encoded_q2,maxlen=max_length,padding='post')
valid_padded_q1_back=np.flip(valid_padded_q1,1)
valid_padded_q2_back=np.flip(valid_padded_q2,1)

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

print("loading embedded model...")
with tf.device("/cpu:0"):
	loaded_model=load_model_from_disk('emb_model.json','emb_model.h5')

print("embedded model loaded")


lstm_layer1=CuDNNLSTM(100,return_sequences=True)
lstm_layer2=CuDNNLSTM(100,return_sequences=True)
#lstm_layer1=LSTM(100,return_sequences=True)
#lstm_layer2=LSTM(100,return_sequences=True)

matching_layer1=MatchingLayer((max_length,max_length))
matching_layer2=MatchingLayer((max_length,max_length))

#aggregate_layer=AggregationLayer()


dropout=Dropout(0.2)
norm=BatchNormalization()

inputs=loaded_model.input 
outputs=loaded_model.output

pre_lstm_q1=dropout(outputs[0])
pre_lstm_q2=dropout(outputs[1])
pre_lstm_q1_back=dropout(outputs[2])
pre_lstm_q2_back=dropout(outputs[3])

lstm_q1=lstm_layer1(pre_lstm_q1)
lstm_q2=lstm_layer2(pre_lstm_q2)
lstm_q1_back=lstm_layer1(pre_lstm_q1_back)
lstm_q2_back=lstm_layer2(pre_lstm_q2_back)

lstm_q1=norm(lstm_q1)
lstm_q2=norm(lstm_q2)
lstm_q1_back=norm(lstm_q1_back)
lstm_q2_back=norm(lstm_q2_back)


matching1=matching_layer1([lstm_q1,lstm_q2])
matching2=matching_layer1([lstm_q1_back,lstm_q2_back])


matching1=Flatten()(matching1)
matching2=Flatten()(matching2)


merged=Concatenate(axis=-1)([matching1,matching2])


output=Dense(512,activation='relu')(merged)
output=dropout(output)
output=Dense(1,activation='sigmoid')(output)



with tf.device("/cpu:0"):
	model_cpu=Model(inputs=inputs,outputs=output)

model=multi_gpu_model(model_cpu,gpus=no_gpu)
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


earlyStopping=EarlyStopping(monitor='val_loss',patience=3,verbose=1,mode='min')
#checkpoint=ModelCheckpoint('best_model.h5',monitor='val_loss',mode='min',save_best_only=True,verbose=1)

class CheckPointer(Callback):

	def __init__(self,model):
		self.model_to_save=model
		self.least_val_loss=math.inf
		self.best_epoch=0
		self.model_to_save=model

	def on_epoch_end(self,epoch,logs={}):
		if logs.get('val_loss')<self.least_val_loss:
			self.least_val_loss=logs.get('val_loss')
			self.best_epoch=epoch 
			self.model_to_save.save_weights('best_model.h5')
			print('validation loss decreased. saving model')

checkpoint=CheckPointer(model_cpu)

model.fit([training_padded_q1,training_padded_q2,training_padded_q1_back,training_padded_q2_back],training_y,epochs=10,validation_data=([valid_padded_q1,valid_padded_q2,valid_padded_q1_back,valid_padded_q2_back],valid_y),batch_size=128,shuffle=True,callbacks=[earlyStopping,checkpoint])

model_cpu.load_weights('best_model.h5')

saved_model=multi_gpu_model(model_cpu,gpus=no_gpu)
saved_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

score=saved_model.evaluate([test_padded_q1,test_padded_q2,test_padded_q1_back,test_padded_q2_back],test_y,verbose=1)
print(score[1])






