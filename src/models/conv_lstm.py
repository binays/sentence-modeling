import os 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
from keras.preprocessing.text import text_to_word_sequence
from sklearn.model_selection import train_test_split
import gensim
from gensim.models.word2vec import Word2Vec
from tqdm import tqdm
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale
from keras.models import Sequential,Model,load_model,model_from_json
from keras.layers import Dense,Activation,Lambda,add,Input,LSTM,Embedding,Dropout,CuDNNLSTM,Reshape
from keras.preprocessing.sequence import pad_sequences
from keras.utils.training_utils import multi_gpu_model
import tensorflow as tf
from keras.layers.merge import concatenate
from keras.preprocessing.text import Tokenizer
from keras import backend as K 
from keras.layers import Layer,Concatenate,Multiply,Dot,Flatten,Conv3D
from keras.layers.normalization import BatchNormalization
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
no_gpu=4
model_name='emb_model'

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

training_encoded_q1_back=[]
training_encoded_q2_back=[]
for q1,q2 in zip(training_encoded_q1,training_encoded_q2):
	training_encoded_q1_back.append(list(reversed(q1)))
	training_encoded_q2_back.append(list(reversed(q2)))

print(training_encoded_q1[0])
print(training_encoded_q1_back[0])


training_padded_q1=pad_sequences(training_encoded_q1,maxlen=max_length,padding='post')
training_padded_q2=pad_sequences(training_encoded_q2,maxlen=max_length,padding='post')
training_padded_q1_back=pad_sequences(training_encoded_q1_back,maxlen=max_length,padding='post')
training_padded_q2_back=pad_sequences(training_encoded_q2_back,maxlen=max_length,padding='post')

test_encoded_q1=t.texts_to_sequences(test_q1)
test_encoded_q2=t.texts_to_sequences(test_q2)
test_encoded_q1_back=[]
test_encoded_q2_back=[]
for q1,q2 in zip(test_encoded_q1,test_encoded_q2):
	test_encoded_q1_back.append(list(reversed(q1)))
	test_encoded_q2_back.append(list(reversed(q2)))

test_padded_q1=pad_sequences(test_encoded_q1,maxlen=max_length,padding='post')
test_padded_q2=pad_sequences(test_encoded_q2,maxlen=max_length,padding='post')
test_padded_q1_back=pad_sequences(test_encoded_q1_back,maxlen=max_length,padding='post')
test_padded_q2_back=pad_sequences(test_encoded_q2_back,maxlen=max_length,padding='post')

valid_encoded_q1=t.texts_to_sequences(valid_q1)
valid_encoded_q2=t.texts_to_sequences(valid_q2)
valid_encoded_q1_back=[]
valid_encoded_q2_back=[]
for q1,q2 in zip(valid_encoded_q1,valid_encoded_q2):
	valid_encoded_q1_back.append(list(reversed(q1)))
	valid_encoded_q2_back.append(list(reversed(q2)))


valid_padded_q1=pad_sequences(valid_encoded_q1,maxlen=max_length,padding='post')
valid_padded_q2=pad_sequences(valid_encoded_q2,maxlen=max_length,padding='post')
valid_padded_q1_back=pad_sequences(valid_encoded_q1_back,maxlen=max_length,padding='post')
valid_padded_q2_back=pad_sequences(valid_encoded_q2_back,maxlen=max_length,padding='post')


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


with tf.device("/cpu:0"):
	loaded_model=load_model_from_disk(model_name+'.json',model_name+'.h5')

print("embedded model loaded")

lstm_layer1=CuDNNLSTM(100,return_sequences=True)
lstm_layer2=CuDNNLSTM(100,return_sequences=True)

matching_layer1=MatchingLayer((max_length,max_length))
matching_layer2=MatchingLayer((max_length,max_length))

#aggregate_layer=AggregationLayer()


#q1_input=Input(shape=(max_length,),dtype='int32')
#q2_input=Input(shape=(max_length,),dtype='int32')
#q1_input_back=Input(shape=(max_length,),dtype='int32')
#q2_input_back=Input(shape=(max_length,),dtype='int32')


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

#lstm_q1=norm(lstm_q1)
#lstm_q2=norm(lstm_q2)
#lstm_q1_back=norm(lstm_q1_back)
#lstm_q2_back=norm(lstm_q2_back)



print(type(lstm_q1))
print(type(lstm_q2))
print(lstm_q1.shape)
print(lstm_q2.shape)

lambda_layer=Lambda(lambda x: x)
#lstm_q1=K.reshape(lstm_q1,(K.shape(lstm_q1)[0],1,lstm_q1.shape[1],lstm_q1.shape[2]))
#lstm_q2=K.reshape(lstm_q2,(K.shape(lstm_q2)[0],1,lstm_q2.shape[1],lstm_q2.shape[2]))
#lstm_q1_back=K.reshape(lstm_q1_back,(K.shape(lstm_q1_back)[0],1,lstm_q1_back.shape[1],lstm_q1_back.shape[2]))
#lstm_q2_back=K.reshape(lstm_q2_back,(K.shape(lstm_q2_back)[0],1,lstm_q2_back.shape[1],lstm_q2_back.shape[2]))

lstm_q1=Lambda(lambda x: K.reshape(x,(K.shape(x)[0],1,x.shape[1],x.shape[2])))(lstm_q1)
lstm_q2=Lambda(lambda x: K.reshape(x,(K.shape(x)[0],1,x.shape[1],x.shape[2])))(lstm_q2)
lstm_q1_back=Lambda(lambda x: K.reshape(x,(K.shape(x)[0],1,x.shape[1],x.shape[2])))(lstm_q1_back)
lstm_q2_back=Lambda(lambda x :K.reshape(x,(K.shape(x)[0],1,x.shape[1],x.shape[2])))(lstm_q2_back)

merged=Concatenate(axis=1)([lstm_q1,lstm_q2])
merged_back=Concatenate(axis=1)([lstm_q1_back,lstm_q2_back])

#merged=K.reshape(merged,(K.shape(merged)[0],merged.shape[1],merged.shape[2],merged.shape[3],1))
#merged_back=K.reshape(merged_back,(K.shape(merged_back)[0],merged_back.shape[1],merged_back.shape[2],merged_back.shape[3],1))

merged=Lambda(lambda x: K.reshape(x,(K.shape(x)[0],x.shape[1],x.shape[2],x.shape[3],1)))(merged)
merged_back=Lambda(lambda x: K.reshape(x,(K.shape(x)[0],x.shape[1],x.shape[2],x.shape[3],1)))(merged_back)


print("merged shape")
print(merged.shape)

#merged=lambda_layer(merged)
#merged_back=lambda_layer(merged_back)
conv3D=Conv3D(64,kernel_size=(2,1,100),strides=(1,1,1),padding='valid',activation='relu')
conv3D_2=Conv3D(64,kernel_size=(2,2,100),strides=(1,1,1),padding='valid',activation='relu')
conv3D_3=Conv3D(64,kernel_size=(2,3,100),strides=(1,1,1),padding='valid',activation='relu')
conv3D_4=Conv3D(64,kernel_size=(2,4,100),strides=(1,1,1),padding='valid',activation='relu')

merged=dropout(merged)
merged_back=dropout(merged_back)
matched=conv3D(merged)
matched_2=conv3D_2(merged)
matched_3=conv3D_3(merged)
matched_4=conv3D_4(merged)

matched_back=conv3D(merged_back)
matched_2_back=conv3D_2(merged_back)
matched_3_back=conv3D_3(merged_back)
print("matched shape")
print(matched.shape)

flatten=Flatten()(matched)
flatten_2=Flatten()(matched_2)
flatten_3=Flatten()(matched_3)
flatten_4=Flatten()(matched_4)
flatten_back=Flatten()(matched_back)
flatten_2_back=Flatten()(matched_2_back)
flatten_3_back=Flatten()(matched_3_back)
output=Concatenate(axis=-1)([flatten,flatten_2,flatten_3,flatten_back,flatten_2_back,flatten_3_back])
output=dropout(output)
output=Dense(1024,activation='relu')(output)
#output=BatchNormalization()(output)
#output=dropout(output)
#output=Dense(512,activation='relu')(output)
#output=BatchNormalization()(output)
output=Dense(1,activation='sigmoid')(output)



with tf.device("/cpu:0"):
	model_cpu=Model(inputs=inputs,outputs=output)

model=multi_gpu_model(model_cpu,gpus=no_gpu)
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


earlyStopping=EarlyStopping(monitor='val_loss',patience=6,verbose=1,mode='min')
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

model.fit([training_padded_q1,training_padded_q2,training_padded_q1_back,training_padded_q2_back],training_y,epochs=30,validation_data=([valid_padded_q1,valid_padded_q2,valid_padded_q1_back,valid_padded_q2_back],valid_y),batch_size=128,shuffle=True,callbacks=[earlyStopping,checkpoint])

model_cpu.load_weights('best_model.h5')

saved_model=multi_gpu_model(model_cpu,gpus=no_gpu)
saved_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

score=saved_model.evaluate([test_padded_q1,test_padded_q2,test_padded_q1_back,test_padded_q2_back],test_y,verbose=1)
print(score[1])


#prediction=model.predict([test_padded_q1,test_padded_q2])

#count=0
#output=open("wrong_prediction1.txt","a")
#for p in prediction:
#	if p[0]<0.5 and y_test[count]==1:
#		output.write(x_test_q1[count]+" "+x_test_q2[count]+"\t"+"Pred:"+str(p[0])+"True:"+str(y_test[count])+"\n")
#	elif p[0]>=0.5 and y_test[count]==0:
#		output.write(x_test_q1[count]+" "+x_test_q2[count]+"\t"+"Pred:"+str(p[0])+"True:"+str(y_test[count])+"\n")
#	count+=1

#output.close()





