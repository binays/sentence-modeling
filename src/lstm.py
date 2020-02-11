from keras.preprocessing.text import text_to_word_sequence
from sklearn.model_selection import train_test_split
import gensim
from gensim.models.word2vec import Word2Vec
from tqdm import tqdm
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale
from keras.models import Sequential,Model
from keras.layers import Dense,Activation,Lambda,add,Input,LSTM,Embedding,Dropout,CuDNNLSTM
from keras.preprocessing.sequence import pad_sequences
from keras.utils.training_utils import multi_gpu_model
import tensorflow as tf
from keras.layers.merge import concatenate
from keras.preprocessing.text import Tokenizer
from keras import backend as K 
from keras.layers import Layer,Concatenate,Multiply,Dot,Flatten


text=open("text.txt","r")
labels=open("labels.txt","r")

n_dim=300
wv_epochs=5
siamese=1
x=[]
y=[]

filter='?,.#@!:\"\'\n'
count=1
for line in text:
	temp_list=text_to_word_sequence(line,lower=True,split=' ')
	x.append(temp_list)
	


#print(x[0])
for line in labels:
	y.append(int(line))

#print(y[0])
text.close()
labels.close()
x=np.array(x)
y=np.array(y)
#x_train,x_test,y_train,y_test=train_test_split(np.array(x),np.array(y),test_size=0.2)

LabeledSentence=gensim.models.doc2vec.LabeledSentence

def labelizeText(texts,label_type):
	labelized=[]
	for i,v in tqdm(enumerate(texts)):
		label='%s_%s'%(label_type,i)
		labelized.append(LabeledSentence(v,[label]))
	return labelized

x=labelizeText(x,'TRAIN')
#x_test=labelizeText(x_test,'TEST')


question_w2v=Word2Vec(size=n_dim,min_count=0)
question_w2v.build_vocab([ex.words for ex in tqdm(x)],progress_per=10000)

print("Training word2vec")
question_w2v.train([ex.words for ex in tqdm(x)],total_examples=question_w2v.corpus_count,epochs=wv_epochs,report_delay=1)

pretrained_weights=question_w2v.wv.syn0
vocab_size,embedding_size=pretrained_weights.shape



text=open("text.txt","r").readlines()
labels=open("labels.txt","r").readlines()
questions_1=[]
questions_2=[]
questions=[]
y=[]

for i,line in enumerate(text):
	try:
		linearr=line.split("?")
		q1=linearr[0]
		q2=linearr[1]
		if len(q1)>0 and len(q2)>0:
			questions_1.append(q1)
			questions_2.append(q2)
			questions.append(q1)
			questions.append(q2)
			y.append(int(labels[i]))
	except:
		print("error splitting questions")

x_train_q1,x_test_q1,y_train,y_test=train_test_split(questions_1,y,test_size=0.2,random_state=10)
x_train_q2,x_test_q2=train_test_split(questions_2,test_size=0.2,random_state=10)



t1=Tokenizer(lower=True,char_level=False)
t2=Tokenizer(lower=True,char_level=False)

x_train=x_train_q1+x_train_q2

t1.fit_on_texts(x_train_q1)
t2.fit_on_texts(x_train_q2)
vocab_size1=len(t1.word_index)+1
vocab_size2=len(t2.word_index)+1


encoded_q1=t1.texts_to_sequences(x_train_q1)
encoded_q2=t2.texts_to_sequences(x_train_q2)

max_length=30
batch_size=64

padded_q1=pad_sequences(encoded_q1,maxlen=max_length,padding='post')
padded_q2=pad_sequences(encoded_q2,maxlen=max_length,padding='post')

print(padded_q1[0].shape)
print(padded_q2[0].shape)

test_encoded_q1=t1.texts_to_sequences(x_test_q1)
test_encoded_q2=t2.texts_to_sequences(x_test_q2)

test_padded_q1=pad_sequences(test_encoded_q1,maxlen=max_length,padding='post')
test_padded_q2=pad_sequences(test_encoded_q2,maxlen=max_length,padding='post')

print(test_padded_q1[0].shape)
print(test_padded_q2[0].shape)

print(y_test[0])

embedding_matrix_1=np.zeros((vocab_size1,n_dim))

for word,i in t1.word_index.items():
	try:
		vec=question_w2v[word]
		embedding_matrix_1[i]=vec 
	except:
		print("word not found")

embedding_matrix_2=np.zeros((vocab_size2,n_dim))
for word,i in t2.word_index.items():
	try:
		vec=question_w2v[word]
		embedding_matrix_2[i]=vec 
	except:
		print("word not found")

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



rate_drop_lstm = 0.15 + np.random.rand() * 0.25
rate_drop_dense = 0.15 + np.random.rand() * 0.25


embedding_layer1=Embedding(vocab_size1,n_dim,weights=[embedding_matrix_1],
	input_length=max_length,trainable=False)

embedding_layer2=Embedding(vocab_size2,n_dim,weights=[embedding_matrix_2],
	input_length=max_length,trainable=False)


lstm_layer1=CuDNNLSTM(100,return_sequences=True)
lstm_layer2=CuDNNLSTM(100,return_sequences=True)
#lstm_layer1=LSTM(100,return_sequences=True)
#lstm_layer2=LSTM(100,return_sequences=True)

matching_layer1=MatchingLayer((max_length,max_length))
matching_layer2=MatchingLayer((max_length,max_length))

#aggregate_layer=AggregationLayer()


q1_input=Input(shape=(max_length,),dtype='int32')
q2_input=Input(shape=(max_length,),dtype='int32')
embedded_q1=embedding_layer1(q1_input)
embedded_q2=embedding_layer2(q2_input)
lstm_q1=lstm_layer1(embedded_q1)
lstm_q2=lstm_layer2(embedded_q2)

matching1=matching_layer1([lstm_q1,lstm_q2])
#matching2=matching_layer2([lstm_q2,lstm_q1])

matching1=Flatten()(matching1)
#matching2=Flatten()(matching2)

#merged=Concatenate(axis=-1)([matching1,matching2])


output=Dense(512,activation='relu')(matching1)
output=Dense(1,activation='sigmoid')(output)

with tf.device("/cpu:0"):
	model_cpu=Model(inputs=[q1_input,q2_input],outputs=output)

model=multi_gpu_model(model_cpu,gpus=4)
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


model.fit([padded_q1,padded_q2],y_train,epochs=2,batch_size=128,shuffle=True)


score=model.evaluate([test_padded_q1,test_padded_q2],y_test,verbose=1)
print(score[1])
print(x_test_q1[0]+" "+x_test_q2[0])

prediction=model.predict([test_padded_q1,test_padded_q2])

count=0
output=open("wrong_prediction1.txt","a")
for p in prediction:
	if p[0]<0.5 and y_test[count]==1:
		output.write(x_test_q1[count]+" "+x_test_q2[count]+"\t"+"Pred:"+str(p[0])+"True:"+str(y_test[count])+"\n")
	elif p[0]>=0.5 and y_test[count]==0:
		output.write(x_test_q1[count]+" "+x_test_q2[count]+"\t"+"Pred:"+str(p[0])+"True:"+str(y_test[count])+"\n")
	count+=1

output.close()





