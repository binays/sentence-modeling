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


def save_model(model):
	model_json=model.to_json()
	file_name="w2v_emb_model.json"
	with open(file_name,"w") as json_file:
		json_file.write(model_json)
	weight_file="w2v_emb_model.h5"
	model.save_weights(weight_file)

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
wv_epochs=10
siamese=1
x=[]
y=[]
max_length=30

(training_q1,training_q2,training_y)=get_data("training.txt")
(test_q1,test_q2,test_y)=get_data("test.txt")
(valid_q1,valid_q2,valid_y)=get_data("validation.txt")

for t in training_q1:
	x.append(text_to_word_sequence(t,lower=True,split=' '))
for t in training_q2:
	x.append(text_to_word_sequence(t,lower=True,split=' '))
for t in test_q1:
	x.append(text_to_word_sequence(t,lower=True,split=' '))
for t in test_q2:
	x.append(text_to_word_sequence(t,lower=True,split=' '))
for t in valid_q1:
	x.append(text_to_word_sequence(t,lower=True,split=' '))
for t in valid_q2:
	x.append(text_to_word_sequence(t,lower=True,split=' '))



#for line in labels:
#	y.append(int(line))

#print(y[0])

x=np.array(x)
#y=np.array(y)
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

t=Tokenizer(lower=True,char_level=False)
t.fit_on_texts(training_q1)
t.fit_on_texts(training_q2)
t.fit_on_texts(test_q1)
t.fit_on_texts(test_q2)
t.fit_on_texts(valid_q1)
t.fit_on_texts(valid_q2)
vocab_size=len(t.word_index)+1

embedding_matrix=np.zeros((vocab_size,n_dim))

for word,i in t.word_index.items():
	vec=question_w2v[word]
	if vec is not None:
		embedding_matrix[i]=vec 


embedding_layer=Embedding(vocab_size,n_dim,weights=[embedding_matrix],
	input_length=max_length,trainable=False)



q1_input=Input(shape=(max_length,),dtype='int32')
q2_input=Input(shape=(max_length,),dtype='int32')
q1_input_back=Input(shape=(max_length,),dtype='int32')
q2_input_back=Input(shape=(max_length,),dtype='int32')

embedded_q1=embedding_layer(q1_input)
embedded_q2=embedding_layer(q2_input)
embedded_q1_back=embedding_layer(q1_input_back)
embedded_q2_back=embedding_layer(q2_input_back)
with tf.device("/cpu:0"):
	model_cpu=Model(inputs=[q1_input,q2_input,q1_input_back,q2_input_back],outputs=[embedded_q1,
		embedded_q2,embedded_q1_back,embedded_q2_back])
save_model(model_cpu)
print('embedded model saved')
	

