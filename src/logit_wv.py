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


