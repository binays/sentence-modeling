from keras.preprocessing.text import text_to_word_sequence
from sklearn.model_selection import train_test_split
import gensim
from gensim.models.word2vec import Word2Vec
from tqdm import tqdm
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale
from keras.models import Sequential,Model
from keras.layers import Dense,Activation,Lambda,add,Input
from keras.preprocessing.sequence import pad_sequences
from keras.utils.training_utils import multi_gpu_model
import tensorflow as tf
from sklearn.svm import SVC 
from sklearn import metrics

text=open("text.txt","r")
labels=open("labels.txt","r")

n_dim=50
wv_epochs=3
siamese=0
x=[]
y=[]
filter='?,.#@!:\"\n'
for line in text:
	temp_list=text_to_word_sequence(line,filters=filter,lower=True,split=' ')
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


text=open("text.txt","r").readlines()
labels=open("labels.txt","r").readlines()
questions=[]
y=[]
data={}
for i,line in enumerate(text):
	try:
		linearr=line.split("?")
		q1=linearr[0]
		q2=linearr[1]
		if len(q1)>0 and len(q2)>0:
			q1=text_to_word_sequence(q1,filters=filter,lower=True,split=' ')
			q2=text_to_word_sequence(q2,filters=filter,lower=True,split=' ')
			questions.append([q1,q2])
			y.append(int(labels[i]))
	except:
		print("error splitting questions")

def create_question_vectors(q,size,i,siamese):
	vec1=np.zeros(size)
	vec2=np.zeros(size)
	q1=q[0]
	q2=q[1]
	count=1.0
	for w in q1:
		tmp=question_w2v[w]
		vec1+=tmp
		count+=1.0
	vec1/=count 
	count=1.0
	for w in q2:
		tmp=question_w2v[w]
		vec2+=tmp
		count+=1.0
	vec2/=count
	if siamese==1:
		return [vec1,vec2]
	else:
		return vec1-vec2

def create_question_vectors_padded(q,size,i,siamese):
	vec1=np.array([])
	vec2=np.array([])
	q1=q[0]
	q2=q[1]
	for w in q1:
		tmp=question_w2v[w]
		vec1=np.concatenate([vec1,tmp])

	for w in q2:
		tmp=question_w2v[w]
		vec2=np.concatenate([vec2,tmp])
	padded=pad_sequences([vec1,vec2])
	if siamese==1:
		return [padded[0],padded[1]]
	else:
		return padded[0]-padded[1]





if siamese==1:

	x=[]
	x_q1=[]
	x_q2=[]
	print("Creating question vectors")
	for i,q in enumerate(questions):
		print("question "+str(i))
		if i==100000:
			break
		x=create_question_vectors(q,n_dim,i,siamese)
		x_q1.append(x[0])
		x_q2.append(x[1])
	print("splitting data")
	x_train_q1,x_test_q1,y_train,y_test=train_test_split(np.array(x_q1),np.array(y[:100000]),test_size=0.2,random_state=10)
	x_train_q2,x_test_q2=train_test_split(np.array(x_q2),test_size=0.2,random_state=10)
	print("done splitting data")
	#x_train_q1=x_train[0]
	##x_train_q2
	#print(x_train.shape)
	#print(y_train.shape)
	#x_train=scale(x_train)
	#x_test=scale(x_test)




	print("Now creating neural net")
	input_q=Input(shape=(n_dim,))
	out=Dense(512,activation='relu')(input_q)
	out=Dense(128,activation='relu')(out)
	#out=Dense(64,activation='relu')(col_q1)



	input_q1=Input(shape=(n_dim,))
	input_q2=Input(shape=(n_dim,))

	model=Model(input_q,out)

	out_q1=model(input_q1)
	out_q2=model(input_q2)

	negative_out_q2=Lambda(lambda x:-x)(out_q2)

	diff=add([out_q1,negative_out_q2])

	final_out=Dense(1,activation='sigmoid')(diff)

	diff_model=Model([input_q1,input_q2],final_out)


	diff_model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

	diff_model.fit([x_train_q1,x_train_q2],y_train,epochs=15,batch_size=32,verbose=2)

	score=diff_model.evaluate([x_test_q1,x_test_q2],y_test,batch_size=32,verbose=2)
	print(score[1])

else:
	x=[]

	print("Creating question vectors")
	for i,q in enumerate(questions):
		print("question "+str(i))
		if i==50000:
			break
		x.append(create_question_vectors_padded(q,n_dim,i,siamese))

	print("padding data")
	x=pad_sequences(x)
	print("splitting data")
	x_train,x_test,y_train,y_test=train_test_split(np.array(x),np.array(y[:50000]),test_size=0.2,random_state=10)
	print("done splitting data")

	logistic=LogisticRegression()

	print("Training logistic regression")
	#logistic.fit(x_train,y_train)

	#print(logistic.score(x_test,y_test))

	print("Training svm")
	clf=SVC(kernel='linear')
	clf.fit(x_train,y_train)
	y_pred=clf.predict(x_test)
	print(metrics.accuracy_score(y_test,y_pred))
	print("creating neural network(vanilla)")
	print("Input shape:"+str(x_train.shape[1]))
	with tf.device("/cpu:0"):
		model_cpu=Sequential()
		model_cpu.add(Dense(512,activation='relu',input_dim=x_train.shape[1]))
		#model.add(Dense(128,activation='relu'))
		model_cpu.add(Dense(1,activation='sigmoid'))

	model=multi_gpu_model(model_cpu,gpus=4)
	model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
	model.fit(x_train,y_train,epochs=10,batch_size=32,verbose=1)
	score=model.evaluate(x_test,y_test,batch_size=32,verbose=1)
	print(score[1])


