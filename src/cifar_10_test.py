import numpy as np 
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,Convolution2D,MaxPooling2D
from keras.utils import np_utils
from keras.datasets import cifar10
from keras.datasets import mnist
from keras.utils import plot_model
from keras.models import model_from_json
from keras import backend as K 
from keras.engine.topology import Layer 

class MyLayer(Layer):

	def __init__(self,output_dim,**kwargs):
		self.output_dim=output_dim
		super(MyLayer,self).__init__(**kwargs)

	def build(self,input_shape):
		self.kernel=self.add_weight(name='kernel',
									shape=(input_shape[1],self.output_dim),
									initializer='uniform',
									trainable=True)
		super(MyLayer,self).build(input_shape)

	def call(self,x):
		print("input shape")
		print(x.shape)
		y=K.dot(x,self.kernel)
		return y 

	def compute_output_shape(self,input_shape):
		return (input_shape[0],self.output_dim)

(x_train,y_train),(x_test,y_test)=cifar10.load_data()

#x_train=x_train.reshape(x_train.shape[0],28,28,1)
#x_test=x_test.reshape(x_test.shape[0],28,28,1)
x_train=x_train.astype('float32')
x_test=x_test.astype('float32')
x_train/=255
x_test/=255
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)



y_train=np_utils.to_categorical(y_train,10)
y_test=np_utils.to_categorical(y_test,10)


model=Sequential()
model.add(Convolution2D(32,3,padding="same",activation="relu",input_shape=(32,32,3)))
model.add(Convolution2D(32,3,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(32,3,padding="same",activation="relu"))
#model.add(Convolution2D(32,3,padding="same",activation="relu"))
#model.add(Convolution2D(32,3,padding="same",activation="relu"))

model.add(Convolution2D(48,3,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(32,3,padding="same",activation="relu"))
model.add(Convolution2D(32,3,padding="same",activation="relu"))

model.add(Dropout(0.4))
model.add(Flatten())
dense512=MyLayer(512)
model.add(dense512)
model.add(Activation('relu'))
dense10=MyLayer(10)
model.add(dense10)
model.add(Activation('softmax'))


model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

model.summary()



#plot_model(model,to_file='model.png')
model.fit(x_train,y_train,batch_size=128,nb_epoch=1,verbose=1)

score=model.evaluate(x_test,y_test,verbose=1)

print(score)