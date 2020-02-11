from keras.models import Model 
from keras.layers import Input
from keras.layers import LSTM
from numpy import array

inputs1=Input(shape=(3,2))

lstm1=LSTM(5,return_sequences=True)(inputs1)
model=Model(inputs=inputs1,outputs=lstm1)

data=[[0.1,0.2],[0.3,0.4],[0.5,0.6]]
data=array(data).reshape((1,3,2))

out=model.predict(data)

print(out)
print(out.shape)