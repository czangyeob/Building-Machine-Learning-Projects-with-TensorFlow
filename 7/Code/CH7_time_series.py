import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt


from tensorflow.python.framework import dtypes
from tensorflow.contrib import learn

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.layers import Dropout

df = pd.read_csv("data/elec_load.csv", error_bad_lines=False)
plt.subplot()
plot_test, = plt.plot(df.values[:1500], label='Load')
plt.legend(handles=[plot_test])

print(df.describe())
array=(df.values- 147.0) /339.0
plt.subplot()
plot_test, = plt.plot(array[:1500], label='Normalized Load')
plt.legend(handles=[plot_test])


listX = []
listy = []
X={}
y={}

for i in range(0,len(array)-6):
    listX.append(array[i:i+5].reshape([5,1]))
    listy.append(array[i+6])

arrayX=np.array(listX)
arrayy=np.array(listy)


X['train']=arrayX[0:13000]
X['test']=arrayX[13000:14000]

y['train']=arrayy[0:13000]
y['test']=arrayy[13000:14000]

# Build the model

model = Sequential()

model.add(LSTM(input_dim=1, output_dim=50, return_sequences=True))

model.add(Dropout(0.2))

model.add(LSTM(input_dim=100, output_dim=200, return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(output_dim=1))
model.add(Activation("linear"))

model.compile(loss="mse", optimizer="rmsprop")

# Fit the model to the data

model.fit(X['train'], y['train'], batch_size=512, nb_epoch=10, validation_split=0.08)
test_results = model.predict(X['test'])

# Rescale the test dataset and predicted data

test_results = test_results * 339 + 147
y['test'] = y['test'] * 339 + 147


plt.subplot()
plot_predicted, = plt.plot(test_results, label='predicted')

plot_test, = plt.plot(y['test']  , label='test')
plt.legend(handles=[plot_predicted, plot_test])
