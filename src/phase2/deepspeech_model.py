import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


NB_UNIT=5


#Model
model = Sequential()
model.add(Dense(32,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(LSTM(NB_UNIT,return_sequences=True, return_state=True,activation='relu'))
model.add(Dense(activation='softmax'))
