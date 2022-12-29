import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import numpy as np


NB_UNIT=5
CHEMIN_DONNEES="mozilla_common_voice_pretraitee"

x_train, y_train = np.load(CHEMIN_DONNEES+'/train_data.npy')[...,np.newaxis], np.load(CHEMIN_DONNEES+'/train_labels.npy')
x_test, y_test = np.load(CHEMIN_DONNEES+'/test_data.npy')[...,np.newaxis], np.load(CHEMIN_DONNEES+'/test_labels.npy')




#Model
model = Sequential()
model.add(Dense(32,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(LSTM(NB_UNIT,return_sequences=True, return_state=True,activation='relu'))
final=model.add(Dense(activation='softmax'))

total_loss=tf.nn.ctc_loss()
