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

total_loss=tf.nn.ctc_loss(labels=y_train, logit=final)



""" definir une fonction de loss basée sur le ctc
regarder LSTM et GRU
"""


# layer={}
# layer['layer_1']=Dense(32,activation='relu')
# layer['layer_2']=Dense(64,activation='relu')
# layer['layer_3']=Dense(128,activation='relu')
# output, output_state= LSTM(NB_UNIT,return_sequences=True, return_state=True,activation='relu')
# layer['layer_4']=output
# layer['layer_4_output']=output_state
# layer['layer_5']=Dense(activation='relu')
# layer['layer_6']=Dense(activation='softmax')


# total_loss = tf.nn.ctc_loss(labels=y_train, inputs=layer['layer_6'], sequence_length=batch_seq_len)
