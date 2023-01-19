""" 
This file  
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from IPython import display
from jiwer import wer
import sys
import datetime

from callback import CallbackEval
from build_model import build_model
from ctc_loss import CTCLoss
from pretaitement import transorm_data






len_arg = len(sys.argv)
if (len_arg>5):
    print(f"Too many argument, expected max :4 , got {len_arg-1}")
    exit() 

elif len_arg==2: #(LANGUE)
    LANGUE=sys.argv[1]

elif len_arg==3:   #(LANGUE  NOMBRE_EPOCH CHEMIN_MODELE) 
    LANGUE=sys.argv[1]
    NB_EPOCH=sys.argv[2]
    CHEMIN_MODELE=sys.argv[3]

elif len_arg==4:   #(LANGUE  NOMBRE_EPOCH CHEMIN_MODELE NOM_MODELE)
    LANGUE=sys.argv[1]
    NB_EPOCH=sys.argv[2]
    CHEMIN_MODELE=sys.argv[3]
    NOM_MODELE=sys.argv[4]

else: 
    LANGUE='fr'
    NB_EPOCH=8
    CHEMIN_MODELE='models'
    NOM_MODELE='deepspeech'



# The set of characters accepted in the transcription.
characters = [x for x in "abcdefghijklmnopqrstuvwxyz'?! "]
# Mapping characters to integers
char_to_num = keras.layers.StringLookup(vocabulary=characters, oov_token="")
# Mapping integers back to original characters
num_to_char = keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)


if LANGUE=='fr':
    chemin='src/phase2/mozilla_common_voice'
    fft_length=1024

elif LANGUE=='en':
    chemin='LJSpeech'
    fft_length=384
else:
    print("The model cannot be train in these language language")
    exit()


train_dataset,validation_dataset=transorm_data(LANGUE)


# Get the model
model = build_model(
    input_dim=fft_length // 2 + 1,
    output_dim=char_to_num.vocabulary_size(),
    CTCLoss=CTCLoss,
    char_to_num=char_to_num,
    rnn_units=512,
)

for i in range(NB_EPOCH):
    #  Define the number of epochs.
    epochs = 1
    # Callback function to check transcription on the val set.
    validation_callback = CallbackEval(validation_dataset)
    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=[validation_callback],
    )


version = open("src/phase2/version_"+LANGUE+".txt", "a")
version.write("\n")
version.write(str(datetime.datetime.today()))
version.write("  "+NOM_MODELE+ " entrainé sur "+chemin+"\n")
version.write(model.summary())
version.write("epoch :"+ str(NB_EPOCH))
version.close()


model.summary()
model.save(CHEMIN_MODELE+'_'+LANGUE+'/'+NOM_MODELE)