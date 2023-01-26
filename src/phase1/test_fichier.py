"""
Ce fichier permet à l'utilisateur de tester le modèle à partir d'un fichier .wav.
Le modèle fait alors son estimation, et affiche le mot retranscrit.

This file can test the model usion g a .wav file and print the result of the prediction .

"""

import argparse
from scipy.io.wavfile import write
import tensorflow as tf
import numpy as np
import tensorflow_io as tfio
import os
from scipy.io import wavfile
import sys


MOTS = ['yes','no','up','down','right','left','stop','go','on','off']

# Command line parser

parser = argparse.ArgumentParser(description="Test the transcription of a wav file with a CNN model",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-p", "--path",  help="Path to the folder containing the .wav files")
parser.add_argument("-m","--model",help="Path to the folder of the model" )

args = vars(parser.parse_args())
f_name=args['path']



if (not f_name):
      print(f"Not enough argument, you have to specify the path to the file using -p")
      exit()

if(args['model']):
      CHEMIN_MODELE=args['model']
else: 
      CHEMIN_MODELE = './src/phase1/modeles/mel-cnn'  


if not os.path.exists(CHEMIN_MODELE):
      print(f"The model {CHEMIN_MODELE} does not exist")
      exit()

if not os.path.exists(f_name):
      print(f"The file {f_name} does not exist")
      exit()

model = tf.keras.models.load_model(CHEMIN_MODELE)


samplerate, data = wavfile.read(f_name)
rms_max = 0
debut = 0
for i in range(len(data)-16000):
      rms=np.mean((data[i:i+16000])**2)
      if rms > rms_max:
            rms_max = rms
            debut = i
scaled = data[debut:debut+16000]

# scaled = np.int16(data* 32767).reshape(16000) # Remise à l'echelle de l'audio

# Ici, on réapplique le mêmes pré-traitements que pour les données d'entraînements.
fade = tfio.audio.fade(scaled, fade_in=1000, fade_out=2000, mode="logarithmic")
spectrogram = tfio.audio.spectrogram(fade, nfft=1024, window=1024, stride=256)
mel_spectrogram = tfio.audio.melscale(spectrogram, rate=16000, mels=128, fmin=0, fmax=8000)
test = np.array(tfio.audio.dbscale(mel_spectrogram, top_db=80))[...,np.newaxis]

# test=np.load(f_name)[...,np.newaxis]

# Prediction
index = np.argmax(model.predict(np.array([test])))

print("\n")
print("\nLe mot retranscrit est ---> " + (MOTS[index]).upper() + " <---")
print("Si ce n'est pas la bonne retranscription, veuillez revérifier le fichier audio, l'erreur vient probablement de "
      "l'enregistrement ")
print("\n")

