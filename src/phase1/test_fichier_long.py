"""
Ce fichier permet à l'utilisateur de tester le modèle à partir d'un fichier .wav.
Le modèle fait alors son estimation, et affiche le mot retranscrit.

This file can test the model using  a .wav file and print the result of the prediction .

"""

import argparse
from scipy.io.wavfile import write
import tensorflow as tf
import numpy as np
import tensorflow_io as tfio
import os
from scipy.io import wavfile
import sys
import matplotlib.pyplot as plt


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


samplerate, data = wavfile.read(f_name)


# Importation du mpdèle entrainé
model = tf.keras.models.load_model(CHEMIN_MODELE)

# Exctraction de la seconde où il y'a le plus de son
median=np.sqrt(np.mean((data[:])**2))
rms_max=0
debut = 0
segment=[]
rms=[]
rms.append(np.sqrt(np.mean((data[0:16000])**2)))
rms.append(np.sqrt(np.mean((data[1:1+16000])**2)))
for i in range(2,len(data)-16000):
      rms.append(np.sqrt(np.mean((data[i:i+16000])**2)))
      if (  rms[i-2]<rms[i-1] and rms[i-1]>rms[i] ):
            if (rms[i-1]>rms_max and rms[i-1]-2>median):
                  rms_max=rms[i-1]
                  debut=i-1
      if rms[i-1]<median:
            if(rms_max!=0):
                  segment.append(data[debut:debut+16000])
            rms_max=0
      



print("###########################################################################################")
print(f" audio : {len(data)} ; segment {len(segment)}")
print("###########################################################################################")


index=[]
for scaled in segment:

      # scaled = np.int16(data* 32767).reshape(16000) # Remise à l'echelle de l'audio

      # Ici, on réapplique le mêmes pré-traitements que pour les données d'entraînements.
      fade = tfio.audio.fade(scaled, fade_in=1000, fade_out=2000, mode="logarithmic")
      spectrogram = tfio.audio.spectrogram(fade, nfft=1024, window=1024, stride=256)
      mel_spectrogram = tfio.audio.melscale(spectrogram, rate=16000, mels=128, fmin=0, fmax=8000)

      test = np.array(tfio.audio.dbscale(mel_spectrogram, top_db=80))[...,np.newaxis]

      # Prediction
      index.append(np.argmax(model.predict(np.array([test]))))

mots_traduit=''
for i in index:
      mots_traduit=mots_traduit+"     "+(MOTS[i]).upper() 


print("\n")
print(f"Il y a {len(index)} mots prononcés")
print("\nLes mot retranscris sont :" + mots_traduit + " <---")
print("Si ce n'est pas la bonne retranscription, veuillez revérifier le fichier audio, l'erreur vient probablement de "
      "l'enregistrement ")
print("\n")

