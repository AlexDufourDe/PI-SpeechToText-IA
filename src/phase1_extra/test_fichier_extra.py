"""
Ce fichier permet à l'utilisateur de tester le modèle à partir d'un fichier .wav.
Le modèle fait alors son estimation, et affiche le mot retranscrit.

This file can test the model usion g a .wav file and print the result of the prediction .

"""


from scipy.io.wavfile import write
import tensorflow as tf
import numpy as np
import tensorflow_io as tfio
import os
from scipy.io import wavfile
import sys


MOTS = ['Sheila','Zero','Go','Bed','Bird','Stop','Marvin','Yes','Four','House','Off','Tree','Wow','Happy','Nine','Up','Three','Right','Five','Two','One','Left','Eight','Six','Down','Dog','No','Cat','On','Seven']

#Path to file
len_arg = len(sys.argv)
if (len_arg>3):
      print(f"Too many argument, expected : 2 , got {len_arg}")
elif len_arg==1:
      print(f"Not enough argument, expected : 2 , got {len_arg}")
elif len_arg==3:
      CHEMIN_MODELE=sys.argv[2]
      f_name=sys.argv[1]
else: 
      CHEMIN_MODELE = './src/phase1_extra/modeles_extra/mel-cnn'  
      f_name=sys.argv[1]

if not os.path.exists(CHEMIN_MODELE):
      print(f"The model {CHEMIN_MODELE} does not exist")
else:
      if not os.path.exists(f_name):
            print(f"The file {f_name} does not exist")
      else:

            samplerate, data = wavfile.read(f_name)


            # Importation du mpdèle entrainé
            model = tf.keras.models.load_model(CHEMIN_MODELE)


            scaled = np.int16(data* 32767).reshape(16000) # Remise à l'echelle de l'audio

            # Ici, on réapplique le mêmes pré-traitements que pour les données d'entraînements.
            fade = tfio.audio.fade(scaled, fade_in=1000, fade_out=2000, mode="logarithmic")
            spectrogram = tfio.audio.spectrogram(fade, nfft=1024, window=1024, stride=256)
            mel_spectrogram = tfio.audio.melscale(spectrogram, rate=16000, mels=128, fmin=0, fmax=8000)

            test = np.array(tfio.audio.dbscale(mel_spectrogram, top_db=80))[...,np.newaxis]

            # Prediction
            index = np.argmax(model.predict(np.array([test])))

            print("\n")
            print("\nLe mot retranscrit est ---> " + (MOTS[index]).upper() + " <---")
            print("Si ce n'est pas la bonne retranscription, veuillez revérifier le fichier audio, l'erreur vient probablement de "
                  "l'enregistrement ")
            print("\n")

