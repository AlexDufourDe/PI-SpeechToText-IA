"""
Ce fichier permet à l'utilisateur de tester le modèle à partir d'un fichier .wav.
Le modèle fait alors son estimation, et affiche le mot retranscrit.
Pour faciliter l'utilisation, l'enregistrement est fait sur 3 secondes, qui sera ensuite réduit à une fenetre de 1 seconde pour le modèle
Penser à verifier le fichier audio en sortie si l'estimation du modèle n'est pas correcte.
"""


from scipy.io.wavfile import write
import tensorflow as tf
import numpy as np
import tensorflow_io as tfio
import wave
from scipy.io import wavfile
import sys


MOTS = ['yes','no','up','down','right','left','stop','go','on','off']
CHEMIN_MODELE = './modeles/mel-cnn'  #Chemin du modèle que l'on souhaite tester

#Path to file
f_name = "./output.wav"
 
# create WaveObject instances
# directly from WAV files on disk

 

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

