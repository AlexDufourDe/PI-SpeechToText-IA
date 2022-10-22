"""
Ce fichier permet à l'utilisateur de tester le modèle par lui même, en prononcant un mot après le signal.
Le modèle fait alors son estimation, et affiche le mot retranscrit.
Pour faciliter l'utilisation, l'enregistrement est fait sur 3 secondes, qui sera ensuite réduit à une fenetre de 1 seconde pour le modèle
Penser à verifier le fichier audio en sortie si l'estimation du modèle n'est pas correcte.
"""

import sounddevice as sd
from scipy.io.wavfile import write
from time import sleep
import tensorflow as tf
import numpy as np
import tensorflow_io as tfio

MOTS = ['yes','no','up','down','right','left','stop','go','on','off']
CHEMIN_MODELE = './modeles/mel-cnn'  #Chemin du modèle que l'on souhaite tester

fs = 16000  # Fréquence d'echantillonage (en Hz)
secondes = 3  # Durée de l'enregistrement


print("Enregistrement dans 3")
sleep(1)
print("2")
sleep(1)
print("1")
sleep(1)
print('go!')
enregistrement = sd.rec(int(secondes * fs), samplerate=fs, channels=1)
sd.wait()  # Wait until recording is finished
print('fin')

# Exctraction de la seconde où il y'a le plus de son
rms_max = 0
debut = 0
for i in range(16000*secondes-16000):
      rms=np.mean((enregistrement[i:i+16000])**2)
      if rms > rms_max:
            rms_max = rms
            debut = i
enregistrement = enregistrement[debut:debut+16000]


print("l'enregistrement est sauvegardé dans un fichier 'output.wav' pour vérification")
write('output.wav', fs, enregistrement)

# Importation du mpdèle entrainé
model = tf.keras.models.load_model(CHEMIN_MODELE)


scaled = np.int16(enregistrement * 32767).reshape(16000) # Remise à l'echelle de l'audio

# Ici, on réapplique le mêmes pré-traitements que pour les données d'entraînements.
fade = tfio.audio.fade(scaled, fade_in=1000, fade_out=2000, mode="logarithmic")
spectrogram = tfio.audio.spectrogram(fade, nfft=1024, window=1024, stride=256)
mel_spectrogram = tfio.audio.melscale(spectrogram, rate=16000, mels=128, fmin=0, fmax=8000)
test = np.array(tfio.audio.dbscale(mel_spectrogram, top_db=80))[...,np.newaxis]

# Prediction
index = np.argmax(model.predict(np.array([test])))

print("Le mot retranscrit est ---> " + (MOTS[index]).upper() + " <---")
print("Si ce n'est pas la bonne retranscription, veuillez revérifier le fichier audio, l'erreur vient probablement de "
      "l'enregistrement ")

