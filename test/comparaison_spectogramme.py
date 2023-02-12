""""
Ce fichier compare deux spectogrammes, un fait par la carte d'evaluation , l'autre à l'aide d'un module de tensorflow.
Il fait également la prediction mots représenté par les spectogrammes. 
Il prend différents paramètres en entrée tel que les chemins des deux fichiers, le modèle a utilisé, le mot qui doit 
etre prédit ainsi que des options pour l'affichage et la sauvegarde des graphiques.

This file compare two spectogram, one made by the ealuation card and the other by a modul of tensorflow.
It also predict the meanning of the two represented words. Differents parameters can be passed to the program as the path to the
file,the model to use to predict the words, the word to predict and other option for the diplay and the saving of the figure. 
"""
import numpy as np 
import tensorflow_io as tfio
from scipy.io import wavfile
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import argparse


MOTS = ['yes','no','up','down','right','left','stop','go','on','off']

parser = argparse.ArgumentParser(description="Compare two spectrogram, one from the evaluation card and the other from a wav file",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-p", "--path",default= './src/phase1/modeles/mel-cnn-5', help="Path to the model to use")
parser.add_argument("-c","--carte",help="Path to the file with the spectogramm of the STM32" )
parser.add_argument("-a","--audio",help="Path the the wav file corresponding to the audio")
parser.add_argument("-w","--word",help="word that should be translate by the model")
parser.add_argument("-d","--display",default=False,help=" equal to True if you want print the figure, default value is False")
parser.add_argument("-s","--save",default=False,help=" equal to True if you want save the figure, default value is False")
parser.add_argument("-n","--name",default='comparaison_spectrogramme.png',help="name of the figure when it is saved")
args = vars(parser.parse_args())


CHEMIN_MODELE = args['path']

spectrogram_carte=args['carte']
spectrogram_audio=args['audio']

if (not spectrogram_carte) or (not spectrogram_audio):
    print("Error : there is a file missing.\nPlease indicate the spectogramme and the wav file using -c and -a.")
    exit()

mot=args['word']
display=args['display']
save=args['save']
name=args['name']




model = tf.keras.models.load_model(CHEMIN_MODELE)


### fichier spectogramme de la carte 
plt.subplot(2,1,1)  
taille=(63,128)
    
with open(spectrogram_carte, 'rb') as f:
    data = np.fromfile(f, dtype='<f')
    array = np.reshape(data,(63,128))

index_carte = np.argmax(model.predict(np.array([array])))

plt.imshow(array)
plt.ylabel("Time")
plt.xlabel("Frequency (db)")
plt.title('Spectrogramme microcontroleur, mot predit: '+MOTS[index_carte])
plt.colorbar()


### fichier .wav
samplerate, data = wavfile.read(spectrogram_audio)
rms_max = 0
debut = 0
for i in range(len(data)-16000):
    rms=np.mean((data[i:i+16000])**2)
    if rms > rms_max:
            rms_max = rms
            debut = i
enregistrement = data[debut:debut+16000]

scaled = np.float32(enregistrement)

# Ici, on réapplique le mêmes pré-traitements que pour les données d'entraînements.
fade = tfio.audio.fade(scaled, fade_in=1000, fade_out=2000, mode="logarithmic")
spectrogram = tfio.audio.spectrogram(fade, nfft=1024, window=1024, stride=256)
mel_spectrogram = tfio.audio.melscale(spectrogram, rate=16000, mels=128, fmin=0, fmax=8000)
test = np.array(tfio.audio.dbscale(mel_spectrogram, top_db=80))[...,np.newaxis]


index_tf = np.argmax(model.predict(np.array([test])))

plt.subplot(2,1,2) 
plt.imshow(test)
plt.ylabel("Time")
plt.xlabel("Frequency (db)")
plt.title('Spectrogramme Tensorflow, mot predit: '+MOTS[index_tf])
plt.colorbar()


plt.subplots_adjust(wspace = 0.5,hspace=0.5)
if mot:
    plt.suptitle("Spectogramme du mot '"+mot+"' par la carte et par tensorflow")
else:
    plt.suptitle("Spectogramme de la carte et de tensorflow")

if save:
    plt.savefig(name)

if display :
    plt.show()


