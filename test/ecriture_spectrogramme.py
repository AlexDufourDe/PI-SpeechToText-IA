""""
Ce fichier enregitre le spectrogramme d'un fichier .wav sous la forme d'un fichier texte binaire.


This file save the spectogram of a wav file as a binary text file.
"""
import numpy as np 
import tensorflow_io as tfio
from scipy.io import wavfile
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser(description="Save a spectrogram from a wav file as a binary text file",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-p", "--path", help="Path to the file to transform")
args = vars(parser.parse_args())
file=args['path']


samplerate, data = wavfile.read(file)
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


name=file.split(".")
with open(name+'.txt', 'wb') as f:
    array=bytearray(test)
    f.write(array)



