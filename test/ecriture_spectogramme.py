
import numpy as np 
import tensorflow_io as tfio
from scipy.io import wavfile
import matplotlib.pyplot as plt



folder_path='./audio/bien/'



import os 
# Importation du mpdèle entrainé

nom=[]
for path, dirs, files in os.walk(folder_path):
    for filename in files:
        samplerate, data = wavfile.read(folder_path+filename)
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

        nn=filename.split(".")[0]

        with open('./audio/spectro/'+nn+'.txt', 'wb') as f:
            array=bytearray(test)
            f.write(array)
        nom.append(filename)

print(nom)
