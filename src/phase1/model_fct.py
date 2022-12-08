import numpy as np
import tensorflow_io as tfio
import wave
from scipy.io import wavfile

MOTS = ['yes','no','up','down','right','left','stop','go','on','off']

def prediction(data,model):
    scaled = np.int16(data* 32767).reshape(16000) # Remise à l'echelle de l'audio
        
    # Ici, on réapplique le mêmes pré-traitements que pour les données d'entraînements.
    fade = tfio.audio.fade(scaled, fade_in=1000, fade_out=2000, mode="logarithmic")
    spectrogram = tfio.audio.spectrogram(fade, nfft=1024, window=1024, stride=256)
    mel_spectrogram = tfio.audio.melscale(spectrogram, rate=16000, mels=128, fmin=0, fmax=8000)
    test = np.array(tfio.audio.dbscale(mel_spectrogram, top_db=80))[...,np.newaxis]

    index = np.argmax(model.predict(np.array([test])))

    #print("\nLe mot retranscrit est ---> " + (MOTS[index]).upper() + " <---")
    return((MOTS[index]).upper())