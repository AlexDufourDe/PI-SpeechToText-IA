
import numpy as np 
import tensorflow_io as tfio
from scipy.io import wavfile
import matplotlib.pyplot as plt
import os 

folder_path='./test_carte/'
# folder_path='./audio/bien/'
i=1
for path, dirs, files in os.walk(folder_path):
    k=len(files)
    c=1
    for filename in files:
        
        plt.subplot(k,1,i)  
        ### fichier.txt
        taille=(63,128)
        ext=filename.split('.')[1]
        if ext=='TXT':
            with open(folder_path+filename, 'rb') as f:
                data = np.fromfile(f, dtype='<f')
                array = np.reshape(data,(128,63))
            transp=np.transpose(array)
            np.save('./test_carte/'+filename.split('.')[0]+'.npy',transp)

            plt.imshow(transp)
            plt.ylabel("Time")
            plt.xlabel("Hz")
            plt.colorbar()
        else:
            ### fichier .wav
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
            
            plt.imshow(test)
            plt.ylabel("Time")
            plt.xlabel("Hz")
            plt.colorbar()

        i+=1
plt.subplots_adjust(wspace = 0.5,hspace=0.5)
plt.show()
