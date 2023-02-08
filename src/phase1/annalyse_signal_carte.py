
import numpy as np 
import tensorflow_io as tfio
from scipy.io import wavfile
import matplotlib.pyplot as plt
import os
import tensorflow as tf


folder_path='./test_carte/'
CHEMIN_MODELE = './src/phase1/modeles/mel-cnn2'

MOTS = ['yes','no','up','down','right','left','stop','go','on','off']

nb_fichier=2

model = tf.keras.models.load_model(CHEMIN_MODELE)
i=1
count=0
ind=0


acc=[0 for i in range(nb_fichier+1)]
for path, dirs, files in os.walk(folder_path):
    k=len(files)
    c=1
    for filename in files:
        
        plt.subplot(nb_fichier,1,i)  
        ### fichier.txt
        taille=(63,128)
        ext=filename.split('.')
        if len(ext)==1:
            with open(folder_path+filename, 'rb') as f:
                data = np.fromfile(f, dtype='<f')
                array = np.reshape(data,(63,128))
            test=array
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


        index = np.argmax(model.predict(np.array([test])))
        print(f"\nLe mot retranscrit par {filename} est ---> {(MOTS[index])} <---")   
        plt.imshow(test)
        plt.ylabel("Time")
        plt.xlabel(filename)
        plt.colorbar()


        if MOTS[index]==MOTS[ind]:
            acc[i]+=1

        
        
        if i==nb_fichier:
            i=1
            print(f"\nLe mot attendu est {MOTS[ind]}\n")
            print('###############################################################')
            plt.subplots_adjust(wspace = 0.5,hspace=0.5)
            plt.show()

            
        else:
            i+=1
        
        count+=1
        if (count==nb_fichier*4):
            ind+=1
            count=0

print(f"accuracies finales:")
print(f"accuracy .wav : {2*acc[1]/k}")
for i in range(2,nb_fichier+1):
    print(f"accuracy fichier {i} : {2*acc[i]/k}")
        
       

