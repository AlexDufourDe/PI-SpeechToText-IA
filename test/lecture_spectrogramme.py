""""
Ce fichier affiche le spectogramme fait sur la carte d'evaluation enregistrer en tant que fichier binaire.
Il prend différents paramètres en entrée tel que les chemins du fichier, le mot repésenté ainsi que des options pour la sauvegarde du spectogramme.

This file compare two spectogram, one made by the ealuation card and the other by a modul of tensorflow.
It also predict the meanning of the two represented words. Differents parameters can be passed to the program as the path to the
file,the model to use to predict the words, the word to predict and other option for the diplay and the saving of the figure. 
"""
import numpy as np 
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser(description="Display a spectrogram from binary text file",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-p", "--path", help="Path to the file to analyse")
parser.add_argument("-w","--word",help="word that should be translate by the model")
parser.add_argument("-s","--save",default=False,help=" equal to True if you want save the figure, default value is False")
parser.add_argument("-n","--name",default='comparaison_spectrogramme.png',help="name of the figure when it is saved")

args = vars(parser.parse_args())
file=args['path']
mot=args['word']
save=args['save']
name=args['name']


with open(file, 'rb') as f:
    data = np.fromfile(f, dtype='<f')
    array = np.reshape(data,(128,63))
transp=np.transpose(array)

plt.imshow(transp)
plt.ylabel("Time")
plt.xlabel("Hz")
plt.colorbar()
if mot:
    plt.title("Spectogramme du mot "+mot+"fais sur la carte d'évaluation")
else:
    plt.title("Spectogramme fais sur la carte d'évaluation")

if save:
    plt.savefig(name)

plt.show()
