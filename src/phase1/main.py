import os 
import tensorflow as tf
from scipy.io import wavfile

from model_fct import prediction
from affichage import affichage


folder_path='../../audio/'


CHEMIN_MODELE = './modeles/mel-cnn'  #Chemin du modèle que l'on souhaite tester
# Importation du mpdèle entrainé
model = tf.keras.models.load_model(CHEMIN_MODELE)
comp=[]
for path, dirs, files in os.walk(folder_path):
    for filename in files:
        
        #lecture du fichier
        samplerate, data = wavfile.read(path+"/"+filename)

        #prediction du model
        pred=prediction(data,model)

        #remplissage de la table de comparaison
        mot_original=filename.split("_")[0]

        comp.append((mot_original,pred,path.split('/')[-1]))
    
affichage(comp)




