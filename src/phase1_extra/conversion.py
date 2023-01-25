import tensorflow as tf
import os

# Importation du modèle
converter = tf.lite.TFLiteConverter.from_saved_model('./src/phase1_extra/modeles_extra/mel-cnn-enhance-25')
chemin='./src/phase1_extra/lite/'
if not os.path.exists(chemin):
    os.makedirs(chemin)

# Conversion normale
tflite_dynamic = converter.convert()
converter.optimizations = [tf.lite.Optimize.DEFAULT]
with open(chemin+"dynamic.tflite", 'wb') as f:
    f.write(tflite_dynamic)

