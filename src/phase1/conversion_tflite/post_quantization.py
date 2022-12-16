"""
Conversion d'un modèle enregsitré Tensorflow en un modèle .tflite.
Lors de la conversion, il est possible d'effectuer une "quantization", qui consiste à réduire la précision des chiffres
pour avoir une taille réduite et un calcul plus rapide.
"""

import tensorflow as tf

# Importation du modèle
converter = tf.lite.TFLiteConverter.from_saved_model('./modeles/mel-cnn2')

# Conversion normale
tflite_model = converter.convert()
with open('./lite/test.tflite', 'wb') as f:
    f.write(tflite_model)


# Conversion avec "post-training float16 quantization"
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_quant = converter.convert()
with open("./lite/post_quant.tflite", 'wb') as f:
    f.write(tflite_quant)