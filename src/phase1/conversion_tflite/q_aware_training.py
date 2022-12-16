""""
Dans ce fichier, on prépare notre modèle à la quantization avant de le convertir
"""

# On commence par lancer créer et entrainer le modèe normalement, comme sur le fichier modele.py
# On notera simplement que l'on change la position des couches de batch_normalization pour qu'ils soient supportés
# par la suite
import numpy as np
import tensorflow as tf
import os

NB_MOTS = 10 # Changer en fonction du nombre de mots du corpus
CHEMIN_DONNEES= './donnees_traitees' # Dossier contenant les données pré-traitées
CHEMIN_SAUVEGARDE_MODELE = './modeles' # Dossier de sauvegarde des modèles entrainés
NOM_MODELE = 'mel-cnn-quant'


x_train, y_train = np.load(CHEMIN_DONNEES+'/train_data.npy')[...,np.newaxis], np.load(CHEMIN_DONNEES+'/train_labels.npy')
x_test, y_test = np.load(CHEMIN_DONNEES+'/test_data.npy')[...,np.newaxis], np.load(CHEMIN_DONNEES+'/test_labels.npy')


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, 3, strides=2, padding='same'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Conv2D(64, 3, padding='same'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Conv2D(128, 3, padding='same'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(NB_MOTS, activation='softmax'))

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.RMSprop(),
    metrics=['accuracy'],
)

history = model.fit(x_train, y_train, validation_split= 0.15, epochs=11)

model.evaluate(x_test, y_test)


# On prépare ensuite le modèle pour la quantization. Pour cela, nous devons recompiler le modèle avec la biblihotèque
# tensorflow-model-optimization

import tensorflow_model_optimization as tfmot

quantize_model = tfmot.quantization.keras.quantize_model
q_aware_model = quantize_model(model)

q_aware_model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.RMSprop(),
    metrics=['accuracy'],
)

q_aware_model.summary()

# Puis on réentraîne le modèle avec une quantité representative des données (fine-tuning)
x_train_subset = x_train[0:1000]
y_train_subset = y_train[0:1000]

q_aware_model.fit(x_train_subset, y_train_subset,
                  batch_size=500, epochs=1, validation_split=0.1)

# Nous pouvons évaluer la precision du nouveau modèle.
_, baseline_model_accuracy = model.evaluate(
    x_test, y_test, verbose=0)

_, q_aware_model_accuracy = q_aware_model.evaluate(
    x_test, y_test, verbose=0)

print('Baseline test accuracy:', baseline_model_accuracy)
print('Quant test accuracy:', q_aware_model_accuracy)

# Nous convertissons maintenant le modèle avec la quantization int8
converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

quantized_tflite_model = converter.convert()

# Nous préparons ensuite une fonction permettant d'évaluer le modèle converti (trouvée dans le site tensorflow)
import numpy as np

def evaluate_model(interpreter):
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    # Run predictions on every image in the "test" dataset.
    prediction_digits = []
    for i, test_spec in enumerate(x_test):
        if i % 1000 == 0:
            print('Evaluated on {n} results so far.'.format(n=i))
        # Pre-processing: add batch dimension and convert to float32 to match with
        # the model's input data format.
        test_spec = np.expand_dims(test_spec, axis=0).astype(np.float32)
        interpreter.set_tensor(input_index, test_spec)

        # Run inference.
        interpreter.invoke()

        # Post-processing: remove batch dimension and find the digit with highest
        # probability.
        output = interpreter.tensor(output_index)
        digit = np.argmax(output()[0])
        prediction_digits.append(digit)

    print('\n')
    # Compare prediction results with ground truth labels to calculate accuracy.
    prediction_digits = np.array(prediction_digits)
    accuracy = (prediction_digits == y_test).mean()
    return accuracy

# Et on compare la précision du modèle converti avec le modèle précédant
interpreter = tf.lite.Interpreter(model_content=quantized_tflite_model)
interpreter.allocate_tensors()

test_accuracy = evaluate_model(interpreter)

print('Quant TFLite test_accuracy:', test_accuracy)
print('Quant TF test accuracy:', q_aware_model_accuracy)
#%% Taille du modèle et sauvegarde
import tempfile

float_converter = tf.lite.TFLiteConverter.from_keras_model(model)
float_tflite_model = float_converter.convert()

_, float_file = tempfile.mkstemp('.tflite')
_, quant_file = tempfile.mkstemp('.tflite')

with open(quant_file, 'wb') as f:
    f.write(quantized_tflite_model)

with open(float_file, 'wb') as f:
    f.write(float_tflite_model)

print("Float model in Mb:", os.path.getsize(float_file) / float(2**20))
print("Quantized model in Mb:", os.path.getsize(quant_file) / float(2**20))

with open('./lite/quantized_aware.tflite', 'wb') as f:
    f.write(quantized_tflite_model)