import numpy as np
import keras
import tensorflow as tf
from jiwer import wer
from prediction import decode_batch_predictions

# A callback class to output a few transcriptions during training
class CallbackEval(keras.callbacks.Callback):
    """Displays a batch of outputs after every epoch."""

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def on_epoch_end(self,model, epoch: int, logs=None):
        predictions = []
        targets = []
        
        # The set of characters accepted in the transcription.
        characters = [x for x in "abcdefghijklmnopqrstuvwxyz'?! "]
        # Mapping characters to integers
        char_to_num = keras.layers.StringLookup(vocabulary=characters, oov_token="")
        # Mapping integers back to original characters
        num_to_char = keras.layers.StringLookup(
            vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
        )
        for batch in self.dataset:
            X, y = batch
            batch_predictions = model.predict(X)
            batch_predictions = decode_batch_predictions(batch_predictions)
            predictions.extend(batch_predictions)
            for label in y:
                label = (
                    tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
                )
                targets.append(label)
        wer_score = wer(targets, predictions)
        print("-" * 100)
        print(f"Word Error Rate: {wer_score:.4f}")
        print("-" * 100)
        
        for i in np.random.randint(0, len(predictions), 2):
            print(f"Target    : {targets[i]}")
            print(f"Prediction: {predictions[i]}")
            print("-" * 100)