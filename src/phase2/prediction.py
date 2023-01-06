""" 
This file is doing the prediction of the result using the model
"""
import keras
import numpy as np
import tensorflow as tf

def decode_batch_predictions(pred,num_to_char):
    """!decode_batch_prediction
    @param pred string
    @param num_to_char function
    """
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    # Iterate over the results and get back the text
    output_text = []
    for result in results:
        result = tf.strings.reduce_join(num_to_char(result)).numpy().decode("utf-8")
        output_text.append(result)
    return output_text