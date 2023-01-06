""" 
This file define the ctc loss for the model.compile
"""
import tensorflow as tf
import keras


def CTCLoss(y_true, y_pred):
    """!CTCLoss
    this function compute the loss of the algorithm ctc
    @param y_true int array
    @param y_pred int array

    @return loss float array
    """
    
    # Compute the training-time loss value
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss