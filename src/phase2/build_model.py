"""
this file build the model
"""
import tensorflow as tf

def build_model(input_dim, output_dim,CTCLoss,char_to_num, rnn_layers=5, rnn_units=128):
    """!build_model
    This function create the model by adding the differrent layers compile it with the indicated loss

    @param input dim int
    @param output_dim int
    @param CTCLoss funtion
    @param char_to_num function
    @param rrn_units int optionnal

    @return model
    """
   
    original=0


    # Model's input
    model = tf.keras.models.Sequential()
   

    if (original ==1):
      model.add(tf.keras.layers.Reshape((-1, input_dim, 1), name="expand_dim"))
      # Couche 1
      model.add(tf.keras.layers.Conv2D(32,  kernel_size=[11, 41], strides=[2, 2], padding='same'))
      model.add(tf.keras.layers.BatchNormalization())
      model.add(tf.keras.layers.ReLU())

      # Couche 2
      model.add(tf.keras.layers.Conv2D(32,  kernel_size=[11, 21], strides=[2, 2], padding='same'))
      model.add(tf.keras.layers.BatchNormalization())
      model.add(tf.keras.layers.ReLU())

      model.add(tf.keras.layers.Reshape((-1, 1568)))

      # Couche 3
      model.add(tf.keras.layers.GRU(units = rnn_units, activation="tanh", recurrent_activation="sigmoid",
                                    use_bias=True,return_sequences=True, reset_after=True))
      # Couche 4
      model.add(tf.keras.layers.GRU(units = rnn_units, activation="tanh", recurrent_activation="sigmoid",
                                    use_bias=True,return_sequences=True, reset_after=True))
      
      # Couche 5
      model.add(tf.keras.layers.GRU(units = rnn_units, activation="tanh", recurrent_activation="sigmoid",
                                    use_bias=True,return_sequences=True, reset_after=True))
      # Couche 6
      model.add(tf.keras.layers.Dense(units=output_dim + 1, activation="softmax"))

    else:

      # Couche 1
      model.add(tf.keras.layers.Dense(units=64 , activation="relu"))
      # Couche 2
      model.add(tf.keras.layers.Dense(units=64 , activation="relu"))
      # Couche 2
      model.add(tf.keras.layers.Dense(units=64 , activation="relu"))

      # Couche 4
      model.add(tf.keras.layers.GRU(units = rnn_units, activation="tanh", recurrent_activation="sigmoid",
                                    use_bias=True,return_sequences=True, reset_after=True))
      
      # Couche 5
      model.add(tf.keras.layers.Dense(units=64 , activation="relu"))

      # Couche 6
      model.add(tf.keras.layers.Dense(units=char_to_num.vocabulary_size()+1 , activation="softmax"))



    model.compile(loss=CTCLoss)
    return model