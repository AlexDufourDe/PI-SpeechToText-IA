"""
This file import the data for training and testing the model
"""

import pandas as pd
import os
import tensorflow as tf
import sys


def transorm_data(langue):
    """!import_data
    this fonction import the data and put them in the dataset form
    @para chemin_sauvegarde repository where the data will be save
    """

    if langue=="fr":
        # An integer scalar Tensor. The window length in samples.
        frame_length = 256
        # An integer scalar Tensor. The number of samples to step.
        frame_step = 160
        # An integer scalar Tensor. The size of the FFT to apply.
        # If not provided, uses the smallest power of 2 enclosing frame_length.
        fft_length=1024

        ##### IMPORT OF DATA
        
        data_path = "src/phase2/mozilla_common_voice"
        wavs_path = data_path
        metadata_path = data_path + "/metadata.csv"

    elif langue=="eng":
        # An integer scalar Tensor. The window length in samples.
        frame_length = 256
        # An integer scalar Tensor. The number of samples to step.
        frame_step = 160
        # An integer scalar Tensor. The size of the FFT to apply.
        # If not provided, uses the smallest power of 2 enclosing frame_length.
        fft_length = 384

        ##### IMPORT OF DATA
        data_url = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
        data_path = tf.keras.utils.get_file("LJSpeech-1.1", data_url, untar=True)
        wavs_path = data_path + "/wavs/"
        metadata_path = data_path + "/metadata.csv"
    else:
        print("error, we do not provide the data for this language")
    


    

    # Read metadata file and parse it
    metadata_df = pd.read_csv(metadata_path, sep="|", header=None, quoting=3)
    metadata_df.columns = ["file_name", "transcription", "normalized_transcription"]
    metadata_df = metadata_df[["file_name", "normalized_transcription"]]
    metadata_df = metadata_df.sample(frac=1).reset_index(drop=True)
    metadata_df.head(3)

    split = int(len(metadata_df) * 0.90)
    df_train = metadata_df[:split]
    df_val = metadata_df[split:]

    print(f"Size of the training set: {len(df_train)}")
    print(f"Size of the training set: {len(df_val)}")




    #### Aplphabet
    # The set of characters accepted in the transcription.
    characters = [x for x in "abcdefghijklmnopqrstuvwxyz'?! "]
    # Mapping characters to integers
    char_to_num = tf.keras.layers.StringLookup(vocabulary=characters, oov_token="")
    # Mapping integers back to original characters
    num_to_char = tf.keras.layers.StringLookup(
        vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
    )

    print(
        f"The vocabulary is: {char_to_num.get_vocabulary()} "
        f"(size ={char_to_num.vocabulary_size()})")


    ### pretraitement
        
    def encode_single_sample(wav_file,label):

    ###########################################
        ##  Process the Audio
        ##########################################
        # 1. Read wav file
        file = tf.io.read_file(wavs_path + wav_file + ".wav")
        # 2. Decode the wav file
        audio, _ = tf.audio.decode_wav(file)
        audio = tf.squeeze(audio, axis=-1)
        # 3. Change type to float
        audio = tf.cast(audio, tf.float32)
        # 4. Get the spectrogram
        spectrogram = tf.signal.stft(
            audio, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length
        )
        # 5. We only need the magnitude, which can be derived by applying tf.abs
        spectrogram = tf.abs(spectrogram)
        spectrogram = tf.math.pow(spectrogram, 0.5)
        # 6. normalisation
        means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
        stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
        spectrogram = (spectrogram - means) / (stddevs + 1e-10)


        ###########################################
        ##  Process the label
        ##########################################
        # 7. Convert label to Lower case
        label = tf.strings.lower(label)
        # 8. Split the label
        label = tf.strings.unicode_split(label, input_encoding="UTF-8")
        # 9. Map the characters in label to numbers
        label = char_to_num(label)
        # 10. Return a dict as our model is expecting two inputs

        
        return spectrogram, label
    
    batch_size = 32
    # Define the trainig dataset
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (list(df_train["file_name"]), list(df_train["normalized_transcription"]))
    )
    train_dataset = (
        train_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
        .padded_batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    # Define the validation dataset
    validation_dataset = tf.data.Dataset.from_tensor_slices(
        (list(df_val["file_name"]), list(df_val["normalized_transcription"]))
    )
    validation_dataset = (
        validation_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
        .padded_batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )


    return(train_dataset,validation_dataset)



if __name__ == "__main__":
    # total arguments
    n = len(sys.argv)
    if (n>2):
        print(f"Too many argument, expected :1 , got {n}")
    elif n==2:
       transorm_data(sys.argv[1])
    else:   
        transorm_data("src/phase2/LJSpeech")



