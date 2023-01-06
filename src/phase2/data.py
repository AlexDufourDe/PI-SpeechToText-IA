"""
This file import the data for training and testing the model
"""
import keras
import pandas as pd


def import_data():
    """!import_data
    thid fonction import the data and put them in the dataset form
    @return df_train training dataset
    @return df_val testing dataset
    """

    data_url = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
    data_path = keras.utils.get_file("LJSpeech-1.1", data_url, untar=True)
    wavs_path = data_path + "/wavs/"
    metadata_path = data_path + "/metadata.csv"

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
    return(df_train,df_val)

