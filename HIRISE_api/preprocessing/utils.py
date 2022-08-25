import pandas as pd


def read_encoded_csv(file_path, autoencoder = False):
  encoded_samples_df = pd.read_csv(file_path)
  encoded_samples_df.drop("Unnamed: 0",axis = 1, inplace = True)
  if autoencoder:
    labels_df = encoded_samples_df['label']
    encoded_samples_df.drop("label",axis = 1, inplace = True)
    return encoded_samples_df, labels_df
  else:
    return encoded_samples_df