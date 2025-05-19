import pandas as pd
import random
import numpy as np
# Set random seed for reproducibility
random_seed = 0

# Load the CSV files
sonyc_df = pd.read_csv('./dataset_creation/sonyc-ust_labels.csv')
sonyc_df = sonyc_df[sonyc_df['group'] == 1]

librispeech_df = pd.read_csv('./dataset_creation/audio_lengths_librispeech.csv')

# Sort both dataframes by length
sonyc_df = sonyc_df.sort_values(by='length').reset_index(drop=True)
librispeech_df = librispeech_df.sort_values(by='length').reset_index(drop=True)

# Shuffle the librispeech_df randomly
librispeech_df = librispeech_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

# Create a list to store the correspondences
correspondences = []

count = 0
# Match files based on length (librispeech < sonyc)
used_librispeech_indices = set()
for sonyc_index, sonyc_row in sonyc_df.iterrows():
    sonyc_length = sonyc_row['length']
    closest_match = None
    closest_diff = float('inf')

    for librispeech_index, librispeech_row in librispeech_df.iterrows():
        if librispeech_index in used_librispeech_indices:
            continue

        librispeech_length = librispeech_row['length']
        if librispeech_length < sonyc_length:
            diff = sonyc_length - librispeech_length

            if diff < closest_diff:
                closest_diff = diff
                closest_match = librispeech_index

    if closest_match is not None:
        used_librispeech_indices.add(closest_match)
        correspondences.append({
            'sonyc_file': sonyc_row['fname'],
            'librispeech_file': librispeech_df.loc[closest_match, 'fname'],
            'length_difference': closest_diff
        })
        count += 1
        print(count)

# Convert correspondences to a DataFrame and save to a CSV
correspondences_df = pd.DataFrame(correspondences)
correspondences_df.to_csv('./dataset_creation/matched_audio_files.csv', index=False)

print("Matching complete. Results saved to 'matched_audio_files.csv'.")