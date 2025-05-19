import os
import pandas as pd
import soundfile as sf

# Path to the directory containing the .flac files
base_path = "/media/user/MT-SSD-3/0-PROJETS_INFO/Thèse/librispeech/test-clean/LibriSpeech/test-clean"

# List to store file names and their lengths
data = []

# Traverse the directory and its subfolders
for root, _, files in os.walk(base_path):
    for file in files:
        if file.endswith(".flac"):
            file_path = os.path.join(root, file)
            # Read the audio file to get its length
            with sf.SoundFile(file_path) as audio_file:
                length_in_seconds = len(audio_file) / audio_file.samplerate
            # Append the file name and length to the list
            data.append([file, length_in_seconds])
            print(len(data))
# Create a DataFrame
df = pd.DataFrame(data, columns=["fname", "length"])

# Display the DataFrame
print(df)

# Optionally, save the DataFrame to a CSV file
df.to_csv("./dataset_creation/audio_lengths_librispeech.csv", index=False)