import pandas as pd
import numpy as np
import os
import shutil
# Set the random seed for reproducibility
random_seed = 0

# Path to the CSV file
csv_file_path = "/media/user/MT-SSD-3/0-PROJETS_INFO/Thèse/SONYC-UST/annotations.csv"
audio_dir = "/media/user/MT-SSD-3/0-PROJETS_INFO/Thèse/SONYC-UST/audio"

# Define the directories
output_dir_untouched = "./dataset_creation/sonyc_untouched"
output_dir_mixed = "./dataset_creation/sonyc_mixed"

# Load the CSV file into a pandas DataFrame
df = pd.read_csv(csv_file_path)

# kept_columns = ['1_engine_presence', '2-2_jackhammer_presence', '4-1_chainsaw_presence', '5-1_car-horn_presence',
#                 '5-3_siren_presence', '6_music_presence', '8_dog_presence']

kept_columns = ['1_engine_presence', '2-2_jackhammer_presence', '5-1_car-horn_presence', '4-1_chainsaw_presence',
                '5-3_siren_presence', '6_music_presence', '7_human-voice_presence', '8_dog_presence']

# kept_columns_proximity = [col.replace("presence", "proximity") for col in kept_columns]

# # Filter rows where the corresponding proximity column has the value "near"
# for col, col_proximity in zip(kept_columns, kept_columns_proximity):
#     df = df[df[col_proximity] == "near"]

# Keep only the columns that are in kept_columns and all other columns that don't begin with a digit
df = df[[col for col in df.columns if col in kept_columns or not col[0].isdigit()]]

# Remove columns that begin with a digit followed by "-" and another letter or digit followed by "_"
# df = df[df.columns[~df.columns.str.match(r'^\d+-\w_')]]

# Group by 'audio_filename' and take the max of each column that begins with a digit + "_"
numeric_columns = [col for col in df.columns if col[0].isdigit()]
# Replace -1 with NaN to exclude them from the mean calculation
df[numeric_columns] = df[numeric_columns].replace(-1, np.nan)

# Group by 'audio_filename' and calculate the mean for each column, ignoring NaN values
df_grouped = df.groupby('audio_filename')[numeric_columns].mean(numeric_only=True).reset_index()
# df_grouped = df.groupby('audio_filename')[numeric_columns].max().reset_index()

# Keep only columns where there is exactly one '1' across all rows
df_grouped = df_grouped[df_grouped[numeric_columns].sum(axis=1) == 1]

# Create a new column "class" with the name of the column where there is a 1
df_grouped['label'] = df_grouped[numeric_columns].idxmax(axis=1)

# Drop rows where the column "label" is "7_human-voice_presence"
df_grouped = df_grouped[df_grouped['label'] != "7_human-voice_presence"]
# Remove the column "7_human-voice_presence" if it exists
if "7_human-voice_presence" in df_grouped.columns:
    df_grouped = df_grouped.drop(columns=["7_human-voice_presence"])
    
# Determine the minimum number of elements in any class
min_elements_per_class = df_grouped['label'].value_counts().apply(lambda x: x if x % 2 == 0 else x - 1).min()

# Filter df_grouped to keep the same number of elements for each class, chosen randomly
df_filtered = df_grouped.groupby('label').apply(lambda x: x.sample(n=min_elements_per_class, random_state=42)).reset_index(drop=True)

# Assign a "group" column to split the dataset into 2 groups with the same number of elements in each label
df_filtered['group'] = df_filtered.groupby('label').cumcount() % 2

df_filtered = df_filtered[['audio_filename', 'label', 'group']]
df_filtered["length"] = 10

# Rename the column 'audio_filename' to 'fname'
df_filtered.rename(columns={'audio_filename': 'fname'}, inplace=True)

print(df_filtered)

# Save the filtered DataFrame to a CSV file
output_csv_path = "./dataset_creation/sonyc-ust_labels.csv"
df_filtered.to_csv(output_csv_path, index=False)

# Create the output directories if they don't exist
os.makedirs(output_dir_untouched, exist_ok=True)
os.makedirs(output_dir_mixed, exist_ok=True)

# Copy the audio files to the respective folders based on the group
for _, row in df_filtered.iterrows():
    src_file = os.path.join(audio_dir, f"{row['fname']}")
    
    # Check if the source file exists
    if not os.path.exists(src_file):
        print(f"File {row['fname']}.wav does not exist in the directory")
        continue

    # Determine the destination file path
    if row['group'] == 0:
        dst_file = os.path.join(output_dir_untouched, f"{row['fname']}")
    else:
        dst_file = os.path.join(output_dir_mixed, f"{row['fname']}")
    
    # Copy the file
    shutil.copy(src_file, dst_file)
