import pandas as pd

# Load the CSV files into DataFrames
sonyc_labels_df = pd.read_csv('./dataset_creation/sonyc-ust_labels.csv')
matched_audio_files_df = pd.read_csv('./dataset_creation/matched_audio_files.csv')
librispeech_scripts_df = pd.read_csv('./dataset_creation/librispeech_scripts.csv')
labelmap_df = pd.read_excel('./dataset_creation/labelmap_sonyc_audioset.xlsx')

# Rename the 'fname' column in sonyc_labels_df to 'sonyc_file'
sonyc_labels_df.rename(columns={'fname': 'sonyc_file'}, inplace=True)

# Merge matched_audio_files_df with librispeech_scripts_df on 'librispeech_file'
merged_df = matched_audio_files_df.merge(librispeech_scripts_df, on='librispeech_file', how='left')

# Merge the result with sonyc_labels_df on 'sonyc_file' and 'fname'
final_df = merged_df.merge(sonyc_labels_df, on='sonyc_file', how='left')

# Filter rows from sonyc_labels_df where group is 0
group_0_df = sonyc_labels_df[sonyc_labels_df['group'] == 0]

# Select only the columns 'sonyc_file', 'label', and 'group', and add NaN for other columns
group_0_df = group_0_df[['sonyc_file', 'label', 'group']]
for col in final_df.columns:
    if col not in ['sonyc_file', 'label', 'group']:
        group_0_df[col] = pd.NA

# Concatenate the group_0_df with the final_df
final_df = pd.concat([final_df, group_0_df], ignore_index=True)

# Remove 'length' and 'length_difference' columns if they exist
final_df = final_df.drop(columns=['length', 'length_difference', 'group'], errors='ignore')

# Map the 'label' column to 'label_audioset' using the labelmap DataFrame
labelmap_dict = dict(zip(labelmap_df['labels_sonyc'], labelmap_df['labels_audioset']))
final_df['label_audioset'] = final_df['label'].map(labelmap_dict)

# Create the 'fname' column by combining 'librispeech_file' and 'sonyc_file'
final_df['fname'] = final_df.apply(
    lambda row: row['librispeech_file'].replace('.flac', '') + '__' + row['sonyc_file']
    if pd.notna(row['librispeech_file']) else row['sonyc_file'], axis=1
)

# Reorder columns to place 'fname' as the first column
columns_order = ['fname'] + [col for col in final_df.columns if col != 'fname']
final_df = final_df[columns_order]

# Rename the 'label' column to 'label1_sonyc' and 'label_audioset' column to 'label1_audioset'
final_df.rename(columns={'label': 'label1_sonyc', 'label_audioset': 'label1_audioset'}, inplace=True)

# Add values to 'label2_sonyc' and 'label2_audioset' for rows with a value in 'librispeech_file'
final_df['label2_sonyc'] = final_df.apply(
    lambda row: '7-1_person-or-small-group-talking_presence' if pd.notna(row['librispeech_file']) else float('nan'), axis=1
)
final_df['label2_audioset'] = final_df.apply(
    lambda row: 'Speech' if pd.notna(row['librispeech_file']) else float('nan'), axis=1
)

print(final_df)
# Save the final DataFrame to a CSV file
final_df.to_csv('./dataset_creation/metadata.csv', index=False, na_rep='NaN')