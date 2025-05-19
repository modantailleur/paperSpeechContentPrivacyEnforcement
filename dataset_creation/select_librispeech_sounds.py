import pandas as pd
import os
import shutil

def load_script_txt(txt_file_path):
    """
    Loads a space-separated script file into a pandas DataFrame.
    Assumes each line starts with a filename followed by a space and a script.

    Parameters:
        txt_file_path (str): Path to the .txt file.

    Returns:
        pd.DataFrame: A DataFrame with columns ["filename", "script"].
    """
    data = []
    with open(txt_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(" ", 1)  # split only on the first space
            if len(parts) == 2:
                filename, script = parts
                filename += ".flac"  # Add .flac to the filename
                data.append((filename, script))

    return pd.DataFrame(data, columns=["fname", "script"])

# Load the CSV file into a pandas DataFrame
file_path = "./dataset_creation/matched_audio_files.csv"
data = pd.read_csv(file_path)

# Define the base directory to search for files
base_dir = "/media/user/MT-SSD-3/0-PROJETS_INFO/Thèse/librispeech/test-clean/LibriSpeech/test-clean"

# Copy the file to the target directory
target_dir = "./dataset_creation/librispeech_mixed/"

# Initialize a list to store the selected text content
selected_texts = []

# Iterate over each file name in the "librispeech_fname" column
for fname in data["librispeech_file"]:
    # Search for the file in the directory and subdirectories
    for root, _, files in os.walk(base_dir):
        if fname in files:
            # Get the subfolder containing the file
            subfolder = os.path.dirname(os.path.join(root, fname))
            
            # Find the unique .txt file in the same subfolder
            txt_files = [f for f in os.listdir(subfolder) if f.endswith(".txt")]

            if len(txt_files) == 1:
                txt_file_path = os.path.join(subfolder, txt_files[0])
                # Open the .txt file with pandas
                txt_df = load_script_txt(txt_file_path)

                # Select the second column for the row matching the fname
                matching_row = txt_df[txt_df["fname"] == fname]
                if not matching_row.empty:
                    selected_texts.append(matching_row["script"].iloc[0])
                    os.makedirs(target_dir, exist_ok=True)
                    target_path = os.path.join(target_dir, fname)
                    shutil.copy2(os.path.join(root, fname), target_path)
            break

print(selected_texts)
# Add the selected texts as a new column to the DataFrame
data["script"] = selected_texts
# Remove the specified columns from the DataFrame
data = data.drop(columns=["sonyc_file", "length_difference"], errors="ignore")

# Save the updated DataFrame to a CSV file
data.to_csv(os.path.join(target_dir, "librispeech_scripts.csv"), index=False)