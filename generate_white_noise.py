import os
import numpy as np
import soundfile as sf

# Paths
output_dir = "./output/noise"
dataset_dirs = {
    "sonyc_librispeech_mixed_anonymized": "./dataset/sonyc_librispeech_mixed",
    "sonyc_raw_anonymized": "./dataset/sonyc_raw"
}

# Create the output noise folder if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Iterate over dataset directories
for folder_name, dataset_path in dataset_dirs.items():
    # Create subfolder in the output directory
    output_subfolder = os.path.join(output_dir, folder_name)
    os.makedirs(output_subfolder, exist_ok=True)

    # Get the number of elements in the dataset folder
    if os.path.exists(dataset_path):
        dataset_files = os.listdir(dataset_path)
        for file_name in dataset_files:
            # Generate white noise
            white_noise = np.random.normal(0, 1, 44100*10)  # Example: 1 second of white noise at 16kHz

            # Save the white noise to a file in the output folder
            output_file_path = os.path.join(output_subfolder, file_name)
            sf.write(output_file_path, white_noise, 44100)
    else:
        print(f"Dataset path does not exist: {dataset_path}")