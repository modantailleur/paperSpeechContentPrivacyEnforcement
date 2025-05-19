import pandas as pd
import librosa
import numpy as np
import soundfile as sf
import os

def normalize_audio(audio):
    """Normalize the audio to -1 dBFS."""
    peak = np.max(np.abs(audio))
    return audio / peak

def calculate_rms(audio):
    """Calculate the Root Mean Square (RMS) of an audio signal."""
    return np.sqrt(np.mean(audio**2))

def mix_audio_files(csv_file, output_dir, db_offset=0):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    for index, row in df.iterrows():
        file1 = row['librispeech_file']
        file2 = row['sonyc_file']
        
        sr = 44100  # Sample rate
        # Load the audio files
        audio1, sr1 = librosa.load(f"{librispeech_dir}/{file1}", sr=sr)
        audio2, sr2 = librosa.load(f"{fsd50k_dir}/{file2}", sr=sr)
        
        # Ensure both audio files have the same sample rate
        if sr1 != sr2:
            raise ValueError("Sample rates of the two audio files must match.")
        
        # Normalize both audio files
        audio1 = normalize_audio(audio1)
        audio2 = normalize_audio(audio2)
        
        # Pad the smaller audio with zeros to match the length of the larger audio
        max_length = max(len(audio1), len(audio2))
        audio1 = np.pad(audio1, (0, max_length - len(audio1)), mode='constant')
        audio2 = np.pad(audio2, (0, max_length - len(audio2)), mode='constant')
        
        # Adjust the background audio (audio2) to match the RMS of the voice audio (audio1) for SNR=0
        rms_voice = calculate_rms(audio1)
        rms_background = calculate_rms(audio2)
        scaling_factor = rms_voice / rms_background
        audio2 = audio2 * scaling_factor
        
        # Apply db_offset to reduce the background sound
        audio2 = audio2 * (10 ** (-db_offset / 20))
        
        # Mix the audio files
        mixed_audio = audio1 + audio2
        
        # Normalize the mixed audio
        mixed_audio = normalize_audio(mixed_audio)
        
        # Remove file extensions from file1 and file2
        file1 = file1.rsplit('.', 1)[0]
        file2 = file2.rsplit('.', 1)[0]
        
        # Save the mixed audio
        output_file = f"{output_dir}/{file1}__{file2}.wav"
        sf.write(output_file, mixed_audio, sr1)
        print(f"Mixed audio saved to {output_file}")

# Dirs
fsd50k_dir = './dataset_creation/sonyc_mixed'
librispeech_dir = './dataset_creation/librispeech_mixed'

# Example usage
csv_file = "./dataset_creation/matched_audio_files.csv"
output_dir = "./dataset_creation/sonyc_librispeech_mixed"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

mix_audio_files(csv_file, output_dir, db_offset=6)