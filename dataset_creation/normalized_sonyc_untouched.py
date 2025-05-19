import os
import librosa
import soundfile as sf

def normalize_audio(input_path, output_path):
    # Load the audio file
    audio, sr = librosa.load(input_path, sr=None)
    # Normalize the audio
    max_amplitude = max(abs(audio))
    if max_amplitude > 0:
        audio = audio / max_amplitude
    # Save the normalized audio
    sf.write(output_path, audio, sr)

def normalize_all_audios_in_folder(folder_path):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.wav'):  # Process only .wav files
                file_path = os.path.join(root, file)
                print(f"Normalizing: {file_path}")
                normalize_audio(file_path, file_path)  # Replace the file with normalized audio

if __name__ == "__main__":
    folder_path = "./dataset_creation/sonyc_untouched/"
    normalize_all_audios_in_folder(folder_path)