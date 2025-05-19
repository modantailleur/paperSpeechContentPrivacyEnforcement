import argparse
import os
import numpy as np
import pandas as pd
import librosa
import sys
import torch

# Add the parent directory of the project to the module search path
project_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_parent_dir)

from voice_metrics import WerMetricWhisper, WerMetricW2V2, WerMetricSPT, WerMetricSB  # Currently only Whisper supported

def main(audio_dir, output_file, asr_system, script_path):
    force_cpu = False
    device = torch.device("cpu") if force_cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if asr_system.lower() == "whisper":
        wer_metric = WerMetricWhisper(device=device)
    elif asr_system.lower() == "fairseq":
        wer_metric = WerMetricSPT(device=device)
    elif asr_system.lower() == "w2v2":
        wer_metric = WerMetricW2V2(device=device)
    elif asr_system.lower() == "crdnn":
        wer_metric = WerMetricSB(device=device, asr_short_name="cr")
    else:
        raise ValueError(f"Unsupported ASR system: {asr_system}. Only 'whisper' is currently supported.")

    # Load scripts
    script_df = pd.read_csv(script_path)
    script_df = script_df.dropna(subset=['librispeech_file'])
    script_df['librispeech_file'] = script_df['librispeech_file'].str.replace('.flac', '', regex=False)
    fname_to_script = dict(zip(script_df['librispeech_file'], script_df['script']))

    # Load matched audio file mappings
    csv_path = "./dataset/metadata.csv"
    matched_files = pd.read_csv(csv_path)
    matched_files = matched_files.dropna(subset=['librispeech_file'])
    sr = 44100
    output_data = []

    for _, row in matched_files.iterrows():
        mix_fname = f"{os.path.splitext(row['librispeech_file'])[0]}__{os.path.splitext(row['sonyc_file'])[0]}.wav"
        mix_path = os.path.join(audio_dir, mix_fname)

        if not os.path.exists(mix_path):
            print(f"File not found: {mix_path}")
            continue

        mixed_audio = librosa.load(mix_path, sr=sr)[0]
        mixed_audio_tensor = torch.tensor(mixed_audio).unsqueeze(0)

        try:
            mixed_transcript = wer_metric.get_transcript(mixed_audio_tensor, sr)
        except Exception as e:
            print(f"Transcription error on {mix_fname}: {e}")
            continue

        ref_key = os.path.splitext(row['librispeech_file'])[0]
        ref_transcript = fname_to_script.get(ref_key)

        if not ref_transcript:
            print(f"No reference transcript found for: {ref_key}")
            continue

        wer = wer_metric.get_wer(ref_transcript, mixed_transcript)

        print(f"WER for {mix_fname}: {wer:.3f}")
        output_data.append({
            "filename": mix_fname,
            "ref_transcript": ref_transcript,
            "predicted_transcript": mixed_transcript,
            "WER": wer
        })

    # Save results
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    pd.DataFrame(output_data).to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute WER for mixed audio files using Whisper ASR.")
    parser.add_argument("-i", "--input_audio_dir", required=True, help="Path to directory containing audio files")
    parser.add_argument("-o", "--output_file", required=True, help="Path to save output CSV file")
    parser.add_argument("-asr", "--asr_system", default="whisper", choices=["whisper", "fairseq", "w2v2", "crdnn"], help="ASR system to use (default: 'whisper')")
    parser.add_argument("-is", "--input_script", required=True, help="Path to CSV file containing reference transcripts")

    args = parser.parse_args()
    main(args.input_audio_dir, args.output_file, args.asr_system, args.input_script)
