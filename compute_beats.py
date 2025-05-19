
import os 
import sys
import torch
import json

# Get the path to the parent directory containing "beats"
current_dir = os.path.dirname(os.path.abspath(__file__))
beats_parent_dir = os.path.join(current_dir, "beats")  # Adjust as per your directory structure
sys.path.insert(0, beats_parent_dir)

from BEATs import BEATs, BEATsConfig
import pandas as pd
import librosa
import numpy as np
import argparse

def get_all_wav_files(folder):
    wav_files = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(".wav"):
                wav_files.append(os.path.join(root, file))
    return wav_files

def main(input_dir, output_path):
    force_cpu = False
    device = torch.device("cpu" if force_cpu or not torch.cuda.is_available() else "cuda")

    # Load BEATs model
    checkpoint = torch.load(os.path.abspath("./beats/models/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"), map_location=device)
    cfg = BEATsConfig(checkpoint['cfg'])
    BEATs_model = BEATs(cfg)
    BEATs_model.load_state_dict(checkpoint['model'])
    BEATs_model.eval().to(device)

    # Load ontology
    with open("./utils/audioset_ontology.json", "r") as f:
        ontology_data = json.load(f)
    id_to_name = {entry["id"]: entry["name"] for entry in ontology_data}
    name_to_id = {entry["name"]: entry["id"] for entry in ontology_data}

    # Speech index
    speech_id = name_to_id['Speech']
    speech_index = next((key for key, value in checkpoint['label_dict'].items() if value == speech_id), None)

    # Load SONYC labels
    metadata = pd.read_csv("./dataset/metadata.csv")
    # Get unique values in "label_audioset" column
    audioset_labels = metadata['label1_audioset'].unique().tolist()

    wav_files = get_all_wav_files(input_dir)
    results = []

    for file_path in wav_files:
        fname = os.path.basename(file_path)
        sonyc_fname = fname.split("__")[1] if "__" in fname else fname

        # Load audio
        audio, sr = librosa.load(file_path, sr=16000)
        audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)

        # Get label
        file_label_row = metadata[metadata['sonyc_file'] == sonyc_fname]
        if file_label_row.empty:
            print(f"Label not found for file: {fname}")
            continue

        file_label = file_label_row['label1_audioset'].values[0]

        # Inference
        with torch.no_grad():
            padding_mask = torch.zeros(audio_tensor.shape, dtype=torch.bool, device=device)
            logits = BEATs_model.extract_features(audio_tensor, padding_mask=padding_mask)[0]

        speech_logit = logits[0, speech_index].item()

        file_results = {
            "fname": fname,
            "label1": "Speech" if "__" in fname else "",
            "label2": file_label,
            "Speech": speech_logit
        }

        for class_name in audioset_labels:
            label_id = name_to_id.get(class_name)
            label_index = next((key for key, value in checkpoint['label_dict'].items() if value == label_id), None)
            label_logit = logits[0, label_index].item() if label_index is not None else None
            file_results[class_name] = label_logit

        print(file_path)
        print(file_results)
        results.append(file_results)

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"Logits saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BEATs inference on .wav files")
    parser.add_argument("-i", "--input_path", required=True, help="Path to input directory containing .wav files")
    parser.add_argument("-o", "--output_path", required=True, help="Path to output CSV file")

    args = parser.parse_args()
    main(args.input_path, args.output_path)
