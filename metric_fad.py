from embs.model_loader import PANNsModel
from utils.fad import FrechetAudioDistance
import os
import librosa
import numpy as np
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
# Load the PANNs model
panns_model = PANNsModel(variant="wavegram-logmel", device=device)
panns_model.load_model()

fad = FrechetAudioDistance(panns_model)

evaluations = [
    ['oracle', 'noise'],
    ['cohen', 'burkhardt', 'ours_envss', 'ours'],
    ['ours', 'ours_novad', 'ours_noss'],
    ['ours', 'ours_with_mixframe'],
]

for mname_list in evaluations:
    print(f"\nEvaluating methods: {mname_list}")
    for name in mname_list:
        # Directory containing the audio files
        baseline_emb_dir = "./output/oracle/embeddings/"
        eval_emb_dir = f"./output/{name}/embeddings/"
        fad_score = fad.score(baseline_emb_dir, eval_emb_dir)

        print(f"FAD score for {name}: {fad_score}")
