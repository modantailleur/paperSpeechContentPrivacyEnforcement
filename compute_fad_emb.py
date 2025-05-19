import argparse
import torch
from embs.model_loader import PANNsModel
from utils.fad import FrechetAudioDistance

def main(input_dir, output_dir, force_compute):
    force_cpu = False
    device = "cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")

    # Load the PANNs model
    panns_model = PANNsModel(variant="wavegram-logmel", device=device)
    panns_model.load_model()

    # Initialize Frechet Audio Distance object
    fad = FrechetAudioDistance(panns_model)

    # Compute embeddings
    fad.compute_embeddings(input_dir, output_dir, force_compute=force_compute)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute FAD embeddings using PANNs model")
    parser.add_argument("-i", "--input", required=True, help="Path to input directory containing audio files")
    parser.add_argument("-o", "--output", required=True, help="Path to output directory for embeddings")
    parser.add_argument("-f", "--force_compute", action="store_true", help="Force computation of embeddings (default: False)")
    args = parser.parse_args()
    main(args.input, args.output, args.force_compute)
