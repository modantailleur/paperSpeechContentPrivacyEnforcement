from abc import ABC, abstractmethod
import logging
from typing import Literal
import numpy as np
import soundfile
import os
import sys
from urllib.parse import unquote

import torch
from torch import nn
from pathlib import Path
import importlib.util
import importlib.metadata

# Add the root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pann import models as panns

log = logging.getLogger(__name__)


class ModelLoader(ABC):
    """
    Abstract class for loading a model and getting embeddings from it. The model should be loaded in the `load_model` method.
    """
    def __init__(self, name: str, num_features: int, sr: int, min_len: int = -1, device: str = torch.device("cpu")):
        """
        Args:
            name (str): A unique identifier for the model.
            num_features (int): Number of features in the output embedding (dimensionality).
            sr (int): Sample rate of the audio.
            min_len (int, optional): Enforce a minimal length for the audio in seconds. Defaults to -1 (no minimum).
        """
        self.model = None
        self.sr = sr
        self.num_features = num_features
        self.name = name
        self.min_len = min_len
        # self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.device = device

    def get_embedding(self, audio: np.ndarray):
        embd = self._get_embedding(audio)
        if self.device == torch.device('cuda'):
            embd = embd.cpu()
        embd = embd.cpu().detach().numpy()
        
        # If embedding is float32, convert to float16 to be space-efficient
        if embd.dtype == np.float32:
            embd = embd.astype(np.float16)

        return embd

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def _get_embedding(self, audio: np.ndarray):
        """
        Returns the embedding of the audio file. The resulting vector should be of shape (n_frames, n_features).
        """
        pass

    def load_wav(self, wav_file: Path):
        wav_data, _ = soundfile.read(wav_file, dtype='int16')
        wav_data = wav_data / 32768.0  # Convert to [-1.0, +1.0]
        
        # Enforce minimum length
        wav_data = self.enforce_min_len(wav_data)

        return wav_data
    
    def enforce_min_len(self, audio: np.ndarray) -> np.ndarray:
        """
        Enforce a minimum length for the audio. If the audio is too short, output a warning and pad it with zeros.
        """
        if self.min_len < 0:
            return audio
        if audio.shape[0] < self.min_len * self.sr:
            log.warning(
                f"Audio is too short for {self.name}.\n"
                f"The model requires a minimum length of {self.min_len}s, audio is {audio.shape[0] / self.sr:.2f}s.\n"
                f"Padding with zeros."
            )
            audio = np.pad(audio, (0, int(np.ceil(self.min_len * self.sr - audio.shape[0]))))
            print()
        return audio

class PANNsModel(ModelLoader):
    """
    Kong, Qiuqiang, et al., "Panns: Large-scale pretrained audio neural networks for audio pattern recognition.",
    IEEE/ACM Transactions on Audio, Speech, and Language Processing 28 (2020): 2880-2894.

    Specify the model to use (cnn14-32k, cnn14-16k, wavegram-logmel).
    You can also specify wether to send the full provided audio or 1-s chunks of audio (cnn14-32k-1s). This was shown 
    to have a very low impact on performances.
    """
    def __init__(self, variant: Literal['cnn14-32k', 'wavegram-logmel'], device: str = "cpu", emb_ckpt_dir: str = None):
        super().__init__(f"panns-{variant}", 2048, 
                         sr=16000 if variant == 'cnn14-16k' else 32000, device=device)
        self.variant = variant
        if emb_ckpt_dir is None:
            self.emb_ckpt_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), ".model-checkpoints/")
        else:
            self.emb_ckpt_dir = os.path(emb_ckpt_dir)
            
    def load_model(self):
        # current_file_dir = os.path.dirname(os.path.realpath(__file__))
        # ckpt_dir = os.path.join(current_file_dir, ".model-checkpoints/")
        os.makedirs(self.emb_ckpt_dir, exist_ok=True)

        # Mapping variants to checkpoint files
        ckpt_urls = {
            'cnn14-16k': "https://zenodo.org/record/3987831/files/Cnn14_16k_mAP%3D0.438.pth",
            'cnn14-32k': "https://zenodo.org/record/3576403/files/Cnn14_mAP%3D0.431.pth",
            'wavegram-logmel': "https://zenodo.org/records/3987831/files/Wavegram_Logmel_Cnn14_mAP%3D0.439.pth"
        }
        ckpt_files = {key: unquote(url.split('/')[-1]) for key, url in ckpt_urls.items()}

        # Check and download the specific checkpoint file if not present
        ckpt_file = ckpt_files.get(self.variant, None)

        if ckpt_file:
            ckpt_path = os.path.join(self.emb_ckpt_dir, ckpt_file)
            if not os.path.exists(ckpt_path):
                print(f"Downloading checkpoint for {self.variant}...")
                os.system(f"wget -P {self.emb_ckpt_dir} {ckpt_urls[self.variant]}")

        features_list = ["2048", "logits"]

        # Load the corresponding model and checkpoint
        if self.variant == 'cnn14-16k':
            self.model = panns.Cnn14(
                features_list=features_list,
                sample_rate=16000,
                window_size=512,
                hop_size=160,
                mel_bins=64,
                fmin=50,
                fmax=8000,
                classes_num=527,
            )
            state_dict = torch.load(os.path.join(self.emb_ckpt_dir, "Cnn14_16k_mAP=0.438.pth"), map_location=self.device)
            self.model.load_state_dict(state_dict["model"])

        elif self.variant == 'cnn14-32k':
            self.model = panns.Cnn14(
                features_list=features_list,
                sample_rate=32000,
                window_size=1024,
                hop_size=320,
                mel_bins=64,
                fmin=50,
                fmax=14000,
                classes_num=527,
            )
            state_dict = torch.load(os.path.join(self.emb_ckpt_dir, "Cnn14_mAP=0.431.pth"), map_location=self.device)
            self.model.load_state_dict(state_dict["model"])

        elif 'wavegram-logmel' in self.variant:
            self.model = panns.Wavegram_Logmel_Cnn14(
                sample_rate=32000,
                window_size=1024,
                hop_size=320,
                mel_bins=64,
                fmin=50,
                fmax=14000,
                classes_num=527,
            )
            state_dict = torch.load(os.path.join(self.emb_ckpt_dir, "Wavegram_Logmel_Cnn14_mAP=0.439.pth"), map_location=self.device)
            self.model.load_state_dict(state_dict["model"])

        self.model.eval()
        self.model.to(self.device)

        self.num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'PANNs: {self.num_params}')

    def _get_embedding(self, audio: np.ndarray) -> np.ndarray:
        
        # The int16-float32 conversion is used for quantization
        audio = self.int16_to_float32(self.float32_to_int16(audio))

        # Split the audio into 10s chunks with 1s hop
        chunk_size = 10 * self.sr  # 10 seconds
        hop_size = self.sr  # 1 second
        chunks = [audio[:, i:i+chunk_size] for i in range(0, audio.shape[1], hop_size)]

        # Calculate embeddings for each chunk
        embeddings = []
        emb_str = '2048' if 'cnn14' in self.variant else 'embedding'
        for chunk in chunks:
            with torch.no_grad():
                chunk = chunk if chunk.shape[1] == chunk_size else np.pad(chunk, ((0,0), (0, chunk_size-chunk.shape[1])))
                chunk = torch.from_numpy(chunk).float().to(self.device)
                emb = self.model.forward(chunk)[emb_str]
                embeddings.append(emb)

        emb = torch.stack(embeddings, dim=0)
        emb = torch.mean(emb, dim=0)

        return emb

    def int16_to_float32(self, x):
        return (x / 32767.0).astype(np.float32)

    def float32_to_int16(self, x):
        x = np.clip(x, a_min=-1., a_max=1.)
        return (x * 32767.).astype(np.int16)