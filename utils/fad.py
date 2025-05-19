import os
import shutil
import subprocess
import tempfile
import traceback
from typing import NamedTuple, Union
import numpy as np
import torch
import torchaudio
from scipy import linalg
from scipy import sqrt as scisqrt
from pathlib import Path
from hypy_utils import write
from hypy_utils.tqdm_utils import tq, tmap
from hypy_utils.logging_utils import setup_logger
from hypy_utils.tqdm_utils import pmap
import librosa

def _process_file(file):
    embd = np.load(file)
    n = embd.shape[0]
    return np.mean(embd, axis=0), np.cov(embd, rowvar=False) * (n - 1), n

def calculate_embd_statistics_online(files):
    """
    Calculate the mean and covariance matrix of a list of embeddings in an online manner.

    :param files: A list of npy files containing ndarrays with shape (n_frames, n_features)
    """
    assert len(files) > 0, "No files provided"

    # Load the first file to get the embedding dimension
    embd_dim = np.load(files[0]).shape[-1]

    # Initialize the mean and covariance matrix
    mu = np.zeros(embd_dim)
    S = np.zeros((embd_dim, embd_dim))  # Sum of squares for online covariance computation
    n = 0  # Counter for total number of frames

    results = pmap(_process_file, files, desc='Calculating statistics')
    for _mu, _S, _n in results:
        delta = _mu - mu
        mu += _n / (n + _n) * delta
        S += _S + delta[:, None] * delta[None, :] * n * _n / (n + _n)
        n += _n

    if n < 2:
        return mu, np.zeros_like(S)
    else:
        cov = S / (n - 1)  # compute the covariance matrix
        return mu, cov

class FADInfResults(NamedTuple):
    score: float
    slope: float
    r2: float
    points: list[tuple[int, float]]


def calc_embd_statistics(embd_lst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the mean and covariance matrix of a list of embeddings.
    If the input has only one sample, return a zero covariance matrix.
    """
    mean = np.mean(embd_lst, axis=0)
    if embd_lst.shape[0] == 1:
        cov = np.zeros((embd_lst.shape[1], embd_lst.shape[1]))
    else:
        cov = np.cov(embd_lst, rowvar=False)
    return mean, cov

def calc_frechet_distance(mu1, cov1, mu2, cov2, eps=1e-6):
    """
    Adapted from: https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py
    
    Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
            inception net (like returned by the function 'get_predictions')
            for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
            representative data set.
    -- cov1: The covariance matrix over activations for generated samples.
    -- cov2: The covariance matrix over activations, precalculated on an
            representative data set.
    Returns:
    --   : The Frechet Distance.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    cov1 = np.atleast_2d(cov1)
    cov2 = np.atleast_2d(cov2)

    assert mu1.shape == mu2.shape, \
        f'Training and test mean vectors have different lengths ({mu1.shape} vs {mu2.shape})'
    assert cov1.shape == cov2.shape, \
        f'Training and test covariances have different dimensions ({cov1.shape} vs {cov2.shape})'

    diff = mu1 - mu2

    # Product might be almost singular
    # NOTE: issues with sqrtm for newer scipy versions
    # using eigenvalue method as workaround
    covmean_sqrtm, _ = linalg.sqrtm(cov1.dot(cov2), disp=False)
    
    # eigenvalue method
    D, V = linalg.eig(cov1.dot(cov2))
    covmean = (V * scisqrt(D)) @ linalg.inv(V)

    if not np.isfinite(covmean).all():
        offset = np.eye(cov1.shape[0]) * eps
        covmean = linalg.sqrtm((cov1 + offset).dot(cov2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    tr_covmean_sqrtm = np.trace(covmean_sqrtm)
    if np.iscomplexobj(tr_covmean_sqrtm):
        if np.abs(tr_covmean_sqrtm.imag) < 1e-3:
            tr_covmean_sqrtm = tr_covmean_sqrtm.real

    if not(np.iscomplexobj(tr_covmean_sqrtm)):
        delt = np.abs(tr_covmean - tr_covmean_sqrtm)

    return (diff.dot(diff) + np.trace(cov1)
            + np.trace(cov2) - 2 * tr_covmean)


class FrechetAudioDistance:
    def __init__(self, ml, device= torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):
        self.ml = ml
        self.device = device

    def compute_embeddings(self, audio_dir, emb_dir, force_compute=False):
        os.makedirs(emb_dir, exist_ok=True)
        # Walk through the directory and its subdirectories
        for root, _, files in os.walk(audio_dir):
            for file_name in files:
                if file_name.endswith(".wav"):
                    file_path = os.path.join(root, file_name)
                    
                    if not force_compute and os.path.exists(os.path.join(emb_dir, os.path.splitext(file_name)[0] + ".npy")):
                        setup_logger().info(f"Embedding already exists for {file_name}, skipping computation.")
                        continue

                    # Load the audio file with librosa
                    audio, _ = librosa.load(file_path, sr=self.ml.sr)
                    # Expand dimensions of the audio to match the expected input shape
                    audio = np.expand_dims(audio, axis=0)

                    # Normalize the audio
                    audio = audio / np.max(np.abs(audio))

                    # Get the embedding for the audio
                    embedding = self.get_embedding(audio)
                    
                    # Save the embedding as a .npy file
                    emb_file_path = os.path.join(emb_dir, os.path.splitext(file_name)[0] + ".npy")
                    setup_logger().info(f"Saving embedding to {emb_file_path}")
                    np.save(emb_file_path, embedding)

    def get_embedding(self, wav_data):
        """
        Compute embedding for an audio file and cache it to a file.
        """
        # Load file, get embedding, save embedding
        embd = self.ml.get_embedding(wav_data)
        return embd
    
    def get_stats(self, emb_dir):
        """
        Load embedding statistics from a directory.
        """

        mu, cov = calculate_embd_statistics_online(list(emb_dir.glob("*.npy")))
        
        return mu, cov

    def score(self, baseline, eval):
        """
        Calculate a single FAD score between a background and an eval set.

        :param baseline: Baseline matrix or directory containing baseline audio files
        :param eval: Eval matrix or directory containing eval audio files
        """
        if isinstance(baseline, str):
            baseline = Path(baseline)
        if isinstance(eval, str):
            eval = Path(eval)

        mu_bg, cov_bg = self.get_stats(baseline)
        mu_eval, cov_eval = self.get_stats(eval)

        return calc_frechet_distance(mu_bg, cov_bg, mu_eval, cov_eval)

    def score_individual(self, baseline, eval):
        """
        Calculate the FAD score for each individual file in eval_dir and write the results to a csv file.

        :param baseline: Baseline matrix or directory containing baseline audio files
        :param eval_dir: Directory containing eval audio files
        :param csv_name: Name of the csv file to write the results to
        :return: Path to the csv file
        """
        if isinstance(baseline, str):
            baseline = Path(baseline)
        if isinstance(eval, str):
            eval = Path(eval)

        # 3. Calculate z score for each eval file
        _files = list(Path(eval).glob("*.*"))

        # Compute reference statistics once
        mu_ref, cov_ref = self.get_stats(baseline)

        # Compute FAD per file
        scores = []
        for file_path in tq(_files, desc="Scoring individual files"):
            embd = np.load(file_path)
            mu_eval, cov_eval = calc_embd_statistics(embd)

            fad = calc_frechet_distance(mu_ref, cov_ref, mu_eval, cov_eval)
            print(fad)
            scores.append(fad)
            # try:
            #     fad = calc_frechet_distance(mu_ref, cov_ref, mu_eval, cov_eval)
            #     scores.append((file_path, fad))
            # except Exception as e:
            #     setup_logger().warning(f"Skipping {file_path} due to error: {e}")

        # 4. Write the sorted z scores to csv
        # pairs = list(zip(_files, scores))
        # pairs = [p for p in pairs if p[1] is not None]
        # pairs = sorted(pairs, key=lambda x: np.abs(x[1]))

        return scores