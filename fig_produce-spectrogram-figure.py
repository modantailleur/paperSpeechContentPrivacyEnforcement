import os
import numpy as np
import librosa
import matplotlib.pyplot as plt

def plot_spectro(x_m, fs, extlmax=None, title='title', vmin=None, vmax=None, save=False, out_path=None):
    if vmin is None:
        vmin = np.min(x_m)
    if vmax is None:
        vmax = np.max(x_m)
    exthmin = 1
    exthmax = len(x_m)
    extlmin = 0
    if extlmax is None:
        extlmax = len(x_m[0])

    plt.figure(figsize=(5, 5))
    plt.imshow(x_m, extent=[extlmin, extlmax, exthmin, exthmax], cmap='inferno',
               vmin=vmin, vmax=vmax, origin='lower', aspect='auto')
    plt.axis('off')  # Remove axes
    if save and out_path:
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()

AUDIO_DIR = './output/audio_for_plot/'
DURATION = 2  # seconds

for filename in os.listdir(AUDIO_DIR):
    if filename.endswith('.wav') or filename.endswith('.mp3'):
        filepath = os.path.join(AUDIO_DIR, filename)
        y, sr = librosa.load(filepath, duration=DURATION)

        # Compute log spectrogram
        S = np.abs(librosa.stft(y))**2
        S_log = librosa.amplitude_to_db(S, ref=np.max)

        # Cut spectrogram between 20 Hz and 12000 Hz
        min_bin = int(20 / (sr / 2) * (S_log.shape[0] - 1))
        max_bin = int(6000 / (sr / 2) * (S_log.shape[0] - 1))
        S_log = S_log[min_bin:max_bin, :]

        # Save plot
        png_filename = os.path.splitext(filename)[0] + '_log.png'
        png_path = os.path.join(AUDIO_DIR, png_filename)
        plot_spectro(S_log, sr, title=filename, save=True, out_path=png_path, vmin=-83, vmax=-10)
