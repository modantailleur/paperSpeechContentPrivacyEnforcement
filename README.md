# Enforcing Speech Content Privacy in Environmental Sound Recordings using Segment-wise Waveform Reversal

This is the code for our paper "Enforcing Speech Content Privacy in Environmental Sound Recordings using Segment-wise Waveform Reversal".

## Setup

The codebase is developed with Python 3.7. Install requirements as follows:
```
pip install -r requirements.txt
```

## Paper Results Replication

### Datasets download

Download our evaluation dataset in:
https://zenodo.org/records/15405950

And put it in the "dataset" folder. Only put in the folder the "cityspeechmix" folder and metadata.csv file (don't put the folder "stems" in "dataset" folder)

### Experiment

To reproduce the experiments of the paper, launch:

```
python3 experiment_anonymization.py
python3 generate_white_noise.py
python3 experiment_beats.py
python3 experiment_fad_emb.py
python3 experiment_wer.py
```

This will create files in the folder "output".

### Metrics

To print the metrics in your console, use:

```
python3 metric_accu_drop.py
python3 metric_fad.py
python3 metric_wer.py
```

### Figures

To reproduce the figures of the paper, launch:

```
python3 metric_accu_drop.py
python3 metric_fad.py
python3 metric_wer.py
```

Figures will appear in ./output/figures/ folder

### Additional code

The code used to create the custom dataset can be found in folder ./dataset_creation/

To create the dataset, you need to launch, in order:

```
python3 ./dataset_creation/select_sonyc-ust.py
python3 ./dataset_creation/select_librispeech_files.py
python3 ./dataset_creation/select_correspondance_librispeech_sonyc.py
python3 ./dataset_creation/select_librispeech_sounds.py
python3 ./dataset_creation/mix_sonyc_librispeech.py
python3 ./dataset_creation/normalized_sonyc_untouched.py
python3 ./dataset_creation/format_final_csv_file.py
```

## Companion page

Please check the companion page for audio examples:
https://modantailleur.github.io/paperSpeechContentPrivacyEnforcement/