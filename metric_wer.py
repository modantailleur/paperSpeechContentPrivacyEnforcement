import pandas as pd
import glob
import os
from scipy.stats import sem, t
import numpy as np
from scipy.stats import ttest_rel

def compute_wer(mname):
    # Path to the folder containing the CSV files
    folder_path = f"/home/user/Documents/Thèse/Code/9-VoiceContentPrivacy/output/{mname}/"

    # Get all CSV files starting with "wer_"
    csv_files = glob.glob(os.path.join(folder_path, "wer_*.csv"))

    # List to store dataframes
    dfs = []

    # Read and process each CSV file
    for file in csv_files:
        df = pd.read_csv(file)
        df['method'] = os.path.basename(file)  # Add the filename as a new column
        dfs.append(df)

    # Concatenate all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)

    # Apply threshold on the "wer" column
    combined_df['WER'] = combined_df['WER'].apply(lambda x: min(x, 1))

    # Compute mean and 95% confidence interval
    mean_wer = combined_df['WER'].mean()
    confidence_interval = 1.96 * combined_df['WER'].std() / (len(combined_df['WER']) ** 0.5)

    # Print results
    print(f"Mean WER: {mean_wer*100:.1f}")
    print(f"95% Confidence Interval: ±{confidence_interval*100:.1f}")

    return combined_df['WER'].tolist()

wer_dict = {}

evaluations = [
    ['oracle', 'noise'],
    ['cohen', 'burkhardt', 'ours_envss', 'ours'],
    ['ours', 'ours_novad', 'ours_noss'],
    ['ours', 'ours_with_mixframe'],
    ['rev_ours', 'rev_ours_with_mixframe']
]

for mname_list in evaluations:
    print(f"\nEvaluating methods: {mname_list}")
    for mname in mname_list:
        print(mname)
        wer_list = compute_wer(mname)
        wer_dict[mname] = wer_list

    # Find the best-performing method (lowest mean accuracy drop)
    best_method = max(wer_dict, key=lambda k: np.mean(wer_dict[k]))
    print(f"\nBest method: {best_method}")

    # Compare the best method to all others
    for method, drop_values in wer_dict.items():
        if method == best_method:
            continue
        # Paired t-test: drop values per class

        t_stat, p_value = ttest_rel(wer_dict[method], wer_dict[best_method])
        significance = "NOT statistically significant" if p_value > 0.01 else "statistically significant"
        print(f"Comparing {best_method} vs {method}: p = {p_value:.4f} → {significance}")