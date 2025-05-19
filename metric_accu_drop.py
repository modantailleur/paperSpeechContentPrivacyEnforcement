import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.stats import ttest_rel
import statsmodels.stats.proportion  as stms
from statsmodels.stats.contingency_tables import mcnemar

def run_mcnemar_test(predictions_A, predictions_B):
    # Build contingency table
    b = sum((a == 1 and b == 0) for a, b in zip(predictions_A, predictions_B))  # A correct, B wrong
    c = sum((a == 0 and b == 1) for a, b in zip(predictions_A, predictions_B))  # A wrong, B correct

    table = [[0, b],
             [c, 0]]

    result = mcnemar(table, exact=True)  # use exact test for small sample sizes
    return result.pvalue, b, c

# Determine the predicted classes
def get_predicted_classes(row, class_columns):
    if pd.isna(row['label1']) and pd.isna(row['label2']):
        return []
    elif (pd.isna(row['label1']) or pd.isna(row['label2'])):
        numeric_row = pd.to_numeric(row[class_columns], errors='coerce').fillna(float('-inf'))
        max_class = numeric_row.idxmax()
        return [max_class]
    else:
        top_classes = sorted(class_columns, key=lambda col: row[col], reverse=True)[:2]
        return top_classes

def compute_accuracies(mname, ref_accuracy=None):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(f'output/{mname}/logits.csv')

    # Define the class columns (all columns after 'label2')
    class_columns = df.columns[3:]

    # Ensure the class columns are numeric and fill NaN values with -inf
    df[class_columns] = df[class_columns].apply(pd.to_numeric, errors='coerce').fillna(float('-inf'))
    df[class_columns] = df[class_columns].astype('float32')

    df['predicted'] = df.apply(lambda row: get_predicted_classes(row, class_columns), axis=1)

    predictions = []
    for label1, label2, predicted_l in zip(df['label1'], df['label2'], df['predicted']):
        for predicted in predicted_l:
            if label1 == predicted or label2 == predicted:
                predictions.append(1)
            else:
                predictions.append(0)

    ciF_low, ciF_upp = stms.proportion_confint(predictions.count(1), len(predictions), method='beta', alpha=0.01)
    mean_accuracy = predictions.count(1) / len(predictions)
    conf_interval = (ciF_upp - ciF_low) / 2
    print(f"Global Accuracy: {mean_accuracy*100:.1f} ± {conf_interval*100:.1f}")

    if ref_accuracy is not None:
        # Calculate the drop in accuracy
        accuracy_drop = (ref_accuracy - mean_accuracy)
        print(f"Accuracy Drop: {accuracy_drop*100:.1f} ± {conf_interval*100:.1f}%")

    return(predictions, mean_accuracy)

_, ref_accuracy = compute_accuracies('oracle') 

evaluations = [
    ['oracle', 'noise'],
    ['cohen', 'burkhardt', 'ours_envss', 'ours'],
    ['ours', 'ours_novad', 'ours_noss'],
    ['ours', 'ours_with_mixframe'],
]


for mname_list in evaluations:
    print(f"\nEvaluating methods: {mname_list}")
    predictions_dict = {}

    for mname in mname_list:
        print(mname)
        predictions, _ = compute_accuracies(mname, ref_accuracy)
        predictions_dict[mname] = predictions

    # Find the best-performing method (lowest mean accuracy drop)
    best_method = max(predictions_dict, key=lambda k: np.mean(predictions_dict[k]))
    print(f"\nBest method: {best_method}")

    # Compare the best method to all others
    for method, drop_values in predictions_dict.items():
        if method == best_method:
            continue
        
        # accuracy = predictions_dict[method].count(1) / len(predictions_dict[method])
        # ciF_low, ciF_upp = stms.proportion_confint(predictions.count(1), len(predictions), method='beta', alpha=0.01)
        # conf_interval = (ciF_upp - ciF_low) / 2

        # accuracy_best = predictions_dict[best_method].count(1) / len(predictions_dict[best_method])
        # ciF_low_best, ciF_upp_best = stms.proportion_confint(predictions_dict[best_method].count(1), len(predictions_dict[best_method]), method='beta', alpha=0.01)
        # conf_interval_best = (ciF_upp_best - ciF_low_best) / 2

        # print(f"Accuracy {method}: {accuracy*100:.2f} ± {conf_interval*100:.2f}")
        # print(f"Accuracy {best_method}: {accuracy_best*100:.2f} ± {conf_interval_best*100:.2f}")

        # Paired t-test: drop values per class
        t_stat, p_value = ttest_rel(predictions_dict[method], predictions_dict[best_method])

        p_value, b, c = run_mcnemar_test(predictions_dict[method], predictions_dict[best_method])
        print(f"McNemar’s test: p = {p_value:.4f} (b={b}, c={c})")
        significance = "NOT statistically significant" if p_value > 0.01 else "statistically significant"
        # significance = "NOT statistically significant" if p_value > 0.05 else "statistically significant"
        print(f"Comparing {best_method} vs {method}: p = {p_value} → {significance}")