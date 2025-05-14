import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import classification_report

class_names = ['Rule-encoding Deficiencies', 'Post-completion Error', 'Difficulties w/ Exponential Developments']

def compute_roc_auc_table(test_df):
    auc_data = []

    targets = np.reshape(np.concatenate(test_df["target"].to_numpy()), (-1, 3)).astype(int)
    probs = np.concatenate(test_df["prob"].to_numpy())

    row = {}
    for class_id in range(3):
        fpr, tpr, _ = roc_curve(targets[:, class_id], probs[:, class_id])
        roc_auc = auc(fpr, tpr)
        row[class_names[class_id]] = roc_auc
    row['Macro AUC'] = roc_auc_score(targets, probs, average='macro', multi_class='ovo')
    row['Weighted AUC'] = roc_auc_score(targets, probs, average='weighted', multi_class='ovo')
    # row['Macro MAP'] = roc_auc_score(targets, probs, average='macro', multi_class='ovo')
    auc_data.append(row)

    return pd.DataFrame(auc_data)

def compute_pr_auc_table(test_df):
    auc_data = []

    targets = np.reshape(np.concatenate(test_df["target"].to_numpy()), (-1, 3)).astype(int)
    probs = np.concatenate(test_df["prob"].to_numpy())

    row = {}
    for class_id in range(3):
        ap = average_precision_score(targets[:, class_id], probs[:, class_id])
        row[class_names[class_id]] = ap
    row['Macro AP'] = average_precision_score(targets, probs, average='macro')
    row['Weighted AP'] = average_precision_score(targets, probs, average='weighted')
    auc_data.append(row)

    return pd.DataFrame(auc_data)

def find_best_thresholds(val_df, step=0.01):
    best_thresholds = {}
    thresholds = np.arange(0.0, 1.01, step)

    targets = np.reshape(np.concatenate(val_df["target"].to_numpy()), (-1, 3)).astype(int)
    probs = np.concatenate(val_df["prob"].to_numpy())
    n_classes = targets.shape[1]

    for class_id in range(n_classes):
        best_f1 = 0
        best_t = 0.5
        for t in thresholds:
            preds = (probs[:, class_id] >= t).astype(int)
            f1 = f1_score(targets[:, class_id], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        best_thresholds[f'Class {class_id}'] = {
            'Best Threshold': best_t,
            'Best F1': best_f1
        }
    return best_thresholds

def evaluate_with_thresholds(test_df, best_thresholds):
    f1_scores = {}
    precision_scores = {}
    recall_scores = {}
    all_preds = []
    
    targets = np.reshape(np.concatenate(test_df["target"].to_numpy()), (-1, 3)).astype(int)
    probs = np.concatenate(test_df["prob"].to_numpy())
    n_classes = targets.shape[1]

    for class_id in range(n_classes):
        threshold = best_thresholds[f'Class {class_id}']['Best Threshold']
        preds = (probs[:, class_id] >= threshold).astype(int)
        f1 = f1_score(targets[:, class_id], preds, zero_division=0)
        precision = precision_score(targets[:, class_id], preds, zero_division=0)
        recall = recall_score(targets[:, class_id], preds, zero_division=0)
        f1_scores[f'Class {class_id}'] = f1
        precision_scores[f'Class {class_id}'] = precision
        recall_scores[f'Class {class_id}'] = recall
        all_preds.append(preds)

    # Stack predictions to compute macro F1
    all_preds = np.stack(all_preds, axis=1)
    macro_f1 = f1_score(targets, all_preds, average='macro', zero_division=0)
    macro_precision = precision_score(targets, all_preds, average='macro', zero_division=0)
    macro_recall = recall_score(targets, all_preds, average='macro', zero_division=0)
    f1_scores['Macro F1'] = macro_f1
    precision_scores['Macro Precision'] = macro_precision
    recall_scores['Macro Recall'] = macro_recall
    weighted_f1 = f1_score(targets, all_preds, average='weighted', zero_division=0)
    f1_scores['Weighted F1'] = weighted_f1
    weighted_precision = precision_score(targets, all_preds, average='weighted', zero_division=0)
    precision_scores['Weighted Precision'] = weighted_precision
    weighted_recall = recall_score(targets, all_preds, average='weighted', zero_division=0)
    recall_scores['Weighted Recall'] = weighted_recall

    return f1_scores, precision_scores, recall_scores

def compute_baseline(train_df):
    baseline_metrics = {}

    targets = np.reshape(np.concatenate(train_df["target"].to_numpy()), (-1, 3)).astype(int)
    n_samples, n_classes = targets.shape

    # Compute empirical label probabilities
    label_probs = targets.mean(axis=0)

    # Generate random predictions based on label frequencies
    preds = np.random.binomial(n=1, p=label_probs, size=(n_samples, n_classes))

    for class_id in range(n_classes):
        f1 = f1_score(targets[:, class_id], preds[:, class_id], zero_division=0)
        precision = precision_score(targets[:, class_id], preds[:, class_id], zero_division=0)
        recall = recall_score(targets[:, class_id], preds[:, class_id], zero_division=0)
        baseline_metrics[f'Class {class_id}'] = {
            'F1': f1,
            'Precision': precision,
            'Recall': recall
        }

    # macro
    macro_f1 = f1_score(targets, preds, average='macro', zero_division=0)
    macro_precision = precision_score(targets, preds, average='macro', zero_division=0)
    macro_recall = recall_score(targets, preds, average='macro', zero_division=0)
    baseline_metrics['Macro'] = {
        'F1': macro_f1,
        'Precision': macro_precision,
        'Recall': macro_recall
    }

    # weighted
    weighted_f1 = f1_score(targets, preds, average='weighted', zero_division=0)
    weighted_precision = precision_score(targets, preds, average='weighted', zero_division=0)
    weighted_recall = recall_score(targets, preds, average='weighted', zero_division=0)
    baseline_metrics['Weighted'] = {
        'F1': weighted_f1,
        'Precision': weighted_precision,
        'Recall': weighted_recall
    }

    return baseline_metrics
