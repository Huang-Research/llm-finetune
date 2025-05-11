import pandas as pd
import numpy as np
import json
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer

data = []
with open('multi_combined_data.json', 'r') as f:
    for line in f:
        j = json.loads(line)
        j['split'] = j['metadata']['split']
        j['requirements'] = j['metadata']['requirements']
        j['task_type'] = j['metadata']['task_type']
        data.append(j)
df = pd.DataFrame(data).fillna(0)
df.label = df.label.map(lambda x: list(set(x)))

print(df[df['split'] == 'test']['task_type'].value_counts())
print(df[df['split'] == 'train']['task_type'].value_counts())
print(df[df['split'] == 'val']['task_type'].value_counts())

print(df.head())

train_df = df[df['split'] == 'train']
val_df = df[df['split'] == 'val']
test_df = df[df['split'] == 'test']

assert len(test_df['label'].explode().unique()) == len(train_df['label'].explode().unique()) == len(val_df['label'].explode().unique()), "Train and validation sets have different number of unique labels."

train_df.to_pickle('train.pkl')
val_df.to_pickle('val.pkl')
test_df.to_pickle('test.pkl')

print(f"Train set: {len(train_df)} samples")
print(f"\tLabel counts:", train_df['label'].explode().value_counts().to_dict())
print(f"Validation set: {len(val_df)} samples")
print(f"\tLabel counts:", val_df['label'].explode().value_counts().to_dict())
print(f"Test set: {len(test_df)} samples")
print(f"\tLabel counts:", test_df['label'].explode().value_counts().to_dict())

for name, df in zip(['train', 'val', 'test'], [train_df, val_df, test_df]):
    print (f"{name}:")
    bin = MultiLabelBinarizer().fit_transform(df['label'])

    y_pred = np.ones_like(bin)

    print("Baseline scores:")
    print(f"F1: {f1_score(bin, y_pred, average='macro')}")
    print(f"Precision: {precision_score(bin, y_pred, average='macro')}")
    print(f"Recall: {recall_score(bin, y_pred, average='macro')}")
