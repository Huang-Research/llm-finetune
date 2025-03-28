import pandas as pd
import json

data = []
with open('multi_combined_data.json', 'r') as f:
    for line in f:
        j = json.loads(line)
        j['split'] = j['metadata']['split']
        # for label in j['label']:
            # j[f'label_{label}'] = 1

        data.append(j)
df = pd.DataFrame(data).fillna(0)
print(df.head())

train_df = df[df['split'] == 'train']
val_df = df[df['split'] == 'val']
train_df.to_pickle('train.pkl')
val_df.to_pickle('val.pkl')
