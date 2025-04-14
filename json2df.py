import pandas as pd
import json

data = []
with open('multi_combined_data.json', 'r') as f:
    for line in f:
        j = json.loads(line)
#         j['text'] = f"""#Find the defects in the code, given the following requirements
# #Start of requirements
# {j['metadata']['requirements']}

# #Start of code
# {j['text']}"""
        j['split'] = j['metadata']['split']
        # for label in j['label']:
            # j[f'label_{label}'] = 1

        data.append(j)
df = pd.DataFrame(data).fillna(0)
print(df.head())

train_df = df[df['split'] == 'train']
val_df = df[df['split'] == 'val']
test_df = df[df['split'] == 'test']

assert len(test_df['label'].explode().unique()) == len(train_df['label'].explode().unique()) == len(val_df['label'].explode().unique()), "Train and validation sets have different number of unique labels."

train_df.to_pickle('train.pkl')
val_df.to_pickle('val.pkl')
test_df.to_pickle('test.pkl')
