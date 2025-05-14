
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class HEDataset(Dataset):
    def __init__(self, path, tokenizer, args, split):
        self.args = args
        self.classes = args.classes
        self.num_classes = len(self.classes)
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = args.max_seq_len
        self.df = self.preprocess(path)
        self.data = self.load_data(self.df)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def save(self, path):
        self.df.to_pickle(path)

    def preprocess(self, path):
        # TODO: dont hardcode this
        label_incl = list(map(lambda x: ['RED', 'PCE', 'DWED'].index(x), self.classes))

        df = pd.read_pickle(path)
        df = df[df['split'] == self.split]
        # remap to be contiguous
        df['label'] = df['label'].apply(lambda x: [np.arange(self.num_classes)[l] for l in x if l in label_incl])

        self.label_dict = {
            i: label for i, label in enumerate(self.classes)
        }

        if self.args.only_jiong:
            df = df[df['task_type'].str.contains('jiong')]

        if self.args.trim:
            df['text'] = df['text'].apply(lambda x: " ".join(x.split()))

        if self.args.requirements:
            df['text'] = (
f"""#Find the defects in the code, given the following requirements
# #Start of requirements
# {df['requirements']}

# #Start of code
# {df['text']}"""
            )

        # append eos as classification token
        df['text'] = df['text'].apply(lambda x: x + self.tokenizer.eos_token)

        label_counts = df['label'].explode().value_counts().sort_index()
        label_counts /= label_counts.sum()
        self.wts = torch.FloatTensor(label_counts.to_list())
        # self.wts = (label_counts.sum() - self.wts) / self.wts
        self.wts = 1 / self.wts
        print(f'Label counts: {label_counts.to_list()}')

        return df

    def load_data(self, df):
        data = []

        output = self.tokenizer(df['text'].to_list(), padding='max_length', max_length=self.max_length, truncation=True, return_tensors='pt')
        input_ids = output['input_ids']
        att_mask = output['attention_mask']

        for i in range(len(df)):
            target = torch.zeros(self.num_classes)
            for l in df['label'].iloc[i]:
                target[l] = 1
            data.append({
                'input_ids': input_ids[i],
                'attention_mask': att_mask[i],
                'target': target
            })
        return data
