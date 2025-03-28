
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class HEDataset(Dataset):
    def __init__(self, path, tokenizer, max_length, classes, split):
        self.classes = classes
        self.num_classes = len(self.classes)
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.df = self.preprocess(path)
        self.data = self.load_data(self.df)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def preprocess(self, path):
        # TODO: dont hardcode this
        label_incl = list(map(lambda x: ['RED', 'PCE', 'DWED'].index(x), self.classes))

        df = pd.read_pickle(path)
        df = df[df['split'] == self.split]
        # remap to be contiguous
        df['label'] = df['label'].apply(lambda x: [np.arange(self.num_classes)[l] for l in x if l in label_incl])
        return df
        
    def load_data(self, df):
        data = []

        self.label_dict = {
            i: label for i, label in enumerate(self.classes)
        }

        input_ids, att_mask = self.tokenizer(df['text'].to_list(), padding='max_length', max_length=self.max_length, truncation=True, return_tensors='pt').values()

        label_freqs = df['label'].explode().value_counts().sort_index()
        label_freqs /= label_freqs.sum()
        self.wts = torch.FloatTensor(label_freqs.to_list())

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
