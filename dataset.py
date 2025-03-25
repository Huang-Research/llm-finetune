import pandas as pd
import json
from transformers import RobertaTokenizer
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class HEDataset(Dataset):
    def __init__(self, path, tokenizer, max_length, num_classes, split):
        self.split = split
        self.num_classes = num_classes
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.load_data(path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def load_data(self, path):
        data = []

        df = pd.read_pickle(path)
        df = df[df['split'] == self.split]
        # df = df[df['label'].apply(lambda x: len(x) > 0)]
        df['label'] = df['label'].apply(lambda x: [0] if 1 in x else [])

        input_ids, att_mask = self.tokenizer(df['text'].to_list(), padding='max_length', max_length=2048, truncation=True, return_tensors='pt').values()

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

        # df = pd.read_pickle(path)
        # df['split'] = df['metadata'].apply(lambda x: x['split'])
        # df = df[df['split'] == self.split]

        # input_ids, att_mask = self.tokenizer.batch_encode(df['text'], padding='max_length', max_length=self.max_length, truncation=True, return_tensors='pt').values()
        # for i in range(len(df)):
        #     targets = [0] * self.num_classes
        #     row = df.iloc[i]
        #     for l in row['label']:
        #         targets[l] = 1
        #     data.append({
        #         'input_ids': input_ids[i],
        #         'attention_mask': att_mask[i],
        #         'target': torch.tensor(targets)
        #     })

        return data
