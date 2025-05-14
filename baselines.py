from models import FeatureExtractor
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import torch
from torch.nn import LSTM
import gensim

class BagOfWordsModel(FeatureExtractor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.embed_dim = len(self.tokenizer)
        print(f"BagOfWordsModel embedding dim: {self.embed_dim}")

    def forward(self, batch):
        input_ids = batch['input_ids']
        feature_batch = []
        for i in range(input_ids.size(0)):
            feature_batch.append(self.vectorize(input_ids[i]))
        features = torch.stack(feature_batch, dim=0)
        return features.to(self.device)

    def vectorize(self, input_ids):
        bow = torch.zeros((self.embed_dim), dtype=torch.float32)
        for i in input_ids:
            bow[i] += 1
        return bow.to(self.device)

class TfIdfModel(FeatureExtractor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
        train_dataset = kwargs['train_dataset']
        self.tokenizer = train_dataset.tokenizer

        self.vectorizer = TfidfVectorizer(
            tokenizer=self.tokenizer.tokenize,
        )
        self.vectorizer.fit(train_dataset.df['text'])
        self.embed_dim = len(self.vectorizer.vocabulary_)
        print(f"TfIdfModel embedding dim: {self.embed_dim}")
        with open('output/tfidf_vocab.txt', 'w') as f:
            for word, idx in self.vectorizer.vocabulary_.items():
                f.write(f"{word}\t{idx}\n")

    def forward(self, batch):
        text = self.tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
        features = self.vectorizer.transform(text).toarray()
        features = torch.tensor(features, dtype=torch.float32).to(self.device)
        return features

class Word2VecModel(FeatureExtractor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        train_dataset = kwargs['train_dataset']
        self.tokenizer = train_dataset.tokenizer

        self.model = gensim.models.Word2Vec(sentences=train_dataset.df['text'], vector_size=128, window=5, min_count=1, workers=4)
        self.embed_dim = self.model.vector_size
        print(f"Word2VecModel embedding dim: {self.embed_dim}")

    def forward(self, batch):
        text = self.tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
        features = []
        for sentence in text:
            words = sentence.split()
            word_vectors = [self.model.wv[word] for word in words if word in self.model.wv]
            if word_vectors:
                features.append(torch.mean(torch.tensor(word_vectors), dim=0))
            else:
                features.append(torch.zeros(self.embed_dim))
        features = torch.stack(features).to(self.device)
        return features
