from transformers import RobertaForSequenceClassification, RobertaTokenizer, get_linear_schedule_with_warmup
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import HEDataset
from tqdm import tqdm

class Agent:
    def __init__(self):
        self.device = 'cuda'
        self.encoder = RobertaForSequenceClassification.from_pretrained('microsoft/codebert-base', num_labels=3).to(self.device)
        self.num_epochs = 10

        # freeze all except last 3 layers
        for param in self.encoder.roberta.parameters():
            param.requires_grad = False
        for param in self.encoder.roberta.encoder.layer[-3:].parameters():
            param.requires_grad = True
        # for param in self.encoder.roberta.pooler.parameters():
        #     param.requires_grad = True

        self.tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
        self.train_dataset = HEDataset('multi_combined_data.json', split='train')
        self.val_dataset = HEDataset('multi_combined_data.json', split='val')
        self.train_loader = DataLoader(self.train_dataset, batch_size=1, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=1, shuffle=False)
        self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr=1e-5)
        self.losses = []

        num_training_steps = len(self.train_dataset) * self.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_training_steps / 2,
            num_training_steps=num_training_steps,
        )

    def load_model(self, path):
        self.encoder.load_state_dict(torch.load(path))

    def train(self):
        for i in range(self.num_epochs):
            self.train_one_epoch()

            if i % 1 == 0:
                self.validate()

                if max(self.losses) == self.losses[-1]:
                    torch.save(self.encoder.state_dict(), 'best_model.pt')
                    print('Model saved')


    def train_one_epoch(self):
        self.encoder.train()

        losses = []
        accs = []
        tqdm_batch = tqdm(self.train_loader, total=len(self.train_loader), desc='Train')
        for batch in tqdm_batch:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            target = torch.stack(batch['target']).T.to(torch.float32).to(self.device)
            self.optimizer.zero_grad()

            outputs = self.encoder(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            acc = (torch.sigmoid(logits).round() == target).sum().item() / target.numel()
            loss = self.compute_loss_multilabel(logits, target)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            losses.append(loss.item())
            accs.append(acc)
            tqdm_batch.set_postfix({'loss': np.mean(losses), 'acc': np.mean(accs)})

    def validate(self):
        self.encoder.eval()

        losses = []
        accs = []
        tqdm_batch = tqdm(self.val_loader, total=len(self.val_loader), desc='Val')
        for batch in tqdm_batch:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            target = torch.stack(batch['target']).T.to(torch.float32).to(self.device)

            outputs = self.encoder(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            acc = (torch.sigmoid(logits).round() == target).sum().item() / target.numel()
            loss = self.compute_loss_multilabel(logits, target)

            losses.append(loss.item())
            accs.append(acc)
            tqdm_batch.set_postfix({'loss': np.mean(losses), 'acc': np.mean(accs)})
        self.losses.append(np.mean(losses))

    def compute_loss_multilabel(self, logits, target):
        # target is onehot
        return F.binary_cross_entropy_with_logits(torch.sigmoid(logits), target, weight=self.train_dataset.wts.to(self.device))

    def dump_val_preds(self):
        np.set_printoptions(precision=4, suppress=True)
        torch.set_printoptions(precision=4, sci_mode=False)

        self.encoder.eval()
        preds = []
        targets = []
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                target = torch.stack(batch['target']).T.to(torch.float32).to(self.device)

                outputs = self.encoder(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                preds.append(torch.sigmoid(logits).round().cpu().numpy())
                targets.append(target.cpu().numpy())
        for p, t in zip(preds, targets):
            print(p.squeeze(0), t.squeeze(0))
