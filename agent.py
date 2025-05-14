import os, pickle
import torch
import wandb
from collections import defaultdict
import os, pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report

from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
)

from dataset import HEDataset
from baselines import (
    BagOfWordsModel,
    TfIdfModel,
    Word2VecModel,
)
from models import (
    FeatureClassifier,
    LayerFTModel,
    LoRAModel,
)

class Agent:
    def __init__(self, args, load_path=""):
        self.args = args
        self.device = args.device if args.device != 'auto' else 'cuda'

        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)

        self.load_or_resume(args)

    def save_checkpoint(self, args, path):
        ''' Save the adapter, optimizer, scheduler, dataset and training stats '''
        if not os.path.exists(path):
            os.makedirs(path)

        self.model.save(path)
        self.train_dataset.save(f"{path}/train.pkl")
        if args.test:
            self.val_dataset.save(f"{path}/test.pkl")
        else:
            self.val_dataset.save(f"{path}/val.pkl")
        torch.save(self.optimizer.state_dict(), f"{path}/optimizer.pt")
        torch.save(self.scheduler.state_dict(), f"{path}/scheduler.pt")
        torch.save(self.epoch, f"{path}/epoch.pt")
        pickle.dump(self.train_metrics, open(f"{path}/train_metrics.pkl", "wb"))
        pickle.dump(self.val_metrics, open(f"{path}/val_metrics.pkl", "wb"))
        pickle.dump(self.args, open(f"{path}/args.pkl", "wb"))

    def load_or_resume(self, args):
        ''' Initialize tokenizer and base model.
            Load the adapter, optimizer, scheduler, dataset and training stats '''

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
        except Exception as e:
            print(f"Could not load tokenizer from {args.model}, defaulting to codebert-base")
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base", use_fast=True, trust_remote_code=True)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.train_dataset = HEDataset(os.path.join(args.load, 'train.pkl'), self.tokenizer, args, split='train')
        if args.test:
            self.val_dataset = HEDataset(os.path.join(args.load, 'test.pkl'), self.tokenizer, args, split='test')
        else:
            self.val_dataset = HEDataset(os.path.join(args.load, 'val.pkl'), self.tokenizer, args, split='val')
        assert self.train_dataset.label_dict.keys() == self.val_dataset.label_dict.keys(), "Train and val datasets should contain the same classes"
        self.train_loader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

        if self.args.model == 'bow':
            embedder_class = BagOfWordsModel
        elif self.args.model == 'tfidf':
            embedder_class = TfIdfModel
        elif self.args.model == 'word2vec':
            embedder_class = Word2VecModel
        elif self.args.ft_layers > 0:
            embedder_class = LayerFTModel
        else:
            embedder_class = LoRAModel

        self.model = FeatureClassifier(
                embedder_class=embedder_class,
                tokenizer=self.tokenizer,
                train_dataset=self.train_dataset,
                args=args,
        )

        self.optimizer = torch.optim.AdamW([
            {
                'params': list(self.model.feature_extractor.parameters()),
                'lr': args.lr
             },
            {   'params': list(self.model.classifier.parameters()),
                'lr': args.lr_clf
            }
        ])

        num_training_steps = len(self.train_dataset) * self.args.epoch
        match args.scheduler:
            case 'linear':
                self.scheduler = get_linear_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=args.warmup_steps,
                    num_training_steps=num_training_steps,
                )
            case 'constant':
                self.scheduler = get_constant_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=args.warmup_steps,
                )
            case 'cosine':
                self.scheduler = get_cosine_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=args.warmup_steps,
                    num_training_steps=num_training_steps,
                )
            case 'cosine_with_restarts':
                self.scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                    self.optimizer,
                    num_warmup_steps=args.warmup_steps,
                    num_training_steps=num_training_steps,
                )
            case _:
                raise ValueError(f"Unknown scheduler: {args.scheduler}")

        if args.load:
            self.optimizer.load_state_dict(torch.load(f"{args.load}/optimizer.pt"))
            self.scheduler.load_state_dict(torch.load(f"{args.load}/scheduler.pt"))
            self.epoch = torch.load(f"{args.load}/epoch.pt")
            self.train_metrics = pickle.load(open(f"{args.load}/train_metrics.pkl", "rb"))
            self.val_metrics = pickle.load(open(f"{args.load}/val_metrics.pkl", "rb"))

    def train(self):
        for self.epoch in range(self.args.epoch):
            if self.args.profile:
                with torch.profiler.profile(
                    with_stack=True) as prof:
                    self.train_one_epoch()
                    self.validate()
                prof.export_chrome_trace("trace.json")
            else:
                self.train_one_epoch()

            if self.epoch % self.args.val_interval == 0:
                self.validate()

                if min(self.val_metrics['loss']) == self.val_metrics['loss'][-1]:
                    self.save_checkpoint(self.args, f'{wandb.run.dir}/best')
                    print(f'Best model saved to {wandb.run.dir}/best')

            if self.epoch == self.args.epoch - 1:
                self.save_checkpoint(self.args, f'{wandb.run.dir}/final')
                print(f'Final model saved to {wandb.run.dir}/final')

    def train_one_epoch(self):
        self.model.train()

        all_preds, all_targets, losses, accs = [], [], [], []

        tqdm_batch = tqdm(self.train_loader, total=len(self.train_loader), desc=f'Train [Epoch {self.epoch}]')
        for batch in tqdm_batch:
            target = batch['target'].to(self.device)
            self.optimizer.zero_grad()

            output = self.model.forward(batch)
            # convert logits to probabilities
            preds = torch.sigmoid(output).round()
            # the defect is predicted if probability >= 50%
            acc = (preds == target).sum().item() / target.numel()

            loss = self.compute_loss_multilabel(output, target)
            loss.backward()

            self.optimizer.step()
            self.scheduler.step()

            all_preds += [preds.detach().cpu().numpy()]
            all_targets += [batch['target'].cpu().numpy()]
            losses += [loss.item()]
            accs += [acc]
            tqdm_batch.set_postfix({'loss': np.mean(losses), 'acc': np.mean(accs)})

        # logging
        all_targets = np.concatenate(all_targets)
        all_preds = np.concatenate(all_preds)
        target_names = list(self.train_dataset.label_dict.values())
        if len(target_names) == 1:
            target_names = ['Negative'] + target_names
        report = classification_report(all_targets, all_preds, target_names=target_names, zero_division=0, output_dict=True)
        report = {f"{k}_train": v for k, v in report.items()}
        report['epoch'] = self.epoch
        report['lr'] = self.optimizer.param_groups[0]['lr']
        report['train_loss'] = np.mean(losses)
        report['train_acc'] = np.mean(accs)
        wandb.log(report)

    @torch.no_grad()
    def validate(self):
        self.model.eval()

        all_preds, all_targets, losses, accs = [], [], [], []

        tqdm_batch = tqdm(self.val_loader, total=len(self.val_loader), desc='Val')
        for batch in tqdm_batch:
            target = batch['target'].to(self.device)\

            output = self.model.forward(batch)
            
            # convert logits to probabilities
            preds = torch.sigmoid(output).round()
            # the defect is predicted if probability >= 50%
            acc = (preds == target).sum().item() / target.numel()
            loss = self.compute_loss_multilabel(output, target)
            # convert logits to probabilities
            preds = torch.sigmoid(output).round()

            all_preds += [preds.detach().cpu().numpy()]
            all_targets += [target.cpu().numpy()]
            losses += [loss.item()]
            accs += [acc]
            tqdm_batch.set_postfix({'loss': np.mean(losses), 'acc': np.mean(accs)})

        # logging
        self.val_metrics['loss'].append(np.mean(losses))
        all_targets = np.concatenate(all_targets)
        all_preds = np.concatenate(all_preds)
        target_names = list(self.train_dataset.label_dict.values())
        if len(target_names) == 1:
            target_names = ['Negative'] + target_names
        report = classification_report(all_targets, all_preds, target_names=target_names, zero_division=0, output_dict=True)
        report['epoch'] = self.epoch
        report['lr'] = self.optimizer.param_groups[0]['lr']
        report['val_loss'] = np.mean(losses)
        report['val_acc'] = np.mean(accs)
        wandb.log(report)

    def compute_loss_multilabel(self, logits, target):
        return F.binary_cross_entropy_with_logits(torch.sigmoid(logits), target, pos_weight=self.train_dataset.wts.to(self.device))

    def _register_embedding_list_hook(self, embeddings_list):
        def forward_hook(module, inputs, outputs):
            embeddings_list.append(outputs.clone().detach().cpu())
        embedding_layer = self.model.embedding_model.model.model.embed_tokens
        for p in embedding_layer.parameters():
            p.requires_grad = True
        handle = embedding_layer.register_forward_hook(forward_hook)
        return handle

    def _register_embedding_gradient_hooks(self, embeddings_gradients):
        def hook_layers(module, grad_in, grad_out):
            embeddings_gradients.append(grad_out[0].clone().detach().cpu())
        embedding_layer = self.model.embedding_model.model.model.embed_tokens
        for p in embedding_layer.parameters():
            p.requires_grad = True
        hook = embedding_layer.register_full_backward_hook(hook_layers)
        return hook

    def evaluate(self, df, salience=False):
        if salience:
            self.model.train()
            embeddings_list, embeddings_gradients = [], []
            handle = self._register_embedding_list_hook(embeddings_list)
            hook = self._register_embedding_gradient_hooks(embeddings_gradients)
        else:
            self.model.eval()
            self.model.requires_grad_(False)

        all_input_ids = []
        all_targets, all_preds, all_probs = [], [], []
        for i, row in tqdm(df.iterrows(), desc='Predicting', total=len(df)):
            text = row['text']

            target = torch.zeros(len(self.args.classes))
            for l in df.loc[i, 'label']:
                target[l] = 1

            batch = self.tokenizer(text, padding='max_length', max_length=self.args.max_seq_len, truncation=True, return_tensors='pt')
            target = torch.FloatTensor(target).to(self.device)
            batch['target'] = target

            self.optimizer.zero_grad()
            output = self.model.forward(batch)

            prob = torch.sigmoid(output).detach().cpu()
            pred = prob.round()

            if salience:
                loss = self.compute_loss_multilabel(output.squeeze(0), target)
                loss.backward()

            all_input_ids.append(batch['input_ids'].detach().cpu().numpy())
            all_targets.append(target.detach().cpu().numpy())
            all_preds.append(pred.numpy())
            all_probs.append(prob.numpy())
        df['text_tokenized'] = [self.tokenizer.tokenize(self.tokenizer.decode(input_ids[0])) for input_ids in all_input_ids]
        df['target'] = all_targets
        df['pred'] = all_preds
        df['prob'] = all_probs

        target_names = list(self.train_dataset.label_dict.values())
        report = classification_report(all_targets, np.concatenate(all_preds), target_names=target_names, zero_division=0)
        print(report)

        if salience:
            hook.remove()
            handle.remove()
            embeddings_gradients = torch.stack(embeddings_gradients).flatten(end_dim=1).cpu().numpy()
            embeddings_list = torch.stack(embeddings_list).flatten(end_dim=1).cpu().numpy()
            salience = np.sum(embeddings_gradients * embeddings_list, axis=1)
            df['salience'] = salience

        return df
