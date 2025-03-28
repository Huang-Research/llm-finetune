from collections import defaultdict
import os, pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb.sklearn
from tqdm import tqdm
from sklearn.metrics import classification_report
import wandb
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)

from dataset import HEDataset

class Agent:
    def __init__(self, args, load_path=""):
        self.device = args.device
        self.num_epochs = args.epoch
        self.val_interval = args.val_interval
        self.profile = args.profile
        self.args = args

        self.load_or_resume(args, load_path)

    def save_checkpoint(self, args, path):
        ''' Save the adapter, optimizer, scheduler, dataset and training stats '''
        self.model.save_pretrained(path)
        self.train_dataset.save(path)
        self.val_dataset.save(path)
        torch.save(self.optimizer.state_dict(), f"{path}/optimizer.pt")
        torch.save(self.scheduler.state_dict(), f"{path}/scheduler.pt")
        torch.save(self.epoch, f"{path}/epoch.pt")
        pickle.dump(self.train_metrics, open(f"{path}/train_metrics.pkl", "wb"))
        pickle.dump(self.val_metrics, open(f"{path}/val_metrics.pkl", "wb"))
        pickle.dump(self.args, open(f"{path}/args.pkl", "wb"))

    def load_or_resume(self, args, load_path = ""):
        ''' Initialize tokenizer and base model.
            Load the adapter, optimizer, scheduler, dataset and training stats '''

        self.tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.train_dataset = HEDataset(os.path.join(load_path, 'train.pkl'), self.tokenizer, args.max_seq_len, classes=args.classes, split='train')
        self.val_dataset = HEDataset(os.path.join(load_path, 'val.pkl'), self.tokenizer, args.max_seq_len, classes=args.classes, split='val')
        assert self.train_dataset.label_dict.keys() == self.val_dataset.label_dict.keys(), "Train and val datasets should contain the same classes"
        self.train_loader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        # initialize new adapter or load existing
        lora_config = LoraConfig.from_pretrained(args.model) if load_path else (
            LoraConfig(
                r = args.lora_r,
                lora_alpha = args.lora_alpha,
                lora_dropout = 0.05,
                target_modules="all-linear",
                task_type="SEQ_CLS",
                inference_mode=False,
            )
        )

        # load pretrained model
        model = AutoModelForSequenceClassification.from_pretrained(args.model,
                                                    device_map="auto",
                                                    low_cpu_mem_usage=True,
                                                    trust_remote_code=False,
                                                    revision="main",
                                                    num_labels=len(args.classes),
                                                    quantization_config=config,
                                                    pad_token_id=self.tokenizer.pad_token_id,
                                                    use_cache=False)
        model = prepare_model_for_kbit_training(model, gradient_checkpointing_kwargs={'use_reentrant': False})
        self.model = get_peft_model(model, lora_config)
        self.model.print_trainable_parameters()

        # load optimizer, scheduler, metrics
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        num_training_steps = len(self.train_dataset) * self.num_epochs
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

        if load_path:
            self.optimizer.load_state_dict(torch.load(f"{load_path}/optimizer.pt"))
            self.scheduler.load_state_dict(torch.load(f"{load_path}/scheduler.pt"))
            self.epoch = torch.load(f"{load_path}/epoch.pt")
            self.train_metrics = pickle.load(open(f"{load_path}/train_metrics.pkl", "rb"))
            self.val_metrics = pickle.load(open(f"{load_path}/val_metrics.pkl", "rb"))
            args = pickle.load(open(f"{load_path}/args.pkl", "rb"))
            assert args == self.args, f"Loaded args do not match current args!!\n{args}"
            self.args = args

    def train(self):
        for self.epoch in range(self.num_epochs):
            if self.profile:
                with torch.profiler.profile(
                    with_stack=True) as prof:
                    self.train_one_epoch()
                    self.validate()
                prof.export_chrome_trace("trace.json")
            else:
                self.train_one_epoch()

            if self.epoch % self.val_interval == 0:
                self.validate()

                if max(self.val_metrics['loss']) == self.val_metrics['loss'][-1]:
                    self.save_checkpoint(self.args, self.args.output_dir)
                    print(f"Model saved to {self.args.output_dir}")

    def train_one_epoch(self):
        self.model.train()

        all_preds, all_targets, losses, accs = [], [], [], []

        tqdm_batch = tqdm(self.train_loader, total=len(self.train_loader), desc=f'Train [Epoch {self.epoch}]')
        for batch in tqdm_batch:
            self.optimizer.zero_grad()

            loss, acc, preds = self.forward(batch)
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

            loss, acc, preds = self.forward(batch)

            all_preds += [preds.detach().cpu().numpy()]
            all_targets += [batch['target'].cpu().numpy()]
            losses += [loss.item()]
            accs += [acc]
            tqdm_batch.set_postfix({'loss': np.mean(losses), 'acc': np.mean(accs)})
            print(classification_report(np.concatenate(all_targets), np.concatenate(all_preds), zero_division=np.nan))

        # logging
        self.val_metrics['loss'].append(np.mean(losses))
        all_targets = np.concatenate(all_targets)
        all_preds = np.concatenate(all_preds)
        target_names = list(self.train_dataset.label_dict.values())
        report = classification_report(all_targets, all_preds, target_names=target_names, zero_division=0, output_dict=True)
        report['epoch'] = self.epoch
        report['lr'] = self.optimizer.param_groups[0]['lr']
        report['val_loss'] = np.mean(losses)
        report['val_acc'] = np.mean(accs)
        wandb.log(report)

    def forward(self, batch):
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        target = batch['target'].to(self.device)

        # call forward pass of the model
        outputs = self.model(input_ids, attention_mask=attention_mask)
        # convert logits to probabilities
        preds = torch.sigmoid(outputs.logits).round()
        # the defect is predicted if probability >= 50%
        acc = (preds == target).sum().item() / target.numel()

        loss = self.compute_loss_multilabel(outputs.logits, target)
        return loss, acc, preds

    def compute_loss_multilabel(self, logits, target):
        return F.binary_cross_entropy_with_logits(torch.sigmoid(logits), target, weight=self.train_dataset.wts.to(self.device))

    def predict(self, input_texts):
        self.model.eval()
        input_texts = [input_texts] if isinstance(input_texts, str) else input_texts
        inputs = self.tokenizer(input_texts, padding=True, truncation=True, max_length=self.args.max_seq_len, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.sigmoid(outputs.logits).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
        return preds, probs
