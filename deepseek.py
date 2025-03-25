from transformers import RobertaForSequenceClassification, RobertaTokenizer, get_linear_schedule_with_warmup
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import HEDataset
from tqdm import tqdm
from sklearn.metrics import classification_report
from collections import defaultdict
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, set_peft_model_state_dict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig
)

model_name = "deepseek-ai/deepseek-coder-6.7b-base"
num_classes = 2

class Agent:
    def __init__(self):
        self.device = 'cuda'
        self.num_epochs = 5

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.train_dataset = HEDataset('multi_combined_data.pkl', self.tokenizer, 2048, 1, split='train')
        self.val_dataset = HEDataset('multi_combined_data.pkl', self.tokenizer, 2048, 1, split='val')
        self.train_loader = DataLoader(self.train_dataset, batch_size=1, shuffle=True, pin_memory=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=1, shuffle=False, pin_memory=True)

        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        lora_config = LoraConfig(
            r = 8,
            lora_alpha = 32,
            lora_dropout = 0.05,
            target_modules="all-linear",
            task_type="SEQ_CLS",
            inference_mode=False,
        )

        model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                    device_map="auto",
                                                    low_cpu_mem_usage=True,
                                                    trust_remote_code=False,
                                                    revision="main",
                                                    num_labels=self.train_dataset.num_classes,
                                                    quantization_config=config,
                                                    pad_token_id=self.tokenizer.pad_token_id,
                                                    use_cache=False)
        model = prepare_model_for_kbit_training(model, gradient_checkpointing_kwargs={'use_reentrant': False})
        self.model = get_peft_model(model, lora_config)
        self.model.print_trainable_parameters()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)

        num_training_steps = len(self.train_dataset) * self.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_training_steps * 0.06,
            num_training_steps=num_training_steps,
        )

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

    def train(self):
        for self.epoch in range(self.num_epochs):
            self.train_one_epoch()

            if self.epoch % 1 == 0:
                self.validate()

                if max(self.val_metrics['loss']) == self.val_metrics['loss'][-1]:
                    print('NOT Saving best model...')
                    # torch.save(self.model.state_dict(), 'best_model.pt')

    def train_one_epoch(self):
        self.model.train()

        all_preds = []
        all_targets = []

        losses = []
        accs = []
        tqdm_batch = tqdm(self.train_loader, total=len(self.train_loader), desc=f'Train [Epoch {self.epoch}]')
        for batch in tqdm_batch:
            self.optimizer.zero_grad()

            loss, acc, preds = self.forward(batch)
            loss.backward()
            
            all_preds.append(preds.detach().cpu().numpy())
            all_targets.append(batch['target'].cpu().numpy())

            self.optimizer.step()
            self.scheduler.step()

            losses.append(loss.item())
            accs.append(acc)
            tqdm_batch.set_postfix({'loss': np.mean(losses), 'acc': np.mean(accs)})
            print(classification_report(np.concatenate(all_targets), np.concatenate(all_preds), zero_division=np.nan))

        self.train_metrics['loss'].append(np.mean(losses))
        self.train_metrics['acc'].append(np.mean(accs))
        self.train_metrics['report_dict'].append(classification_report(np.concatenate(all_targets), np.concatenate(all_preds), zero_division=0, output_dict=True))
        self.train_metrics['report'].append(classification_report(np.concatenate(all_targets), np.concatenate(all_preds), zero_division=np.nan))
        self.train_metrics['lr'].append(self.optimizer.param_groups[0]['lr'])

    @torch.no_grad()
    def validate(self):
        self.model.eval()

        all_preds = []
        all_targets = []

        losses = []
        accs = []
        tqdm_batch = tqdm(self.val_loader, total=len(self.val_loader), desc='Val')
        for batch in tqdm_batch:

            loss, acc, preds = self.forward(batch)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch['target'].cpu().numpy())
            losses.append(loss.item())
            accs.append(acc)
            tqdm_batch.set_postfix({'loss': np.mean(losses), 'acc': np.mean(accs)})
            print(classification_report(np.concatenate(all_targets), np.concatenate(all_preds), zero_division=np.nan))
        self.val_metrics['loss'].append(np.mean(losses))
        self.val_metrics['acc'].append(np.mean(accs))
        self.val_metrics['report_dict'].append(classification_report(np.concatenate(all_targets), np.concatenate(all_preds), zero_division=0, output_dict=True))
        self.val_metrics['report'].append(classification_report(np.concatenate(all_targets), np.concatenate(all_preds), zero_division=np.nan))

    def forward(self, batch):
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        target = batch['target'].to(self.device)
        outputs = self.model(input_ids, attention_mask=attention_mask)

        logits = outputs.logits
        preds = torch.sigmoid(logits).round()
        acc = (preds == target).sum().item() / target.numel()
        loss = self.compute_loss_multilabel(logits, target)
        return loss, acc, preds

    def compute_loss_multilabel(self, logits, target):
        # target is onehot
        return F.binary_cross_entropy_with_logits(torch.sigmoid(logits), target, weight=self.train_dataset.wts.to(self.device))

    def dump_val_preds(self):
        np.set_printoptions(precision=4, suppress=True)
        torch.set_printoptions(precision=4, sci_mode=False)

        self.model.eval()
        preds = []
        targets = []
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                target = batch['target'].to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                preds.append(torch.sigmoid(logits).round().cpu().numpy())
                targets.append(target.cpu().numpy())
        for p, t in zip(preds, targets):
            print(p.squeeze(0), t.squeeze(0))
