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
from llm2vec.llm2vec.models.bidirectional_llama import LlamaBiModel
from transformers import (
    LlamaPreTrainedModel,
    AutoModelForSequenceClassification,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
)
from peft import (
    PeftModel,
    PeftModelForSequenceClassification,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    prepare_model_for_kbit_training,
)

from dataset import HEDataset

# # adapted from llm2vec
# class LlamaBiForSeqCls(torch.nn.Module):
#     def __init__(self, config):
#         self.model = LlamaBiModel(config)
#         self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels, bias=False, dtype=torch.float)

#         # Initialize weights and apply final processing
#         self.post_init()

#     def forward(self, input_ids, attention_mask=None, labels=None):
#         outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
#         return self.classifier(outputs[0].mean(dim=1).to(self.classifier.weight.dtype))

class Agent:
    def __init__(self, args, load_path=""):
        self.device = args.device if args.device != 'auto' else 'cuda'
        self.num_epochs = args.epoch
        self.val_interval = args.val_interval
        self.profile = args.profile
        self.args = args

        self.load_or_resume(args)

    def save_checkpoint(self, args, path):
        ''' Save the adapter, optimizer, scheduler, dataset and training stats '''
        if not os.path.exists(path):
            os.makedirs(path)

        torch.save(get_peft_model_state_dict(self.model), f"{path}/adapter_model.pt")
        torch.save(self.classifier.weight, f"{path}/classifier.pt")
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

        self.tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        assert self.tokenizer.model_max_length >= args.max_seq_len, f"Tokenizer max length {self.tokenizer.model_max_length} is less than max_seq_len {args.max_seq_len}"

        self.train_dataset = HEDataset(os.path.join(args.load, 'train.pkl'), self.tokenizer, args, split='train')
        if args.test:
            self.val_dataset = HEDataset(os.path.join(args.load, 'test.pkl'), self.tokenizer, args, split='test')
        else:
            self.val_dataset = HEDataset(os.path.join(args.load, 'val.pkl'), self.tokenizer, args, split='val')
        assert self.train_dataset.label_dict.keys() == self.val_dataset.label_dict.keys(), "Train and val datasets should contain the same classes"
        self.train_loader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            # TODO: the classification head should be separate from the base model
            llm_int8_skip_modules=["classifier", "pre_classifier"] if args.model.find('codebert') != -1 else None,
        )

        lora_config = LoraConfig(
                r = args.lora_r,
                lora_alpha = args.lora_alpha,
                lora_dropout = 0.1,
                target_modules="all-linear",
                task_type="FEATURE_EXTRACTION" if args.encoder else "CAUSAL_LM",
                inference_mode=False,
        )

        # load pretrained model
        model_class = AutoModel if args.encoder else AutoModelForCausalLM
        model_class = LlamaBiModel if args.bidirectional else model_class 
        model = model_class.from_pretrained(args.model,
                                            device_map=args.device,
                                            low_cpu_mem_usage=True,
                                            trust_remote_code=True,
                                            revision="main",
                                            # num_labels=len(args.classes),
                                            quantization_config=quant_config,
                                            pad_token_id=self.tokenizer.pad_token_id)

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=(args.device == 'auto'), gradient_checkpointing_kwargs={'use_reentrant': False})
        model = get_peft_model(model, lora_config)
        self.model = model
        self.classifier = torch.nn.Linear(model.config.hidden_size, len(args.classes), bias=False, dtype=torch.float, device=self.device)

        if args.load:
            set_peft_model_state_dict(self.model, torch.load(f"{args.load}/adapter_model.pt"))
            with torch.no_grad():
                self.classifier.weight.copy_(torch.load(f"{args.load}/classifier.pt"))
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

        if args.load:
            self.optimizer.load_state_dict(torch.load(f"{args.load}/optimizer.pt"))
            self.scheduler.load_state_dict(torch.load(f"{args.load}/scheduler.pt"))
            self.epoch = torch.load(f"{args.load}/epoch.pt")
            self.train_metrics = pickle.load(open(f"{args.load}/train_metrics.pkl", "rb"))
            self.val_metrics = pickle.load(open(f"{args.load}/val_metrics.pkl", "rb"))

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
                    self.save_checkpoint(self.args, f'{wandb.run.dir}/checkpoint')
                    print(f'Best model saved to {wandb.run.dir}/checkpoint')

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

            loss, acc, preds = self.forward(batch)

            all_preds += [preds.detach().cpu().numpy()]
            all_targets += [batch['target'].cpu().numpy()]
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

    def forward(self, batch):
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        target = batch['target'].to(self.device)

        # call forward pass of the model
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden_state = output.hidden_states[-1]
        if self.args.encoder:
            mean_pooled = last_hidden_state.sum(axis=1) / attention_mask.sum(axis=-1).unsqueeze(-1)
            output = self.classifier(mean_pooled.to(self.classifier.weight.dtype))
        else:
            last_token_idx = attention_mask.bool().sum(1) - 1
            embeds = last_hidden_state[torch.arange(last_hidden_state.shape[0]), last_token_idx]
            # embeds = last_hidden_state[:, 0, :]
            output = self.classifier(embeds.to(self.classifier.weight.dtype))
        # convert logits to probabilities
        preds = torch.sigmoid(output).round()
        # the defect is predicted if probability >= 50%
        acc = (preds == target).sum().item() / target.numel()

        loss = self.compute_loss_multilabel(output, target)
        return loss, acc, preds

    def compute_loss_multilabel(self, logits, target):
        return F.binary_cross_entropy_with_logits(torch.sigmoid(logits), target, pos_weight=self.train_dataset.wts.to(self.device))

    def _register_embedding_list_hook(self, embeddings_list):
        def forward_hook(module, inputs, outputs):
            embeddings_list.append(outputs.clone().detach().cpu())
        # embedding_layer = self.model.base_model.model.model.embed_tokens
        embedding_layer = self.model.model.model.embed_tokens
        for p in embedding_layer.parameters():
            p.requires_grad = True
        handle = embedding_layer.register_forward_hook(forward_hook)
        return handle

    def _register_embedding_gradient_hooks(self, embeddings_gradients):
        def hook_layers(module, grad_in, grad_out):
            embeddings_gradients.append(grad_out[0].clone().detach().cpu())
        embedding_layer = self.model.model.model.embed_tokens
        for p in embedding_layer.parameters():
            p.requires_grad = True
        hook = embedding_layer.register_full_backward_hook(hook_layers)
        return hook

    def generate_salience(self, texts, targets):
        salience = []
        embeddings_list, embeddings_gradients = [], []
        probs, inputs = [], []

        handle = self._register_embedding_list_hook(embeddings_list)
        hook = self._register_embedding_gradient_hooks(embeddings_gradients)

        for text, target in tqdm(zip(texts, targets), desc='Generating salience', total=len(texts)):
            output = self.tokenizer(text, padding='max_length', max_length=self.args.max_seq_len, truncation=True, return_tensors='pt')
            input_ids = output['input_ids'].to(self.device)
            attention_mask = output['attention_mask'].to(self.device)
            target = target.to(self.device)

            # call forward pass of the model
            output = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            last_hidden_state = output.hidden_states[-1]
            if self.args.encoder:
                mean_pooled = last_hidden_state.sum(axis=1) / attention_mask.sum(axis=-1).unsqueeze(-1)
                output = self.classifier(mean_pooled.to(self.classifier.weight.dtype))
            else:
                last_token_idx = attention_mask.bool().sum(1) - 1
                embeds = last_hidden_state[torch.arange(last_hidden_state.shape[0]), last_token_idx]
                # embeds = last_hidden_state[:, 0, :]
                output = self.classifier(embeds.to(self.classifier.weight.dtype)) 
            prob = torch.sigmoid(output).detach().cpu().numpy()
            loss = self.compute_loss_multilabel(output.squeeze(0), target)
            loss.backward()
            probs.append(prob)
            inputs.append(input_ids)

        hook.remove()
        handle.remove()
        embeddings_gradients = torch.stack(embeddings_gradients).flatten(end_dim=1).cpu().numpy()
        embeddings_list = torch.stack(embeddings_list).flatten(end_dim=1).cpu().numpy()
        salience = np.sum(embeddings_gradients * embeddings_list, axis=1)
        toks = torch.stack(inputs, dim=0).cpu().numpy()
        toks = [self.tokenizer.tokenize(self.tokenizer.decode(t[0])) for t in toks]
        print(salience.shape)
        return salience, toks, probs, targets
    
    def predict(self, input_texts, targets):
        self.model.train()
        texts_tokenized, preds, probs = [], [], []

        with torch.no_grad():
            for text, target in tqdm(zip(input_texts, targets), desc='Predicting', total=len(input_texts)):
                output = self.tokenizer(text, padding='max_length', max_length=self.args.max_seq_len, truncation=True, return_tensors='pt')
                input_ids = output['input_ids'].to(self.device)
                attention_mask = output['attention_mask'].to(self.device)
                texts_tokenized.append(self.tokenizer.decode(input_ids[0]))
                target = target.to(self.device)

                # call forward pass of the model
                output = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                last_hidden_state = output.hidden_states[-1]
                if self.args.encoder:
                    mean_pooled = last_hidden_state.sum(axis=1) / attention_mask.sum(axis=-1).unsqueeze(-1)
                    output = self.classifier(mean_pooled.to(self.classifier.weight.dtype))
                else:
                    last_token_idx = attention_mask.bool().sum(1) - 1
                    embeds = last_hidden_state[torch.arange(last_hidden_state.shape[0]), last_token_idx]
                    # embeds = last_hidden_state[:, 0, :]
                    output = self.classifier(embeds.to(self.classifier.weight.dtype))
                # convert logits to probabilities
                prob = torch.sigmoid(output).detach().cpu()
                pred = prob.round()
                preds.append(pred)
                probs.append(prob)

        target_names = list(self.train_dataset.label_dict.values())
        if len(target_names) == 1:
            target_names = ['Negative'] + target_names
        report = classification_report(targets, np.concatenate(preds), target_names=target_names, zero_division=0, output_dict=True)
        print(report)

        return texts_tokenized, preds, probs
    
    def tsne(self, input_texts, targets):
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt
                
        self.model.train()
        texts_tokenized, preds, probs = [], [], []
        embeds = []

        with torch.no_grad():
            for text, target in tqdm(zip(input_texts, targets), desc='Predicting', total=len(input_texts)):
                output = self.tokenizer(text, padding='max_length', max_length=self.args.max_seq_len, truncation=True, return_tensors='pt')
                input_ids = output['input_ids'].to(self.device)
                attention_mask = output['attention_mask'].to(self.device)
                texts_tokenized.append(self.tokenizer.decode(input_ids[0]))
                target = target.to(self.device)

                # call forward pass of the model
                output = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                last_hidden_state = output.hidden_states[-1]
                if self.args.encoder:
                    embed = last_hidden_state.sum(axis=1) / attention_mask.sum(axis=-1).unsqueeze(-1)
                else:
                    last_token_idx = attention_mask.bool().sum(1) - 1
                    embed = last_hidden_state[torch.arange(last_hidden_state.shape[0]), last_token_idx]
                output = self.classifier(embed.to(self.classifier.weight.dtype))
                embeds.append(output.cpu().numpy())

        embeds = np.concatenate(embeds)
        # pca = PCA(n_components=3)
        # pca_result = pca.fit_transform(embeds)
        tsne = TSNE(n_components=2, perplexity=15, n_iter=400)
        tsne_result = tsne.fit_transform(embeds)
        fig, ax = plt.subplots(3, 1, figsize=(5, 15))
        for i in range(len(targets)):
            ax[0].scatter(tsne_result[i, 0], tsne_result[i, 1], c='red' if targets[i][0] else 'blue')
            ax[1].scatter(tsne_result[i, 0], tsne_result[i, 1], c='red' if targets[i][1] else 'blue')
            ax[2].scatter(tsne_result[i, 0], tsne_result[i, 1], c='red' if targets[i][2] else 'blue')

        plt.savefig('output/tsne.png')        
        return texts_tokenized, preds, probs
