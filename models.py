import torch
import torch.nn as nn

from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    prepare_model_for_kbit_training,
)

class FeatureExtractor(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.args = kwargs['args']
        self.tokenizer = kwargs['tokenizer']
        self.device = self.args.device if self.args.device != 'auto' else 'cuda'

    def load(self, path):
        pass
    
    def save(self, path):
        pass

    def forward(self, batch):
        raise NotImplementedError()

class EmbeddingModel(FeatureExtractor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            # TODO: the classification head should be separate from the base model
            llm_int8_skip_modules=["classifier", "pre_classifier"] if self.args.model.find('codebert') != -1 else None,
        )

        # load pretrained model
        model_class = AutoModel if self.args.encoder else AutoModelForCausalLM
        self.model = model_class.from_pretrained(
            self.args.model,
            device_map=self.args.device,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            revision="main",
            # num_labels=len(args.classes),
            quantization_config=quant_config,
            pad_token_id=self.tokenizer.pad_token_id
        )

        self.num_chunks = int(1 + (self.args.max_seq_len - self.args.chunk_size) / self.args.chunk_stride)
        self.embed_dim = self.num_chunks * self.model.config.hidden_size
        print(f'Feature vec: {self.num_chunks} * {self.model.config.hidden_size} = {self.embed_dim}')
        assert (self.args.max_seq_len - self.args.chunk_size) % self.args.chunk_stride == 0

    def load(self, path):
        self.model.load_state_dict(torch.load(f'{path}/model.pt'))
        self.model = self.model.to(self.model.device)

    def save(self, path):
        torch.save(self.model.state_dict(), f'{path}/model.pt')

    def forward(self, batch):
        input_ids = batch['input_ids'].to(self.model.device)
        attention_mask = batch['attention_mask'].to(self.model.device)
        window_embeds = []
        for i in range(self.num_chunks):
            start = i * self.args.chunk_stride
            end = start + self.args.chunk_size
            chunked_input_ids = input_ids[:, start:end]
            chunked_attention_mask = attention_mask[:, start:end]
            output = self.model(input_ids=chunked_input_ids, attention_mask=chunked_attention_mask, output_hidden_states=True)
            last_hidden_state = output.hidden_states[-1]
            if self.args.encoder:
                # Take mean of all tokens in the sequence
                embeds = last_hidden_state.sum(dim=1) / chunked_attention_mask.sum(dim=-1, keepdim=True).clamp(min=1)
            else:
                assert self.tokenizer.padding_side == 'left'
                # Take last token of each sequence in the batch
                last_tok_idx = (chunked_attention_mask.sum(dim=1) - 1).clamp(min=0)
                embeds = last_hidden_state[torch.arange(last_hidden_state.shape[0]), last_tok_idx]
            window_embeds.append(embeds)

        embedding = torch.cat(window_embeds, dim=1)
        return embedding

class LoRAModel(EmbeddingModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        lora_config = LoraConfig(
                r = self.args.lora_r,
                lora_alpha = self.args.lora_alpha,
                lora_dropout = 0.1,
                target_modules='all-linear',
                task_type="FEATURE_EXTRACTION" if self.args.encoder else "CAUSAL_LM",
                inference_mode=False,
        )

        self.model = prepare_model_for_kbit_training(
            self.model,
            # CodeSage does not support gradient checkpointing or auto mapping
            use_gradient_checkpointing=(self.args.device == 'auto'),
            gradient_checkpointing_kwargs={'use_reentrant': False}
        )
        self.model = get_peft_model(self.model, lora_config)

    def load(self, path):
        set_peft_model_state_dict(self.model, torch.load(f'{path}/adapter_model.pt'))
        self.model = self.model.to(self.model.device)
    
    def save(self, path):
        torch.save(get_peft_model_state_dict(self.model), f'{path}/adapter_model.pt')

class LayerFTModel(EmbeddingModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        for param in self.model.parameters():
            param.requires_grad = False
        # TODO: generalize this to all models
        for param in self.model.roberta.encoder.layer[-self.args.ft_layers:].parameters():
            param.requires_grad = True

class ClassificationHead(nn.Module):
    def __init__(self, args, d_in, d_out):
        super().__init__()
        self.args = args
        self.dropout = nn.Dropout(args.dropout)
        self.classifier = nn.Linear(
            in_features=d_in,
            out_features=d_out,
            bias=False,
            dtype=torch.float,
        )

    def load(self, path):
        with torch.no_grad():
            self.classifier.weight.copy_(torch.load(f'{path}/classifier.pt'))

    def save(self, path):
        torch.save(self.classifier.state_dict(), f'{path}/classifier.pt')

    def forward(self, embeds):
        embeds = self.dropout(embeds)
        logits = self.classifier(embeds.to(self.classifier.weight.dtype))
        return logits

class FeatureClassifier(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.args = kwargs['args']
        self.device = self.args.device if self.args.device != 'auto' else 'cuda'
        embedder_class = kwargs.pop('embedder_class')
        self.feature_extractor = embedder_class(**kwargs)
        self.classifier = ClassificationHead(self.args,
            d_in=self.feature_extractor.embed_dim,
            d_out=len(self.args.classes)
        ).to(self.device)

    def load(self, path):
        self.feature_extractor.load(path)
        self.classifier.load(path)
        self.classifier = self.classifier.to(self.device)

    def save(self, path):
        self.feature_extractor.save(path)
        self.classifier.save(path)

    def forward(self, batch):
        embeds = self.feature_extractor(batch)
        logits = self.classifier(embeds)
        return logits
