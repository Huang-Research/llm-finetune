def main():
    import argparse, pickle
    from pprint import pprint

    parser = argparse.ArgumentParser()

    parser.add_argument('--classes', type=str, default='RED,PCE,DWED', help='Comma separated list of datasets to use')
    parser.add_argument('--model', type=str, default="deepseek-ai/deepseek-coder-6.7b-base", help='Model name')
    parser.add_argument('--epoch', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=0, help='Warmup steps')
    parser.add_argument('--lora_r', type=int, default=8, help='Lora rank')
    parser.add_argument('--lora_alpha', type=int, default=32, help='Lora alpha')
    parser.add_argument('--max_seq_len', type=int, default=2048, help='Max sequence length')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--val_interval', type=int, default=1, help='Validation frequency')
    parser.add_argument('--profile', action='store_true', help='Enable profiling')
    parser.add_argument('--scheduler', type=str, default='linear', help='Scheduler to use')
    parser.add_argument('--load', type=str, default="", help='Path to load/resume from')
    parser.add_argument('--infer', action='store_true', help='Run inference only')
    parser.add_argument('--salience', type=int, help='Index of salience map', default=None)
    parser.add_argument('--test', action='store_true', help='Run test dataset')
    parser.add_argument('--encoder', action='store_true', help='Specify model is an encoder')
    parser.add_argument('--requirements', action='store_true', help='Insert requirements into the prompt')
    parser.add_argument('--trim', action='store_true', help='Trim the input text')
    parser.add_argument('--bidirectional', action='store_true', help='Use bidirectional attention')
    parser.add_argument('--only_jiong', action='store_true', help='Use only Jiong data')

    args = parser.parse_args()
    args.classes = args.classes.split(',')
    assert len(args.classes) > 0, "At least one class must be specified"

    if args.load:
        # set default values to the saved arguments
        last_args = pickle.load(open(f"{args.load}/args.pkl", "rb"))
        parser.set_defaults(**vars(last_args))
        # need to reparse to get the new defaults
        args = parser.parse_args()
        print("Loaded args:")
        pprint(vars(args), indent=4, sort_dicts=False)
    
    if args.infer:
        infer(args)
    else:
        train(args)

def train(args):
    import wandb
    
    a = Agent(args)
    wandb.login(key='9291a5186a8cbb6815a7be81e224c73a70504e20')
    run = wandb.init(entity='faultdiagnosis', project='faultdiagnosis',
              config={
                    "model": args.model,
                    "classes": args.classes,
                    "epoch": args.epoch,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "warmup_steps": args.warmup_steps,
                    "lora_r": args.lora_r,
                    "lora_alpha": args.lora_alpha,
                    "max_seq_len": args.max_seq_len,
                    "device": args.device,
                    "val_interval": args.val_interval,
                    "profile": args.profile,
                    "scheduler": args.scheduler,
                    "load": args.load,
                    "infer": args.infer,
                    "salience": args.salience,
                    "test": args.test,
                    "encoder": args.encoder,
                    "requirements": args.requirements,
                    "trim": args.trim,
                    "bidirectional": args.bidirectional,
                    "only_jiong": args.only_jiong,
                  })

    a.train()
    run.finish()

def infer(args):
    import json

    a = Agent(args)
    
    targets = [i['target'] for i in a.val_dataset]
    texts = [i['text'] for _, i in a.val_dataset.df.iterrows()]

    if args.salience:
        print("Generating salience...")
        salience = a.generate_salience(texts[args.salience], targets[args.salience])
        np.save('output/salience.npy', salience)
    else:
        print("Predicting...")
        text_tokenized, preds, probs = a.predict(texts, targets)
        
        items = zip(targets, text_tokenized, preds, probs)
        json.dump(list(map(lambda x: {
            'target': x[0].tolist(),
            'text': x[1],
            'pred': x[2].tolist(),
            'probs': x[3].tolist(),
        }, items)), open(f'output/predictions.json', 'w'))

if __name__ == "__main__":
    import os
    
    os.environ["HF_HOME"] = os.environ.get("HF_HOME", default="/research/huang/workspaces/hytopot/faultdiagnosis/.hf")
    print(f"HF_HOME set to: {os.environ['HF_HOME']}")  # Debugging line to check HF_HOME path

    import torch
    import numpy as np

    torch.random.manual_seed(42)
    np.random.seed(42)

    from agent import Agent

    main()
