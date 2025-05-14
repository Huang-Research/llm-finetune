def main():
    import argparse, pickle
    from pprint import pprint
    import os
    import torch
    import numpy as np

    torch.random.manual_seed(42)
    np.random.seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--HF_HOME', type=str, default=None, help='Huggingface cache directory')
    parser.add_argument('--WANDB_KEY', type=str, default=None, help='Wandb key')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--profile', action='store_true', help='Enable profiling')

    # DATASET
    parser.add_argument('--classes', type=str, default='RED,PCE,DWED', help='Comma separated list of datasets to use')
    parser.add_argument('--requirements', action='store_true', help='Insert requirements into the prompt')
    parser.add_argument('--trim', action='store_true', help='Trim the input text')
    parser.add_argument('--only_jiong', action='store_true', help='Use only Jiong data')
    parser.add_argument('--max_seq_len', type=int, default=2048, help='Max sequence length')
    parser.add_argument('--test', action='store_true', help='Run test dataset')

    # MODEL
    parser.add_argument('--model', type=str, default="microsoft/codebert-base", help='Model name')
    parser.add_argument('--encoder', action='store_true', help='Specify model is an encoder')
    # parser.add_argument('--bidirectional', action='store_true', help='Use bidirectional attention')
    parser.add_argument('--val_interval', type=int, default=1, help='Validation frequency')
    parser.add_argument('--ft_layers', type=int, default=0, help='Number of layers to fine-tune')

    # PARAMETERS
    parser.add_argument('--epoch', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--lr_clf', type=float, default=1e-5, help='Classifier learning rate')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=0, help='Warmup steps')
    parser.add_argument('--lora_r', type=int, default=8, help='Lora rank')
    parser.add_argument('--lora_alpha', type=int, default=32, help='Lora alpha')
    parser.add_argument('--scheduler', type=str, default='linear', help='Scheduler to use')
    parser.add_argument('--chunk_size', type=int, default=2048, help='Chunk size for chunking')
    parser.add_argument('--chunk_stride', type=int, default=256, help='Chunk stride for chunking')

    # EVAL
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the model (inference)')
    parser.add_argument('--load', type=str, default="", help='Path to load/resume from')
    parser.add_argument('--salience', type=int, help='Index of salience map', default=None)

    args = parser.parse_args()

    if args.HF_HOME:
        os.environ["HF_HOME"] = args.HF_HOME
        print(f"HF_HOME set to: {os.environ['HF_HOME']}")
    if args.WANDB_KEY:
        os.environ["WANDB_API_KEY"] = args.WANDB_KEY
        print(f"WANDB_API_KEY set to: {os.environ['WANDB_API_KEY']}")
    if args.classes:
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
    if args.evaluate:
        evaluate(args)
    else:
        train(args)

def train(args):
    from agent import Agent
    import wandb

    wandb_config = {
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
        "salience": args.salience,
        "test": args.test,
        "encoder": args.encoder,
        "requirements": args.requirements,
        "trim": args.trim,
        "only_jiong": args.only_jiong,
        "chunk_size": args.chunk_size,
        "chunk_stride": args.chunk_stride,
        "lr_clf": args.lr_clf,
        "dropout": args.dropout,
        "ft_layers": args.ft_layers,
    }

    a = Agent(args)
    run = wandb.init(entity='faultdiagnosis', project='faultdiagnosis', config=wandb_config)

    a.train()
    run.finish()

def evaluate(args):
    from agent import Agent
    from dataset import HEDataset
    from evaluate import (
        compute_baseline,
        compute_roc_auc_table,
        compute_pr_auc_table,
        find_best_thresholds,
        evaluate_with_thresholds,
    )

    a = Agent(args)
    test_dataset = HEDataset('test.pkl', a.tokenizer, args, split='test')
    val_df = a.evaluate(a.val_dataset.df, args.salience)
    test_df = a.evaluate(test_dataset.df, args.salience)

    print("Baseline performance metrics:")
    baseline_metrics = compute_baseline(test_df)
    for class_id, metrics in baseline_metrics.items():
        print(f"{class_id}: F1: {metrics['F1']:.3f}, Precision: {metrics['Precision']:.3f}, Recall: {metrics['Recall']:.3f}")
    
    print(compute_roc_auc_table(test_df).round(3))
    print(compute_pr_auc_table(test_df).round(3))
    
    thresholds = find_best_thresholds(val_df)
    print(f'Val dataset thresholds: {thresholds}')
    f1_scores, precision_scores, recall_scores = evaluate_with_thresholds(test_df, thresholds)
    print(f'Test dataset F1 scores: {f1_scores}')
    print(f'Test dataset precision scores: {precision_scores}')
    print(f'Test dataset recall scores: {recall_scores}')

if __name__ == "__main__":
    main()
