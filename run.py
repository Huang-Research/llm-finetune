import torch
import numpy as np
from deepseek import Agent

# a = agent.Agent()
# a.train()
# # a.load_model('best_model.pt')
# a.dump_val_preds()

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

torch.random.manual_seed(42)
np.random.seed(42)

a = Agent()

for a.epoch in range(a.num_epochs):
    a.train_one_epoch()
    a.validate()

    ax, fig = plt.subplots(1, 2, figsize=(12, 4))
    ax2 = fig[0].twinx()
    ax2.plot(a.train_metrics['lr'], color='black', alpha=0.3)
    ax2.set_ylabel('lr')
    fig[0].set_xlim(0, a.num_epochs * len(a.train_loader))
    fig[0].plot(a.train_metrics['loss'], label='train')
    fig[0].plot(a.val_metrics['loss'], label='val')
    fig[0].set_title('Loss')
    fig[0].legend()

    fig[1].set_xlim(0, a.num_epochs * len(a.train_loader))
    fig[1].set_ylim(0, 1)
    # fig[1].plot(list(map(lambda x: x['0']['f1-score'], a.train_metrics['report_dict'])), c='lightred')
    # fig[1].plot(list(map(lambda x: x['1']['f1-score'], a.train_metrics['report_dict'])), c='red')
    # fig[1].plot(list(map(lambda x: x['2']['f1-score'], a.train_metrics['report_dict'])), c='darkred')
    # fig[1].plot(list(map(lambda x: x['0']['f1-score'], a.train_metrics['report_dict'])), c='lightblue')
    # fig[1].plot(list(map(lambda x: x['1']['f1-score'], a.train_metrics['report_dict'])), c='blue')
    # fig[1].plot(list(map(lambda x: x['2']['f1-score'], a.train_metrics['report_dict'])), c='darkblue')
    # fig[1].plot(list(map(lambda x: x['0']['precision'], a.train_metrics['report_dict'])), c='r')
    # fig[1].plot(list(map(lambda x: x['0']['recall'], a.train_metrics['report_dict'])), c='darkred')
    # fig[1].plot(list(map(lambda x: x['1']['precision'], a.train_metrics['report_dict'])), c='g')
    # fig[1].plot(list(map(lambda x: x['1']['recall'], a.train_metrics['report_dict'])), c='darkgreen')
    # fig[1].plot(list(map(lambda x: x['2']['precision'], a.train_metrics['report_dict'])), c='b')
    # fig[1].plot(list(map(lambda x: x['2']['recall'], a.train_metrics['report_dict'])), c='darkblue')
    # fig[1].plot(list(map(lambda x: x['0']['precision'], a.val_metrics['report_dict'])), c='r', linestyle='--')
    # fig[1].plot(list(map(lambda x: x['0']['recall'], a.val_metrics['report_dict'])), c='darkred', linestyle='--')
    # fig[1].plot(list(map(lambda x: x['1']['precision'], a.val_metrics['report_dict'])), c='g', linestyle='--')
    # fig[1].plot(list(map(lambda x: x['1']['recall'], a.val_metrics['report_dict'])), c='darkgreen', linestyle='--')
    # fig[1].plot(list(map(lambda x: x['2']['precision'], a.val_metrics['report_dict'])), c='b', linestyle='--')
    # fig[1].plot(list(map(lambda x: x['2']['recall'], a.val_metrics['report_dict'])), c='darkblue', linestyle='--')
    fig[1].plot(a.train_metrics['acc'], label='train')
    fig[1].plot(a.val_metrics['acc'], label='val')
    fig[1].set_title('Accuracy')
    # lines = [Line2D([0], [0], color='black'), Line2D([0], [0], color='black', linestyle='--'),
    #         Line2D([0], [0], color='red'), Line2D([0], [0], color='darkred')]
    # fig[1].legend(lines, ['Train', 'Val', 'Precision', 'Recall'])
    plt.savefig(f'output/train_val_{a.epoch}.png')
    plt.close()

    with open(f'output/metrics_train_{a.epoch}.txt', 'w') as f:
        f.write(str(a.train_metrics))
    with open(f'output/metrics_val_{a.epoch}.txt', 'w') as f:
        f.write(str(a.val_metrics))
