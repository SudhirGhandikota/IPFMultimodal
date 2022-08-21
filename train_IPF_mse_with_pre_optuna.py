import time
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from src.utils.dataloader_IPF_pre import *
from torch.utils.tensorboard import SummaryWriter
from src.models.prekg_multimodal import PreKGNet
from src.models.prekg_multimodal_attn import PreKGNetAttn
from src.models.prekg_multimodal_crossattn import PreKGNetCrossAttn

from src.metrics import eval_mse_model
import optuna
from optuna.trial import TrialState
from optuna.samplers import TPESampler

parser = argparse.ArgumentParser()
parser.add_argument('--num_trials', type=int, default=100,
                    help='Max number of trials')
parser.add_argument('--num_epochs', type=int, default=100,
                    help='Max number of epochs')
parser.add_argument('--model', type=str, default='KGNet',
                    choices=['KGNet', 'KGNetAttn', 'KGNetCrossAttn'],
                    help='Model name')
parser.add_argument('--reduce', type=str, default=None,
                    choices=['sum', 'max', 'mean'],
                    help='Feature aggregation method')
parser.add_argument('--save_path', type=str, default='/',
                    help='Path to save model checkpoints and metrics')
parser.add_argument('--indir', type=str,
                    help='Path of training data')
parser.add_argument('--emb_file', type=str,
                    help='Path of pre-trained embeddings')
args = parser.parse_args()

def load_data(root):
    #train_data = IPFDataset(root=root, infile='train.npy')
    train_data = IPFDataset(root=root, infile='train_sample.npy')
    train_loader = DataLoader(train_data, batch_size=256, shuffle=False)
    train_data = None

    #val_data = IPFDataset(root=root, infile='val.npy')
    val_data = IPFDataset(root=root, infile='val_sample.npy')
    val_loader = DataLoader(val_data, batch_size=256, shuffle=False)
    val_data = None

    #test_data = IPFDataset(root=root, infile='test.npy')
    test_data = IPFDataset(root=root, infile='test_sample.npy')
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False)
    test_data = None

    return train_loader, val_loader, test_loader

def load_model(params):
    pre_gene_embedding = np.load(args.emb_file)
    pre_gene_embedding = torch.from_numpy(pre_gene_embedding)

    if args.model == 'KGNet':
        model = PreKGNet(gene_hetero_emb=pre_gene_embedding, **params)
    elif args.model == 'KGNetAttn':
        model = PreKGNetAttn(gene_hetero_emb=pre_gene_embedding, **params)
    elif args.model == 'KGNetCrossAttn':
        model = PreKGNetCrossAttn(gene_hetero_emb=pre_gene_embedding, **params)
    return model

def train_mse(trial, train_loader, val_loader, test_loader, num_epochs=100,
              lr=1e-3, hid_dims_exp=[1024,512], hid_dims_emb=[1024, 512], save_dir=None,
              reduce=None, epoch_patience=10, dropout=0.1, decay=0.01, act=nn.ReLU()):
    print('*'*5, 'Model Name:', model_name, '*'*5)
    start_time = time.time()

    params = {'hid_dims_exp': hid_dims_exp,
              'hid_dims_emb': hid_dims_emb,
              'act': act,
              'reduce': reduce,
              'dropout': dropout}

    model = load_model(params)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
    num_steps = len(train_loader)*num_epochs
    print(f'Number of batches: {len(train_loader)}, Number of epochs: {num_epochs}'
          f' Total Steps: {num_steps}')

    comment = f'{model_name}, LR={lr}, Epochs={num_epochs}'
    writer = SummaryWriter(comment=comment)

    model.train()
    losses = []
    best_epoch, best_loss, best_mse = 0, np.inf, np.inf
    pbar = tqdm(range(num_epochs), position=1)
    patience = 0
    stop_train = False
    # creating save path if not done already
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    best_model = None

    sample_scores = {}
    for epoch in pbar:
        pbar.set_description(f'Epoch: {epoch}/{num_epochs}')
        #pbar2 = tqdm(train_loader, position=0, desc='Training')
        total_loss = 0
        for idx, batch in enumerate(train_loader):
            batch = batch.to(device)
            optimizer.zero_grad() # removing any existing gradients
            scores = model(batch)
            sample_scores[f'{epoch}_{idx}'] = scores[0:10]
            batch_labels = batch.y.float().to(device)
            if len(batch_labels.shape) == 1:
                scores = scores.squeeze(1)
            loss = criterion(scores, batch_labels)
            total_loss += loss.item()
            losses.append(loss.item())

            loss.backward() # computing gradients
            optimizer.step() # backpropagation

        # computing test metrics
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        val_metrics= eval_mse_model(model, val_loader, criterion, device)
        trial.report(val_metrics['mse'], epoch)

        if val_metrics['mse'] < best_mse:
            # resetting patience parameters
            patience=0
            best_epoch=epoch
            best_mse=val_metrics['mse']
            best_loss = val_metrics['avg_loss']
            save_path=f'{save_dir}/models/{args.model}_{trial.number}_checkpoint.model'
            torch.save(model, save_path)
            best_model = save_path
        else:
            patience += 1
            if patience > epoch_patience:
                stop_train=True

        if stop_train:
            break

    model = torch.load(best_model)
    model = model.to(device)
    test_metrics = eval_mse_model(model, test_loader, criterion, device)
    test_metrics['model_state'] = best_model
    print(f'\nTest Loss: {test_metrics["avg_loss"]:.4f}, '
          f' Test MSE: {test_metrics["mse"]:.3f}, '
          f' Correlation: {test_metrics["cor"]:.3f} ({test_metrics["cor_pval"]:.3e}),'
          f' Concordance Score: {test_metrics["concordance_score"]}')

    return test_metrics['mse']

def objective(trial):
    # generating trial parameters
    num_epochs = args.num_epochs
    lr = trial.suggest_loguniform('lr', 1e-3, 1e-1)
    dropout = trial.suggest_float('dropout', 0.0, 0.6)
    decay = trial.suggest_loguniform('decay', 1e-3, 1e-1)

    # number of hidden layers
    n_layers_exp = trial.suggest_int('n_layers_exp', 1, 3)
    n_layers_emb = trial.suggest_int('n_layers_emb', 1, 3)
    hid_dims_exp, hid_dims_emb = [], []
    # generating hidden layer dimension
    for i in range(n_layers_exp):
        hid_dim = trial.suggest_int(f'n_units_exp_l{i}',128, 1024)
        hid_dims_exp.append(hid_dim)
    for i in range(n_layers_emb):
        hid_dim = trial.suggest_int(f'n_units_emb_l{i}',128, 1024)
        hid_dims_emb.append(hid_dim)

    train_loader, val_loader, test_loader = load_data(indir)
    metrics = train_mse(trial, train_loader, val_loader, test_loader,
                        num_epochs=num_epochs, hid_dims_exp=hid_dims_exp, hid_dims_emb=hid_dims_emb,
                        save_dir=save_path, reduce=reduce, epoch_patience=10, lr=lr, decay=decay, dropout=dropout, act=nn.ReLU())
    # Handle pruning based on the intermediate value.
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()
    return metrics

if __name__ == '__main__':
    model_name = args.model
    save_path = args.save_path
    model_dir = f'{save_path}/models/'
    emb_file = args.emb_file
    indir = args.indir
    reduce = args.reduce

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    train_loader, val_loader, test_loader = load_data(indir)

    study_name = emb_file.split('/')[-1].split('.')[0]
    study_name = f'trials_{study_name}' # "trials_pre_dim_32_neg_10"
    storage_name = f'sqlite:///{save_path}/{study_name}.db'

    # to reproduce the results https://optuna.readthedocs.io/en/stable/faq.html
    torch.random.manual_seed(10)
    sampler = TPESampler(seed=10)
    study = optuna.create_study(direction="minimize", storage=storage_name,
                                sampler=sampler)
    study.optimize(objective, n_trials=args.num_trials, timeout=None,
                   n_jobs=1, gc_after_trial=True)
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print('*'*10, 'Trial Stats', '*'*10)
    print(f'Number of finished trials: {len(study.trials)}')
    print(f'Number of pruned trials: {len(pruned_trials)}')
    print(f'Number of complete trials: {len(complete_trials)}')

    print('*'*10, 'Best trial', '*'*10)
    best_trial = study.best_trial
    print("Params:")

    for key, val in best_trial.params.items():
        print(f'Parameter: {key}, Value: {val}')

    # saving the best model
    path = f'{model_dir}/{args.model}_{best_trial.number}_checkpoint.model'
    model = torch.load(path)
    # resaving the model
    save_path = f'{model_dir}/{args.model}_checkpoint.model'
    torch.save(model, save_path)

## Usage:
#       python train_IPF_mse_with_pre_optuna.py --model KGNet --save_path saved/training_optuna/gold --indir data/processed/ --emb_file saved/pretrain_attn/gold/embedding_dim_32_neg_10.0.npy --num_trials 1000