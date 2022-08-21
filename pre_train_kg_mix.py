import argparse
import os
import numpy as np
import torch
from src.models.pre_training.mixed_pretrain_model import PretrainGNN
#from data_util_kg import get_kg_data
from src.utils.data_utils import get_kg
import torch_geometric.transforms as T
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm
import nvidia_smi

def get_memory_info():
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    memory_info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    total_memory = round(((memory_info.total/1024)/1024)/1024,3)
    free_memory = round(((memory_info.free / 1024) / 1024) / 1024,3)
    used_memory = round(((memory_info.used / 1024) / 1024) / 1024,3)
    print("*"*5, "Total Memory (GB):", total_memory, "Used Memory (GB):",
          used_memory, "Free Memory (GB):", free_memory, "*"*5)

torch.manual_seed(0)
parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=100,
                    help='Max Epochs')
parser.add_argument('--dim_size', type=int, default=32,
                    help='Embedding Size')
parser.add_argument('--prop_val', type=float, default=0.2,
                    help='Proportion of validation edges')
parser.add_argument('--prop_test', type=float, default=0.2,
                    help='Proportion of test edges')
parser.add_argument('--num_neg', type=float, default=10.0,
                    help='Number of negative edges for each positive edge')
parser.add_argument('--outdir', type=str, default='gold',
                    choices=['gold', 'top_25', 'top_3'],
                    help='Network name')
parser.add_argument('--patience', type=int, default=10,
                    help='Early stopping tolerance')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
network_name = args.outdir
#dataset = get_kg(network_name=network_name)
dataset = get_kg(network_name=network_name).to(device)
print(dataset)

for key in dataset.metadata()[1]:
    if 'rev' in key[1]:
        del dataset[key].edge_label

edge_types = [key for key in dataset.metadata()[1] if 'rev' not in key[1]]
rev_edge_types = [key for key in dataset.metadata()[1] if
                  'rev' in key[1] or key[0] == key[2]]
print(edge_types, rev_edge_types)

train_data, val_data, test_data = T.RandomLinkSplit(
    num_val=args.prop_val,
    num_test=args.prop_test,
    neg_sampling_ratio=args.num_neg,
    edge_types=edge_types,
    rev_edge_types=rev_edge_types,

)(dataset)

relations = [key for key in dataset.metadata()[1] if 'rev' not in key[1]]

hidden_channels = args.dim_size
model = PretrainGNN(hidden_channels=hidden_channels, relations=relations,
                    data=dataset)
model = model.to(device)

dataset=None
# Due to lazy initialization, we need to run one model step so the number
# of parameters can be inferred:
#train_data.to(device) # loading the training data onto the GPU
with torch.no_grad():
    model.encoder(train_data.x_dict, train_data.edge_index_dict)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.BCEWithLogitsLoss()

def compute_loss(pred, target):
    loss = 0
    for key, value in pred.items():
        y_hat = pred[key].squeeze(1)
        y = target[key]
        loss += loss_fn(y_hat, y)
    return loss

@torch.no_grad()
def compute_scores(pred, target):
    aupr, auroc = 0, 0
    for key, value in pred.items():
        y_hat = pred[key].sigmoid().squeeze(1).cpu().numpy()
        y = target[key].cpu().numpy().astype(int)
        aupr += average_precision_score(y, y_hat)
        auroc += roc_auc_score(y, y_hat)


    aupr /= len(pred)
    auroc /= len(pred)
    return aupr, auroc

# training method
def train():
    model.train()
    optimizer.zero_grad()
    # filtering out reverse (link-prediction) edges
    # edge_label_index -> link-prediction edges
    # train_data.edge_index_dict -> network edges for training
    edge_label_index = {rel: train_data[rel[0], rel[2]].edge_label_index
                        for rel in relations}
    _, pred = model(train_data.x_dict, train_data.edge_index_dict, edge_label_index)
    # pred -> link-prediction scores
    target = {rel: train_data[rel[0], rel[2]].edge_label.float() for rel in relations}
    # target -> link-prediction labels
    loss = compute_loss(pred, target)
    loss.backward()
    optimizer.step()
    edge_label_index = None
    target=None
    pred=None
    optimizer.zero_grad(set_to_none=True)
    return float(loss)

@torch.no_grad()
def test(data):
    model.eval()
    edge_label_index = {rel: data[rel[0], rel[2]].edge_label_index for rel in relations}
    # edge_label_index -> link-prediction edges
    # data.edge_index_dict -> network edges for testing
    _, pred = model(data.x_dict, data.edge_index_dict,
                 edge_label_index)
    target = {rel: data[rel[0], rel[2]].edge_label.float() for rel in relations}
    loss = compute_loss(pred, target)
    aupr, auroc = compute_scores(pred, target)
    pred=None
    target=None
    edge_label_index=None
    return float(loss), float(aupr), float(auroc)

save_dir = f'saved/pretrain_mix/{args.outdir}/'
if not os.path.exists(save_dir): os.makedirs(f'saved/pretrain_mix/{args.outdir}/')

num_epochs = args.num_epochs
val_auprs, val_aurocs = [], []
test_auprs, test_aurocs = [], []
best_epoch, best_aupr, best_auroc = 0, -np.inf, -np.inf
patience = 0
best_model = None
stop_flag = False

pbar = tqdm(range(1, num_epochs+1), position=1)
for epoch in pbar:
    pbar.set_description(f'Epoch: {epoch}/{num_epochs} ')
    loss = train()
    #loss = batch_train()
    train_loss, train_aupr, train_auroc = test(train_data)
    val_loss, val_aupr, val_auroc = test(val_data)
    val_auprs.append(val_aupr)
    val_aurocs.append(val_auroc)
    test_loss, test_aupr, test_auroc = test(test_data)
    test_auprs.append(test_aupr)
    test_aurocs.append(test_auroc)
    print(f' Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_loss:.4f}, '
          f'Val: {val_loss:.4f}, Test: {test_loss:.4f}')
    print(f'AUPR: Train: {train_aupr:.4f}, '
          f'Val: {val_aupr:.4f}, Test: {test_aupr:.4f}')
    print(f'AUROC: Train: {train_auroc:.4f}, '
          f'Val: {val_auroc:.4f}, Test: {test_auroc:.4f}')
    print('='*20)
    if val_aupr > best_aupr:
        # resetting patience parameters
        patience = 0
        best_epoch = epoch
        best_aupr = val_aupr
        best_auroc = val_auroc
        save_path = f'{save_dir}/checkpoint_dim_{hidden_channels}' \
                    f'_epochs_{num_epochs}_neg_{int(args.num_neg)}.model'
        #torch.save(model, save_path)
        best_model = save_path
    else:
        patience += 1
        if patience > args.patience:
            stop_flag = True
        print(f'Patience: {patience}')

    if stop_flag:
        break

print(f'Best Epoch: {best_epoch}, Best AUPR: {best_aupr:.4f}, '
      f'Best AUROC: {best_auroc:.4f}, Model saved to {save_path}')

# saving the metrics
np.savez(os.path.join(save_dir, f'metrics_dim_{hidden_channels}_epochs_{num_epochs}'
                                f'_neg_{int(args.num_neg)}.npz'),
         val_auprs=val_auprs, val_aurocs=val_aurocs,
         test_auprs=test_auprs, test_aurocs=test_aurocs)

# nohup python -u pre_train_kg_mix.py --dim_size 32 --num_neg 10.0 --num_epochs 500
# --prop_val 0.2 --prop_test 0.2 --patience 10 --outdir gold > nohup.out &
