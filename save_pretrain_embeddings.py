import torch
from src.utils.data_utils import get_kg
import torch_geometric.transforms as T
import numpy as np
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=100,
                    help='Max Epochs')
parser.add_argument('--dim_size', type=int, default=32,
                    help='Embedding Size')
parser.add_argument('--num_neg', type=float, default=10.0,
                    help='Number of negative edges for each positive edge')
parser.add_argument('--outdir', type=str, default='gold',
                    choices=['gold', 'top_25', 'top_3'],
                    help='Network name')
parser.add_argument('--attn', action='store_true',
                    help='Boolean flag to indicate if attentive models were used for Unsupervised KG pretraining')
args = parser.parse_args()
print(args)

network_name = args.outdir
#dataset = get_kg(network_name=network_name)
dataset = get_kg(network_name=network_name).to(device)

# doing the split step to generate sample data for running the model
# need a smaller training data for memory issues
for key in dataset.metadata()[1]:
    if 'rev' in key[1]:
        del dataset[key].edge_label

edge_types = [key for key in dataset.metadata()[1] if 'rev' not in key[1]]
rev_edge_types = [key for key in dataset.metadata()[1] if 'rev' in key[1] or key[0] == key[2]]
print(edge_types, rev_edge_types)

train_data, _, _ = T.RandomLinkSplit(
    num_val=0.3,
    num_test=0.3,
    neg_sampling_ratio=1.0,
    edge_types=edge_types,
    rev_edge_types=rev_edge_types,

)(dataset)

keys = [key for key in dataset.metadata()[1] if key[0] == 'gene' and key[2]=='gene']

model_dir = f'saved/pretrain/{args.outdir}/' if not args.attn else \
    f'saved/pretrain_attn/{args.outdir}/'
model_name = f'{model_dir}/checkpoint_dim_{args.dim_size}_epochs_{args.num_epochs}' \
             f'_neg_{int(args.num_neg)}.model'
print(model_name)
model = torch.load(model_name)

# method used to retrieve sample edges centered around a given node
# these edges used for link-prediction
def get_edge_sample(node_idx=0):
    edge_sample_dict = {}
    for key in keys:
        edge_sample = train_data.edge_index_dict[key]
        idx_ = torch.where(edge_sample[0] == node_idx)[0]
        edge_sample = edge_sample[:,idx_]
        edge_sample_dict[key] = edge_sample

    return edge_sample_dict

embedding, _ = model(train_data.x_dict, train_data.edge_index_dict, get_edge_sample(0))
embedding = embedding['gene'].detach().cpu().numpy()
print(f'Embedding shape: {embedding.shape}')
outdir = f'saved/pretrain/{args.outdir}' if not args.attn else \
    f'saved/pretrain_attn/{args.outdir}'
outfile = f'{outdir}/embedding_dim_{args.dim_size}_neg_{args.num_neg}.npy'
print(f'Outfile: {outfile}')
with open(outfile, 'wb') as f:
    np.save(f, embedding)

print('saved embedding!')

# Usage:
#   python save_pretrain_embeddings.py --outdir gold --dim_size 32 --num_neg 10 --num_epochs 100
#   python save_pretrain_embeddings.py --outdir gold --dim_size 32 --num_neg 10 --num_epochs 100 --attn







