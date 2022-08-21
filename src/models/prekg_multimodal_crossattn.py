import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, HGTConv, SAGEConv, Linear, GCNConv

class PreKGNetCrossAttn(torch.nn.Module):
    def __init__(self, n_output=1, output_dim=128, num_exp=160,
                 hid_dims_exp=[1024, 512], num_emb=768, hid_dims_emb=[1024, 512],
                 dropout=0.2, num_heads=4, reduce=None, act=nn.ReLU(),
                 gene_hetero_emb=None):
        """
        :param n_output: output dimension i.e., num of classes
        :param output_dim: Final feature dimension
        :param num_exp: dimensionality of expression features
        :param hid_dims_exp: hidden layer weight dimensions for expression block
        :param num_emb: dimensionality of text features
        :param hid_dims_emb:hidden layer weight dimensions for text block
        :param dropout: dropout rate
        :param concat: boolean to indicate if features from each datatype
        must be concatenated or not (will be added if FALSE)
        :param act: non-linear activation
        :param gene_hetero_emb: gene embeddings from pre-trained heterogeneous graph
        """
        super(PreKGNetCrossAttn, self).__init__()
        self.n_outputs = n_output
        self.output_dim = output_dim
        self.reduce = reduce
        self.act = act
        self.drop_rate = dropout
        self.dropout = nn.Dropout(dropout)
        self.num_heads = num_heads

        self.hid_dims_exp = [num_exp] + hid_dims_exp + [self.output_dim]
        self.hid_dims_emb = [num_emb] + hid_dims_emb + [self.output_dim]

        # pre-trained gene embeddings from KG
        self.kg_emb = nn.Embedding.from_pretrained(gene_hetero_emb)
        self.kg_emb_dim = gene_hetero_emb.shape[1]
        self.kg_fc = nn.Linear(self.kg_emb_dim, self.output_dim)

        # Expression block
        self.exp_fcs = nn.ModuleList()
        for idx in range(1, len(self.hid_dims_exp)):
            self.exp_fcs.append(Linear(self.hid_dims_exp[idx-1],
                                       self.hid_dims_exp[idx]))

        # Text Embedding block
        self.emb_fcs = nn.ModuleList()
        for idx in range(1, len(self.hid_dims_emb)):
            self.emb_fcs.append(Linear(self.hid_dims_emb[idx-1],
                                       self.hid_dims_emb[idx]))

        # cross-attention
        self.cross_attn = nn.MultiheadAttention(self.output_dim, num_heads=self.num_heads,
                                                dropout=self.drop_rate, batch_first=True)

        # combined layers
        self.fc1 = Linear(self.output_dim*3, self.output_dim*3)
        self.fc2 = Linear(self.output_dim*3, self.output_dim*2)
        self.fc3 = Linear(self.output_dim*2, self.output_dim)
        self.out = Linear(self.output_dim, self.n_outputs)


    def forward(self, batch_data):
        exprs, embs, batch = batch_data.exprs, batch_data.embs, batch_data.batch

        # expression block
        x_exprs = exprs
        for fc in self.exp_fcs:
            x_exprs = self.dropout(fc(x_exprs))
            x_exprs = self.act(x_exprs)

        # embedding block
        x_embs = embs
        for fc in self.emb_fcs:
            x_embs = self.dropout(fc(x_embs))
            x_embs = self.act(x_embs)

        # pretrained KG block
        x_kg = self.kg_emb(batch_data.gene_idx)
        x_kg = self.dropout(self.kg_fc(x_kg))
        x_kg = self.act(x_kg)

        # attention part
        x_combined = torch.cat((x_exprs.unsqueeze(1), x_embs.unsqueeze(1), x_kg.unsqueeze(1)), 1) # x_combined = (batch_size, 3, self.output_dim)
        attn_outs, _ = self.cross_attn(x_combined, x_combined, x_combined)
        # attn_outs = (batch_size, 3, self.output_dim)
        x_combined = x_combined + attn_outs
        x_combined = x_combined.view(x_combined.size(0), -1) # (batch_size, 3*self.output_dim)

        # final layers
        x_combined = self.fc1(x_combined)
        x_combined = self.dropout(self.act(x_combined))
        x_combined = self.fc2(x_combined)
        x_combined = self.dropout(self.act(x_combined))
        x_combined = self.fc3(x_combined)
        x_combined = self.dropout(self.act(x_combined))
        out = self.out(x_combined)
        return out

