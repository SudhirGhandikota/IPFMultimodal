import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, HGTConv, SAGEConv, Linear, GCNConv

class PreKGNetAttn(torch.nn.Module):
    def __init__(self, n_output=1, output_dim=128, num_exp=160,
                 hid_dims_exp=[1024, 512], num_emb=768, hid_dims_emb=[1024, 512],
                 dropout=0.2, num_heads=2, reduce=None, act=nn.ReLU(),
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
        super(PreKGNetAttn, self).__init__()
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

        # combined layers
        self.embed_dim = self.output_dim*3 \
            if self.reduce is None or self.reduce == 'concat' else self.output_dim

        self.fc1 = Linear(self.embed_dim, 1024)
        self.fc2 = Linear(1024, 512)
        self.out = Linear(512, self.n_outputs)

        # attention layers
        # attention of expression features on KG features
        self.exp_kg_attn = nn.MultiheadAttention(self.output_dim, num_heads=self.num_heads,
                                                 dropout=self.drop_rate,
                                                 batch_first=True)
        # expression of KG features on expression features
        self.kg_exp_attn = nn.MultiheadAttention(self.output_dim, num_heads=self.num_heads,
                                                 dropout=self.drop_rate,
                                                 batch_first=True)
        # attention of text features on KG features
        self.emb_kg_attn = nn.MultiheadAttention(self.output_dim, num_heads=self.num_heads,
                                                 dropout=self.drop_rate,
                                                 batch_first=True)
        # attention of KG features on text features
        self.kg_emb_attn = nn.MultiheadAttention(self.output_dim, num_heads=self.num_heads,
                                                 dropout=self.drop_rate,
                                                 batch_first=True)
        # attention of text features on expression features
        self.emb_exp_attn = nn.MultiheadAttention(self.output_dim, num_heads=self.num_heads,
                                                 dropout=self.drop_rate,
                                                 batch_first=True)
        # attention of expression features on text features
        self.exp_emb_attn = nn.MultiheadAttention(self.output_dim, num_heads=self.num_heads,
                                                 dropout=self.drop_rate,
                                                 batch_first=True)

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

        # attention block
        x_expr2kg, _ = self.exp_kg_attn(x_kg.unsqueeze(1), x_exprs.unsqueeze(1),
                                        x_exprs.unsqueeze(1)) # shape = (batch_size, 1, output_dim)
        x_expr2kg = x_expr2kg.squeeze(1) # shape = (batch_size, output_dim)
        x_kg2expr, _ = self.kg_exp_attn(x_exprs.unsqueeze(1), x_kg.unsqueeze(1), x_kg.unsqueeze(1))
        x_kg2expr = x_kg2expr.squeeze(1)
        x_emb2kg, _ = self.emb_kg_attn(x_kg.unsqueeze(1), x_exprs.unsqueeze(1), x_exprs.unsqueeze(1))
        x_emb2kg = x_emb2kg.squeeze(1)
        x_kg2emb, _ = self.kg_emb_attn(x_embs.unsqueeze(1), x_kg.unsqueeze(1), x_embs.unsqueeze(1))
        x_kg2emb = x_kg2emb.squeeze(1)
        x_exp2emb, _ = self.exp_emb_attn(x_embs.unsqueeze(1), x_exprs.unsqueeze(1), x_exprs.unsqueeze(1))
        x_exp2emb = x_exp2emb.squeeze(1)
        x_emb2exp, _ = self.emb_exp_attn(x_exprs.unsqueeze(1), x_embs.unsqueeze(1), x_embs.unsqueeze(1))
        x_emb2exp = x_emb2exp.squeeze(1)

        # consolidation all attention features for each modality
        x_exprs = x_exprs + x_expr2kg + x_kg2expr + x_exp2emb + x_emb2exp
        x_embs = x_embs + x_emb2kg + x_kg2emb + x_emb2exp + x_exp2emb
        x_kg = x_kg + x_kg2expr + x_expr2kg + x_kg2emb + x_emb2kg

        if self.reduce is None or self.reduce == 'concat':
            x_combined = torch.cat((x_exprs, x_embs, x_kg), 1)
        if self.reduce == 'sum':
            x_combined = x_exprs + x_embs + x_kg
        if self.reduce == 'mean':
            x_combined = torch.stack([x_exprs, x_embs, x_kg])
            x_combined = x_combined.mean(0)
        if self.reduce == 'max':
            x_combined = torch.stack([x_exprs, x_embs, x_kg])
            x_combined = x_combined.max(0)[0]

        # final layers
        x_combined = self.fc1(x_combined)
        x_combined = self.dropout(self.act(x_combined))
        x_combined = self.fc2(x_combined)
        x_combined = self.dropout(self.act(x_combined))
        out = self.out(x_combined)
        return out

    def fetch_embs(self, data):
        exprs, embs = data.exprs, data.embs

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
        x_kg = self.kg_emb(data.gene_idx)
        x_kg = self.dropout(self.kg_fc(x_kg))
        x_kg = self.act(x_kg)

        # attention block
        x_expr2kg, _ = self.exp_kg_attn(x_kg.unsqueeze(1), x_exprs.unsqueeze(1),
                                        x_exprs.unsqueeze(1))  # shape = (batch_size, 1, output_dim)
        x_expr2kg = x_expr2kg.squeeze(1)  # shape = (batch_size, output_dim)
        x_kg2expr, _ = self.kg_exp_attn(x_exprs.unsqueeze(1), x_kg.unsqueeze(1), x_kg.unsqueeze(1))
        x_kg2expr = x_kg2expr.squeeze(1)
        x_emb2kg, _ = self.emb_kg_attn(x_kg.unsqueeze(1), x_exprs.unsqueeze(1), x_exprs.unsqueeze(1))
        x_emb2kg = x_emb2kg.squeeze(1)
        x_kg2emb, _ = self.kg_emb_attn(x_embs.unsqueeze(1), x_kg.unsqueeze(1), x_embs.unsqueeze(1))
        x_kg2emb = x_kg2emb.squeeze(1)
        x_exp2emb, _ = self.exp_emb_attn(x_embs.unsqueeze(1), x_exprs.unsqueeze(1), x_exprs.unsqueeze(1))
        x_exp2emb = x_exp2emb.squeeze(1)
        x_emb2exp, _ = self.emb_exp_attn(x_exprs.unsqueeze(1), x_embs.unsqueeze(1), x_embs.unsqueeze(1))
        x_emb2exp = x_emb2exp.squeeze(1)

        # consolidation all attention features for each modality
        x_exprs = x_exprs + x_expr2kg + x_kg2expr + x_exp2emb + x_emb2exp
        x_embs = x_embs + x_emb2kg + x_kg2emb + x_emb2exp + x_exp2emb
        x_kg = x_kg + x_kg2expr + x_expr2kg + x_kg2emb + x_emb2kg

        if self.reduce is None or self.reduce == 'concat':
            x_combined = torch.cat((x_exprs, x_embs, x_kg), 1)
        if self.reduce == 'sum':
            x_combined = x_exprs + x_embs + x_kg
        if self.reduce == 'mean':
            x_combined = torch.stack([x_exprs, x_embs, x_kg])
            x_combined = x_combined.mean(0)
        if self.reduce == 'max':
            x_combined = torch.stack([x_exprs, x_embs, x_kg])
            x_combined = x_combined.max(0)[0]
        return x_combined
