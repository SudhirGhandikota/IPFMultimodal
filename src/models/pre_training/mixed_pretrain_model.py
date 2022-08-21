import torch.nn
from torch.nn import Linear
from torch_geometric.nn import SAGEConv, GATConv, GCNConv, HeteroConv

class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers=2):
        super().__init__()

        self.conv_layers = torch.nn.ModuleList()
        hidden_channels = [hidden_channels] + [out_channels]
        for idx in range(num_layers):
            conv = HeteroConv({('gene', 'g2g', 'gene'): GCNConv(-1, hidden_channels[idx]),
                               ('gene', 'genemsig', 'msig'): GATConv((-1, -1), hidden_channels[idx]),
                               ('gene', 'genereact', 'reactome'): GATConv((-1, -1), hidden_channels[idx]),
                               ('gene', 'genebp', 'bp'): SAGEConv((-1, -1), hidden_channels[idx]),
                               ('msig', 'rev_genemsig', 'gene'): GATConv((-1, -1), hidden_channels[idx]),
                               ('reactome', 'rev_genereact', 'gene'): GATConv((-1, -1), hidden_channels[idx]),
                               ('bp', 'rev_genebp', 'gene'): SAGEConv((-1, -1), hidden_channels[idx])},
                              aggr='sum')
            self.conv_layers.append(conv)

    def forward(self, x_dict, edge_index_dict):
        # x_dict, edge_index_dict = data.x_dict, data.edge_index_dict
        for layer in self.conv_layers:
            x_dict = layer(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}

        return x_dict

class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels, relations):
        # relations -> edge types
        super(EdgeDecoder, self).__init__()
        self.relations = relations
        self.relations2idx = {relation:idx for idx, relation in enumerate(relations)}

        self.decoder = torch.nn.ModuleDict({
            rel[0]+'_'+rel[2]: torch.nn.Sequential(
                Linear(2*hidden_channels, hidden_channels),
                torch.nn.ReLU(),
                Linear(hidden_channels, 1)
            ) for rel in self.relations
        })

    def forward(self, z_dict, edge_label_index_dict):
        z = {}
        for key, edge_label_index in edge_label_index_dict.items():
            dec_rel2idx = key[0] + '_' + key[2]
            row, col = edge_label_index
            z[key] = torch.cat([z_dict[key[0]][row],
                                z_dict[key[2]][col]], dim = 1)
            z[key] = self.decoder[dec_rel2idx](z[key])
        return z

class PretrainGNN(torch.nn.Module):
    def __init__(self, hidden_channels, relations, data):
        super(PretrainGNN, self).__init__()
        self.encoder = GNNEncoder(hidden_channels, hidden_channels)
        self.decoder = EdgeDecoder(hidden_channels, relations)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        # edge_index_dict -> network edges used for encoding nodes
        # edge_label_index -> edges (positive+negative) used for link prediction
        z_dict = self.encoder(x_dict, edge_index_dict)
        return z_dict, self.decoder(z_dict, edge_label_index)

