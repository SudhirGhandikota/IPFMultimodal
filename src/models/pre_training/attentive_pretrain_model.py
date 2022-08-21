import torch.nn
from torch.nn import Linear
from torch_geometric.nn import SAGEConv, GATConv, GCNConv, to_hetero

class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super(GNNEncoder, self).__init__()
        self.conv1 = GATConv((-1, -1), hidden_channels)
        self.conv2 = GATConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

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

class AttnPretrainGNN(torch.nn.Module):
    def __init__(self, hidden_channels, relations, data):
        super(AttnPretrainGNN, self).__init__()
        self.encoder = GNNEncoder(hidden_channels, hidden_channels)
        self.encoder = to_hetero(self.encoder, data.metadata(), aggr='sum')
        self.decoder = EdgeDecoder(hidden_channels, relations)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        # edge_index_dict -> network edges used for encoding nodes
        # edge_label_index -> edges (positive+negative) used for link prediction
        z_dict = self.encoder(x_dict, edge_index_dict)
        return z_dict, self.decoder(z_dict, edge_label_index)

