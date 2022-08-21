import numpy as np
import os
from torch_geometric.data import HeteroData
import torch
import torch_geometric.transforms as T

def get_kg(network_name='gold'):
    node_list, edge_list, feats = None, None, None
    node_list, edge_list, feats = load_raw_data(network=network_name)

    ipf_kg = HeteroData()
    #ipf_kg['gene'].x = torch.FloatTensor(feats)
    ipf_kg['gene'].x = torch.eye(node_list[0].shape[0])
    ipf_kg['msig'].x = torch.eye(node_list[1].shape[0])
    ipf_kg['reactome'].x = torch.eye(node_list[2].shape[0])
    ipf_kg['bp'].x = torch.eye(node_list[3].shape[0])
    #ipf_kg['bp'].x = torch.ones(node_list[3].shape[0])

    ipf_kg['gene', 'g2g', 'gene'].edge_index = torch.LongTensor(edge_list[0].T)
    ipf_kg['gene', 'genemsig', 'msig'].edge_index = torch.LongTensor(edge_list[1].T)
    ipf_kg['gene', 'genereact', 'reactome'].edge_index = torch.LongTensor(edge_list[2].T)
    ipf_kg['gene', 'genebp', 'bp'].edge_index = torch.LongTensor(edge_list[3].T)

    ipf_kg = T.ToUndirected()(ipf_kg)
    ipf_kg = T.AddSelfLoops()(ipf_kg)
    ipf_kg = T.NormalizeFeatures()(ipf_kg)
    return ipf_kg

def load_raw_data(network='gold'):
    indir = f'data/raw/kg_data/{network}'
    gene_nodes = np.load(os.path.join(indir, 'gene_nodes.npy'))
    msig_terms = np.load(os.path.join(indir, 'msigdb_nodes.npy'))
    reactome_terms = np.load(os.path.join(indir, 'reactome_nodes.npy'))
    bp_terms = np.load(os.path.join(indir, 'bp_nodes.npy'))
    print(f'Genes: {gene_nodes.shape}, MSig Terms: {msig_terms.shape}'
          f' Reactome terms: {reactome_terms.shape}, BP Terms: {bp_terms.shape}')

    # loading all edges and features - gold standard v2
    gene_edges = np.load(os.path.join(indir, 'gene_links.npy'))
    gene_msigdb_links = np.load(os.path.join(indir, 'gene_msigdb_links.npy'))
    gene_react_links = np.load(os.path.join(indir, 'gene_reactome_links.npy'))
    gene_bp_links = np.load(os.path.join(indir, 'gene_bp_links.npy'))
    # loading features
    cell_feats = np.load(os.path.join(indir, 'gene_feats.npy'))
    print(f'gene-gene edges: {gene_edges.shape},'
          f' gene-msigdb edges: {gene_msigdb_links.shape}'
          f' gene-reactome edges: {gene_react_links.shape}'
          f' gene-bp edges: {gene_bp_links.shape}')

    nodes = [gene_nodes, msig_terms, reactome_terms, bp_terms]
    edges = [gene_edges, gene_msigdb_links, gene_react_links, gene_bp_links]
    assert len(nodes) == len(edges)
    return nodes, edges, cell_feats