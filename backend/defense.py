import os
import glob
import torch
import pandas as pd
import numpy as np
import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GINEConv, BatchNorm, Linear, to_hetero

BATCH_SIZE = 8192
NUM_NEIGHBORS = 50
N_HIDDEN = 64
N_GNN_LAYERS = 2

class GINe(torch.nn.Module):
    def __init__(self, num_features, num_gnn_layers, n_classes=2, n_hidden=100, edge_updates=False, edge_dim=None, dropout=0.0, final_dropout=0.5):
        super().__init__()
        self.n_hidden = n_hidden
        self.num_gnn_layers = num_gnn_layers
        self.edge_updates = edge_updates
        self.final_dropout = final_dropout
        self.node_emb = nn.Linear(num_features, n_hidden)
        self.edge_emb = nn.Linear(edge_dim, n_hidden)
        self.convs = nn.ModuleList()
        self.emlps = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(self.num_gnn_layers):
            conv = GINEConv(nn.Sequential(nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU(), nn.Linear(self.n_hidden, self.n_hidden)), edge_dim=self.n_hidden)
            if self.edge_updates:
                self.emlps.append(nn.Sequential(nn.Linear(3 * self.n_hidden, self.n_hidden), nn.ReLU(), nn.Linear(self.n_hidden, self.n_hidden)))
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(n_hidden))
        self.mlp = nn.Sequential(Linear(n_hidden * 3, 50), nn.ReLU(), nn.Dropout(self.final_dropout), Linear(50, 25), nn.ReLU(), nn.Dropout(self.final_dropout), Linear(25, n_classes))

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index
        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)
        for i in range(self.num_gnn_layers):
            x = (x + F.relu(self.batch_norms[i](self.convs[i](x, edge_index, edge_attr)))) / 2
            if self.edge_updates:
                edge_attr = edge_attr + self.emlps[i](torch.cat([x[src], x[dst], edge_attr], dim=-1)) / 2
        x = x[edge_index.T].reshape(-1, 2 * self.n_hidden).relu()
        x = torch.cat((x, edge_attr.view(-1, edge_attr.shape[1])), dim=1)
        return self.mlp(x)

class HeteroEdgeLoader:
    def __init__(self, data, edge_inds, batch_size, shuffle=False, num_neighbors=50, add_ego_ids=True):
        self.data = data
        self.edge_inds = edge_inds
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_neighbors = num_neighbors
        self.add_ego_ids = add_ego_ids
        
        n_nodes = data['node'].x.shape[0]
        src_to = data['node', 'to', 'node'].edge_index[0]
        self._perm_to = src_to.argsort()
        counts_to = torch.bincount(src_to, minlength=n_nodes)
        self._ptr_to = torch.cat([torch.zeros(1, dtype=torch.long), counts_to.cumsum(0)])
        
        src_rev = data['node', 'rev_to', 'node'].edge_index[0]
        self._perm_rev = src_rev.argsort()
        counts_rev = torch.bincount(src_rev, minlength=n_nodes)
        self._ptr_rev = torch.cat([torch.zeros(1, dtype=torch.long), counts_rev.cumsum(0)])

    def __len__(self):
        return (len(self.edge_inds) + self.batch_size - 1) // self.batch_size

    def _neighbor_edges(self, nodes, ptr, perm):
        starts, ends = ptr[nodes], ptr[nodes + 1]
        parts = []
        for s, e in zip(starts.tolist(), ends.tolist()):
            if s == e: continue
            eidx = perm[s:e]
            if len(eidx) > self.num_neighbors:
                eidx = eidx[torch.randperm(len(eidx))[:self.num_neighbors]]
            parts.append(eidx)
        return torch.cat(parts) if parts else torch.empty(0, dtype=torch.long)

    def _make_batch(self, chunk):
        seed_edges = chunk
        seed_nodes = self.data['node', 'to', 'node'].edge_index[:, seed_edges].reshape(-1).unique()
        
        nbr_to = self._neighbor_edges(seed_nodes, self._ptr_to, self._perm_to)
        nbr_rev = self._neighbor_edges(seed_nodes, self._ptr_rev, self._perm_rev)
        
        all_to = torch.cat([seed_edges, nbr_to]).unique()
        all_rev = torch.cat([seed_edges, nbr_rev]).unique()
        
        nodes_to = self.data['node', 'to', 'node'].edge_index[:, all_to].reshape(-1)
        nodes_rev = self.data['node', 'rev_to', 'node'].edge_index[:, all_rev].reshape(-1)
        all_nodes = torch.cat([nodes_to, nodes_rev]).unique()
        
        node_map = torch.full((self.data['node'].x.shape[0],), -1, dtype=torch.long)
        node_map[all_nodes] = torch.arange(len(all_nodes))
        
        sub_ei_to = node_map[self.data['node', 'to', 'node'].edge_index[:, all_to]]
        sub_ei_rev = node_map[self.data['node', 'rev_to', 'node'].edge_index[:, all_rev]]
        
        batch = HeteroData()
        x_batch = self.data['node'].x[all_nodes]
        if self.add_ego_ids:
            ego_feat = torch.zeros((len(all_nodes), 1), dtype=torch.float)
            ego_nodes = self.data['node', 'to', 'node'].edge_index[:, seed_edges].reshape(-1).unique()
            ego_feat[node_map[ego_nodes]] = 1.0
            x_batch = torch.cat([x_batch, ego_feat], dim=1)
            
        batch['node'].x = x_batch
        batch['node', 'to', 'node'].edge_index = sub_ei_to
        batch['node', 'rev_to', 'node'].edge_index = sub_ei_rev
        batch['node', 'to', 'node'].edge_attr = self.data['node', 'to', 'node'].edge_attr[all_to]
        batch['node', 'rev_to', 'node'].edge_attr = self.data['node', 'rev_to', 'node'].edge_attr[all_rev]
        batch.input_id = chunk
        batch._seed_ids = self.data['node', 'to', 'node'].edge_attr[seed_edges, 0]
        return batch

    def __iter__(self):
        n = len(self.edge_inds)
        order = torch.randperm(n) if self.shuffle else torch.arange(n)
        for start in range(0, n, self.batch_size):
            chunk = self.edge_inds[order[start:start + self.batch_size]]
            yield self._make_batch(chunk)

def encode_shared(df, cols):
    shared = {}
    for col in cols:
        for val in df[col]:
            if val not in shared:
                shared[val] = len(shared)
        df[col] = df[col].map(shared)
    return df

def build_hetero_data(df):
    print("Building Graph Features")
    
    nodes = list(set(df['Account'].tolist() + df['Account.1'].tolist()))
    node_map = {acc: i for i, acc in enumerate(nodes)}
    
    src = torch.tensor(df['Account'].map(node_map).values, dtype=torch.long)
    dst = torch.tensor(df['Account.1'].map(node_map).values, dtype=torch.long)
    edge_index = torch.stack([src, dst], dim=0)
    
    df = encode_shared(df, ['Receiving Currency', 'Payment Currency'])
    df['Payment Format'] = pd.factorize(df['Payment Format'])[0]
    
    base_attr = torch.tensor(
        df[['Amount Received', 'Receiving Currency', 'Amount Paid', 'Payment Format']].to_numpy(),
        dtype=torch.float
    )
    
    mean = base_attr.mean(dim=0, keepdim=True)
    std  = base_attr.std(dim=0, keepdim=True)
    std[std == 0] = 1.0
    edge_attr = (base_attr - mean) / std
    
    ids = torch.arange(len(df)).view(-1, 1).float()
    edge_attr_w_id = torch.cat([ids, edge_attr], dim=1)
    
    x = torch.ones((len(nodes), 1), dtype=torch.float)
    
    h = HeteroData()
    h['node'].x = x
    h['node', 'to', 'node'].edge_index = edge_index
    h['node', 'rev_to', 'node'].edge_index = edge_index.flipud()
    h['node', 'to', 'node'].edge_attr = edge_attr_w_id
    
    rev_attr = edge_attr_w_id.clone()
    if rev_attr.shape[1] >= 9:
        rev_attr[:, [6, 7]] = rev_attr[:, [7, 6]]
        rev_attr[:, [8, 9]] = rev_attr[:, [9, 8]]
    h['node', 'rev_to', 'node'].edge_attr = rev_attr
    
    return h

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Blue Team Engine (Defense) Initialized on {device}")
    
    attack_files = sorted(glob.glob('backend/attacks/attack_*.csv'))
    if not attack_files:
        attack_files = sorted(glob.glob('attacks/attack_*.csv'))
        if not attack_files:
            raise FileNotFoundError("No attack CSV files found in backend/attacks/")
    
    latest_attack_file = attack_files[-1]
    print(f"Loading incoming adversarial attack: {latest_attack_file}")
    
    df = pd.read_csv(latest_attack_file)
    total_transactions = len(df)
    
    if 'Is Laundering' not in df.columns:
        raise ValueError("'Is Laundering' column is missing from the attack data.")
    
    y_true = df['Is Laundering'].values
    total_frauds_before = int(np.sum(y_true))
    total_real_before = total_transactions - total_frauds_before
    
    df = df.drop(columns=['Is Laundering'])
    hetero_data = build_hetero_data(df)
    all_inds = torch.arange(total_transactions)
    loader = HeteroEdgeLoader(hetero_data, all_inds, BATCH_SIZE, shuffle=False, num_neighbors=NUM_NEIGHBORS, add_ego_ids=True)
    
    sample_batch = next(iter(loader))
    EDGE_DIM = sample_batch['node', 'to', 'node'].edge_attr.shape[1] - 1
    NUM_FEATURES = sample_batch['node'].x.shape[1]
    
    base_model = GINe(num_features=NUM_FEATURES, num_gnn_layers=N_GNN_LAYERS, n_classes=2, n_hidden=N_HIDDEN, edge_updates=True, edge_dim=EDGE_DIM, dropout=0.0, final_dropout=0.105)
    model = to_hetero(base_model, hetero_data.metadata(), aggr='mean').to(device)
    
    model_paths = ['model/best_model.pt', '../model/best_model.pt', 'best_model.pt', '/kaggle/working/best_model.pt']
    model_load_path = next((p for p in model_paths if os.path.exists(p)), None)
    
    if model_load_path is None:
        raise FileNotFoundError("best_model.pt not found! Ensure the trained model exists.")
        
    print(f"Loading defense model weights from: {model_load_path}")
    model.load_state_dict(torch.load(model_load_path, map_location=device, weights_only=True))
    model.eval()
    
    preds = []
    print("Defending against attack (Running Inference)")
    with torch.no_grad():
        for batch in tqdm.tqdm(loader, desc="Detecting"):
            seed_ids = batch._seed_ids
            batch_edge_ids = batch['node', 'to', 'node'].edge_attr[:, 0].cpu()
            
            batch['node', 'to', 'node'].edge_attr = batch['node', 'to', 'node'].edge_attr[:, 1:]
            batch['node', 'rev_to', 'node'].edge_attr = batch['node', 'rev_to', 'node'].edge_attr[:, 1:]
            
            batch = batch.to(device)
            out_dict = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
            
            subgraph_edge_ids = batch_edge_ids.to(device)
            subgraph_out = out_dict[('node', 'to', 'node')]
            
            for s_id in seed_ids:
                idx = (subgraph_edge_ids == s_id).nonzero(as_tuple=True)[0]
                if len(idx) > 0:
                    prob = torch.softmax(subgraph_out[idx[0]], dim=-1)[1].item()
                    preds.append((int(s_id.item()), 1 if prob >= 0.5 else 0))
                    
    preds.sort(key=lambda x: x[0])
    y_pred = np.array([p[1] for p in preds])
    
    true_positives = np.sum((y_pred == 1) & (y_true == 1))
    false_positives = np.sum((y_pred == 1) & (y_true == 0))
    true_negatives = np.sum((y_pred == 0) & (y_true == 0))
    false_negatives = np.sum((y_pred == 0) & (y_true == 1))
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    print("\n" + "="*50)
    print("BLUE TEAM DEFENSE REPORT")
    print("="*50)
    print(f"Total number of transactions           : {total_transactions:,}")
    print(f"Total true frauds before preprocessing : {total_frauds_before:,}")
    print(f"Total real trans before preprocessing  : {total_real_before:,}")
    print("-" * 50)
    print(f"Frauds detected (True Positives)       : {true_positives:,}")
    print(f"Real trans passed (True Negatives)     : {true_negatives:,}")
    print(f"Real trans caught as fraud (False Pos.): {false_positives:,}")
    print("-" * 50)
    print(f"Precision                              : {precision:.4f}")
    print(f"Recall (Fraud Catch Rate)              : {recall:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
