import os
import glob
import torch
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
    def __init__(self, data, edge_inds, batch_size, shuffle=False, num_neighbors=50, add_ego_ids=True, balanced=False):
        self.data = data
        self.edge_inds = edge_inds
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_neighbors = num_neighbors
        self.add_ego_ids = add_ego_ids
        self.balanced = balanced
        n_nodes = data['node'].x.shape[0]
        src_to = data['node', 'to', 'node'].edge_index[0]
        self._perm_to = src_to.argsort()
        counts_to = torch.bincount(src_to, minlength=n_nodes)
        self._ptr_to = torch.cat([torch.zeros(1, dtype=torch.long), counts_to.cumsum(0)])
        src_rev = data['node', 'rev_to', 'node'].edge_index[0]
        self._perm_rev = src_rev.argsort()
        counts_rev = torch.bincount(src_rev, minlength=n_nodes)
        self._ptr_rev = torch.cat([torch.zeros(1, dtype=torch.long), counts_rev.cumsum(0)])
        if balanced:
            y_edges = data['node', 'to', 'node'].y[edge_inds]
            pos_mask = y_edges == 1
            self._pos_inds = edge_inds[pos_mask]
            self._neg_inds = edge_inds[~pos_mask]

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
        batch['node', 'to', 'node'].y = self.data['node', 'to', 'node'].y[all_to]
        batch.input_id = chunk
        batch._seed_ids = self.data['node', 'to', 'node'].edge_attr[seed_edges, 0]
        return batch

    def __iter__(self):
        if self.balanced:
            half = self.batch_size // 2
            n_batches = len(self)
            for _ in range(n_batches):
                pos_i = torch.randint(0, len(self._pos_inds), (half,))
                neg_i = torch.randint(0, len(self._neg_inds), (half,))
                chunk = torch.cat([self._pos_inds[pos_i], self._neg_inds[neg_i]])
                yield self._make_batch(chunk)
        else:
            n = len(self.edge_inds)
            order = torch.randperm(n) if self.shuffle else torch.arange(n)
            for start in range(0, n, self.batch_size):
                chunk = self.edge_inds[order[start:start + self.batch_size]]
                yield self._make_batch(chunk)

def to_hetero_data(data):
    h = HeteroData()
    h['node'].x = data.x
    h['node', 'to', 'node'].edge_index = data.edge_index
    h['node', 'rev_to', 'node'].edge_index = data.edge_index.flipud()
    h['node', 'to', 'node'].edge_attr = data.edge_attr
    h['node', 'rev_to', 'node'].edge_attr = data.edge_attr.clone()
    if h['node', 'rev_to', 'node'].edge_attr.shape[1] >= 8:
        h['node', 'rev_to', 'node'].edge_attr[:, [-4, -3]] = h['node', 'rev_to', 'node'].edge_attr[:, [-3, -4]]
    h['node', 'to', 'node'].y = data.y
    return h

def add_arange_ids(data_list):
    for data in data_list:
        n_edges = data['node', 'to', 'node'].edge_attr.shape[0]
        ids = torch.arange(n_edges).view(-1, 1)
        data['node', 'to', 'node'].edge_attr = torch.cat([ids, data['node', 'to', 'node'].edge_attr], dim=1)
        data['node', 'rev_to', 'node'].edge_attr = torch.cat([ids.clone(), data['node', 'rev_to', 'node'].edge_attr], dim=1)

@torch.no_grad()
def evaluate_hetero(loader, model, device, threshold=0.5):
    model.eval()
    preds, ground_truths = [], []
    for batch in tqdm.tqdm(loader, desc='Evaluating', leave=False):
        seed_ids = batch._seed_ids
        mask = torch.isin(batch['node', 'to', 'node'].edge_attr[:, 0].detach().cpu(), seed_ids.cpu())
        batch['node', 'to', 'node'].edge_attr = batch['node', 'to', 'node'].edge_attr[:, 1:]
        batch['node', 'rev_to', 'node'].edge_attr = batch['node', 'rev_to', 'node'].edge_attr[:, 1:]
        batch = batch.to(device)
        mask = mask.to(device)
        out_dict = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
        out = out_dict[('node', 'to', 'node')][mask]
        probs = torch.softmax(out, dim=-1)[:, 1]
        pred = (probs >= threshold).long()
        preds.append(pred.cpu())
        ground_truths.append(batch['node', 'to', 'node'].y[mask].cpu())
    pred = torch.cat(preds).numpy()
    ground_truth = torch.cat(ground_truths).numpy()
    return {
        'f1': f1_score(ground_truth, pred, zero_division=0),
        'precision': precision_score(ground_truth, pred, zero_division=0),
        'recall': recall_score(ground_truth, pred, zero_division=0)
    }

def run_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOCAL_PATH = os.path.join(os.path.dirname(__file__), 'split_data.pt')
    paths_to_check = [LOCAL_PATH, 'model/split_data.pt', 'split_data.pt']
    kaggle_glob = glob.glob('/kaggle/input/**/split_data*', recursive=True)
    if kaggle_glob: paths_to_check.insert(0, kaggle_glob[0])
    split_path = next((p for p in paths_to_check if os.path.exists(p)), None)
    if split_path is None: raise FileNotFoundError("split_data.pt not found!")
    split = torch.load(split_path, weights_only=False)
    te_data = to_hetero_data(split['te_data'])
    te_inds = split['te_inds']
    add_arange_ids([te_data])
    te_loader = HeteroEdgeLoader(te_data, te_inds, BATCH_SIZE, shuffle=False, num_neighbors=NUM_NEIGHBORS, add_ego_ids=True, balanced=False)
    sample_batch = next(iter(te_loader))
    EDGE_DIM = sample_batch['node', 'to', 'node'].edge_attr.shape[1] - 1
    NUM_FEATURES = sample_batch['node'].x.shape[1]
    base_model = GINe(num_features=NUM_FEATURES, num_gnn_layers=N_GNN_LAYERS, n_classes=2, n_hidden=N_HIDDEN, edge_updates=True, edge_dim=EDGE_DIM, dropout=0.0, final_dropout=0.105)
    model = to_hetero(base_model, te_data.metadata(), aggr='mean').to(device)
    LOCAL_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best_model.pt')
    paths_to_check_model = [LOCAL_MODEL_PATH, 'model/best_model.pt', 'best_model.pt', '/kaggle/working/best_model.pt']
    model_load_path = next((p for p in paths_to_check_model if os.path.exists(p)), None)
    if model_load_path is None: raise FileNotFoundError("best_model.pt not found!")
    model.load_state_dict(torch.load(model_load_path, map_location=device, weights_only=True))
    final_metrics = evaluate_hetero(te_loader, model, device)
    print(f"Final Test Set Results: F1 = {final_metrics['f1']:.4f} | Precision = {final_metrics['precision']:.4f} | Recall = {final_metrics['recall']:.4f}")

if __name__ == "__main__":
    run_inference()
