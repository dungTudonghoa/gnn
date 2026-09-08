"""Heterogeneous assignment network; feature layout is an explicit reconstruction."""
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

RELATIONS = {'fh': ('factory','hub',3), 'hf': ('hub','factory',3),
             'hr': ('hub','retailer',1), 'rh': ('retailer','hub',1),
             'hh': ('hub','hub',1)}
NODE_DIMS = {'factory':5, 'hub':8, 'retailer':4}


def features(d):
    o,h,r = d.shape
    nodes = {
        'factory': np.column_stack([d.capacity, d.supply_cost.mean(1), d.supply_cost.min(1),
                                    d.fixed_cost.mean(1), d.fixed_cost.min(1)]),
        'hub': np.column_stack([d.inbound, d.inventory, d.supply_cost.mean(0),
                               d.supply_cost.min(0), d.fixed_cost.mean(0), d.fixed_cost.min(0),
                               d.delivery_cost.mean(1), d.transfer_cost.sum(1)/max(h-1,1)]),
        'retailer': np.column_stack([d.demand, d.shortage_cost,
                                    d.delivery_cost.mean(0), d.delivery_cost.min(0)])}
    # Include eligibility as an edge feature; disallowed pairs are masked at output.
    a,b = np.indices((o,h)); fh = np.stack([a.ravel(), b.ravel()])
    a,b = np.indices((h,r)); hr = np.stack([a.ravel(), b.ravel()])
    a,b = np.where(~np.eye(h, dtype=bool)); hh = np.stack([a,b])
    fh_attr = np.column_stack([d.supply_cost.ravel(), d.fixed_cost.ravel(), d.eligible.ravel()])
    edges = {'fh': (fh,fh_attr), 'hf': (fh[::-1].copy(),fh_attr.copy()),
             'hr': (hr,d.delivery_cost.reshape(-1,1)),
             'rh': (hr[::-1].copy(),d.delivery_cost.reshape(-1,1)),
             'hh': (hh,d.transfer_cost[a,b].reshape(-1,1))}
    return nodes, edges


class Standardizer:
    def fit(self, instances):
        records = [features(d) for d in instances]
        self.stats = {}
        for key in list(NODE_DIMS) + list(RELATIONS):
            arrays = [n[key] if key in n else e[key][1] for n,e in records]
            values = np.concatenate(arrays)
            if len(values):
                mean, std = values.mean(0), values.std(0)
            else:
                mean, std = np.zeros(values.shape[1]), np.ones(values.shape[1])
            std[std < 1e-8] = 1
            if key in ('fh','hf'): mean[-1], std[-1] = 0, 1
            self.stats[key] = (mean, std)
        return self

    def transform(self, d):
        nodes, edges = features(d)
        def norm(key, array):
            mean,std = self.stats[key]
            return torch.tensor((array-mean)/std, dtype=torch.float32)
        return ({k:norm(k,v) for k,v in nodes.items()},
                {k:(torch.tensor(i.copy(), dtype=torch.long),norm(k,v)) for k,(i,v) in edges.items()},
                torch.tensor(d.eligible, dtype=torch.bool))

    def state_dict(self):
        return {k:[m.tolist(),s.tolist()] for k,(m,s) in self.stats.items()}

    @classmethod
    def from_state(cls, values):
        result = cls(); result.stats = {k:tuple(np.asarray(v) for v in pair) for k,pair in values.items()}
        return result


def mlp(input_dim, output_dim, hidden=32):
    return nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, output_dim))


class AssignmentGNN(nn.Module):
    def __init__(self, hidden=32, rounds=2):
        super().__init__()
        self.encoders = nn.ModuleDict({k:nn.Linear(dim, hidden) for k,dim in NODE_DIMS.items()})
        self.messages = nn.ModuleList([nn.ModuleDict({k:mlp(2*hidden+dim,hidden,hidden)
                                      for k,(_,_,dim) in RELATIONS.items()}) for _ in range(rounds)])
        self.updates = nn.ModuleList([nn.ModuleDict({k:nn.Linear(hidden,hidden) for k in NODE_DIMS})
                                     for _ in range(rounds)])
        self.norms = nn.ModuleList([nn.ModuleDict({k:nn.LayerNorm(hidden) for k in NODE_DIMS})
                                   for _ in range(rounds)])
        self.scorer = mlp(2*hidden+3,1,hidden)

    def forward(self, graph):
        nodes,edges,eligible = graph
        state = {k:F.relu(self.encoders[k](v)) for k,v in nodes.items()}
        for messages,updates,norms in zip(self.messages,self.updates,self.norms):
            received = {k:torch.zeros_like(v) for k,v in state.items()}
            for key,(src,dst,_) in RELATIONS.items():
                indices,attr = edges[key]; a,b = indices
                msg = messages[key](torch.cat([state[src][a],state[dst][b],attr],dim=1))
                aggregate = torch.zeros_like(state[dst]).index_add(0,b,msg)
                degree = torch.bincount(b,minlength=len(state[dst])).clamp_min(1).unsqueeze(1)
                received[dst] = received[dst] + aggregate/degree
            state = {k:F.relu(norms[k](v + updates[k](received[k]))) for k,v in state.items()}
        indices,attr = edges['fh']; a,b = indices
        logits = self.scorer(torch.cat([state['factory'][a],state['hub'][b],attr],1))
        return logits.reshape(eligible.shape).masked_fill(~eligible, -torch.inf)


@torch.no_grad()
def probabilities(model, normalizer, instance):
    model.eval()
    return model(normalizer.transform(instance)).softmax(dim=0).cpu().numpy()


def train(training, validation, labels, epochs=100, seed=3101):
    """Whole-training-set optimizer step; validation CE chooses checkpoint."""
    if not training or not validation or epochs < 1:
        raise ValueError('Nonempty train/validation and positive epochs required')
    if {d.name for d in training} & {d.name for d in validation}:
        raise ValueError('Train/validation leakage')
    torch.manual_seed(seed)
    normalizer = Standardizer().fit(training)
    model = AssignmentGNN()
    optimizer = torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=1e-4)
    def prepare(instances):
        return [(normalizer.transform(d), torch.tensor(labels[d.name],dtype=torch.long)) for d in instances]
    train_data, val_data = prepare(training), prepare(validation)
    history, best_loss, wait, best = [], float('inf'), 0, None
    for epoch in range(1, epochs+1):
        model.train(); optimizer.zero_grad()
        loss = sum(F.cross_entropy(model(g).t(),y,reduction='sum') for g,y in train_data)
        loss = loss / sum(len(y) for _,y in train_data)
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),5.); optimizer.step()
        model.eval()
        with torch.no_grad():
            logits = [(model(g).t(),y) for g,y in val_data]
            val_loss = sum(F.cross_entropy(z,y,reduction='sum').item() for z,y in logits)/sum(len(y) for _,y in logits)
            val_acc = sum((z.argmax(1)==y).sum().item() for z,y in logits)/sum(len(y) for _,y in logits)
        history.append(dict(epoch=epoch,train_loss=loss.item(),val_loss=val_loss,val_accuracy=val_acc))
        if val_loss < best_loss-1e-8:
            best_loss,wait,best = val_loss,0,copy.deepcopy(model.state_dict())
            best_epoch = epoch
        else:
            wait += 1
        if wait >= 25: break
    model.load_state_dict(best)
    return model,normalizer,history,best_epoch
