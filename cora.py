"""Thực hành node classification bằng GCN trên Cora."""
import argparse
import csv
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import NormalizeFeatures


def load_data(root='data/Planetoid'):
    return Planetoid(root=root, name='Cora', split='public',
                     transform=NormalizeFeatures())


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim=16, output_dim=7):
        super().__init__()
        self.first = GCNConv(input_dim, hidden_dim)
        self.second = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        hidden = F.relu(self.first(x, edge_index))
        hidden = F.dropout(hidden, p=0.5, training=self.training)
        return self.second(hidden, edge_index)


@torch.no_grad()
def accuracy(logits, labels, mask):
    if not mask.any():
        raise ValueError('Mask đánh giá rỗng')
    return (logits.argmax(1)[mask] == labels[mask]).float().mean().item()


def inspect_data(dataset):
    graph = dataset[0]
    print(graph)
    print('Số đồ thị:', len(dataset))
    for name in ('train_mask', 'val_mask', 'test_mask'):
        print(name, int(graph[name].sum()))
    print('Đặc trưng khác 0 của node 0:', graph.x[0].nonzero().flatten().tolist())
    print('Giá trị tương ứng:', graph.x[0][graph.x[0] != 0].tolist())
    print('10 cạnh đầu (source, target):', graph.edge_index[:, :10].t().tolist())
    print('10 nhãn đầu:', graph.y[:10].tolist())
    print('Các file nguồn Planetoid:', dataset.raw_paths)


def plot_embedding(logits, labels, destination, seed):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    points = TSNE(n_components=2, random_state=seed, init='pca',
                  learning_rate='auto').fit_transform(logits.detach().cpu().numpy())
    fig, ax = plt.subplots(figsize=(8, 6))
    dots = ax.scatter(points[:, 0], points[:, 1], c=labels.cpu(), s=8, cmap='tab10')
    fig.colorbar(dots, ax=ax, label='Class ID')
    ax.set_title(destination.stem + ' (t-SNE of logits)')
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def run(epochs=100, root='data/Planetoid', output='outputs', seed=1234567,
        device='cpu', plots=True):
    if epochs < 1:
        raise ValueError('epochs phải >= 1')
    torch.manual_seed(seed)
    dataset = load_data(root)
    graph = dataset[0].to(device)
    model = GCN(dataset.num_features, 16, dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    if plots:
        model.eval()
        with torch.no_grad():
            plot_embedding(model(graph.x, graph.edge_index), graph.y,
                           destination / 'before.png', seed)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(graph.x, graph.edge_index)
        loss = F.cross_entropy(logits[graph.train_mask], graph.y[graph.train_mask])
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            logits = model(graph.x, graph.edge_index)
            val = accuracy(logits, graph.y, graph.val_mask)
        history.append({'epoch': epoch, 'loss': loss.item(), 'val_accuracy': val})
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f'Epoch {epoch:03d} | loss={loss.item():.4f} | val={val:.4f}')
    # Đánh giá checkpoint cuối, chỉ xem test sau khi hoàn tất training.
    result = {'epochs': epochs, 'seed': seed, 'device': str(device),
              'torch_version': torch.__version__,
              'val_accuracy': val,
              'test_accuracy': accuracy(logits, graph.y, graph.test_mask)}
    with (destination / 'history.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    (destination / 'metrics.json').write_text(json.dumps(result, indent=2))
    torch.save({'state_dict': model.state_dict(), 'input_dim': dataset.num_features,
                'hidden_dim': 16, 'output_dim': dataset.num_classes}, destination / 'gcn.pt')
    if plots:
        plot_embedding(logits, graph.y, destination / 'after.png', seed)
    print(json.dumps(result, indent=2))
    return model, graph, history, result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inspect', action='store_true')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--root', default='data/Planetoid')
    parser.add_argument('--output', default='outputs')
    parser.add_argument('--seed', type=int, default=1234567)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    if args.inspect:
        inspect_data(load_data(args.root))
    else:
        run(args.epochs, args.root, args.output, args.seed, args.device, not args.no_plots)
