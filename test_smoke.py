"""Offline integration check; synthetic data is NOT a Cora benchmark."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch_geometric.data import Data
import cora


class SmokeTest(unittest.TestCase):
    def test_training_and_artifacts(self):
        torch.manual_seed(7)
        n = 40
        graph = Data(x=torch.rand(n, 8),
                     edge_index=torch.stack([torch.arange(n), torch.arange(n).roll(1)]),
                     y=torch.arange(n) % 3,
                     train_mask=torch.arange(n) < 20,
                     val_mask=(torch.arange(n) >= 20) & (torch.arange(n) < 30),
                     test_mask=torch.arange(n) >= 30)
        class Dataset:
            num_features, num_classes = 8, 3
            def __getitem__(self, index):
                return graph
        with tempfile.TemporaryDirectory() as folder:
            with patch('cora.load_data', return_value=Dataset()):
                model, data, history, metrics = cora.run(epochs=2, output=folder)
            self.assertEqual(model(data.x, data.edge_index).shape, (n, 3))
            self.assertEqual(len(history), 2)
            self.assertTrue(all(torch.isfinite(torch.tensor(row['loss'])) for row in history))
            self.assertTrue(0 <= metrics['test_accuracy'] <= 1)
            for name in ['gcn.pt', 'metrics.json', 'history.csv', 'before.png', 'after.png']:
                self.assertGreater((Path(folder) / name).stat().st_size, 0)
            saved = torch.load(Path(folder) / 'gcn.pt', weights_only=True)
            restored = cora.GCN(saved['input_dim'], saved['hidden_dim'], saved['output_dim'])
            restored.load_state_dict(saved['state_dict'])
            model.eval(); restored.eval()
            torch.testing.assert_close(model(data.x, data.edge_index), restored(data.x, data.edge_index))


if __name__ == '__main__':
    unittest.main()
