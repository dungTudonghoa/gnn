"""Explicit JSON schema and a documented substitute generator."""
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import numpy as np


@dataclass
class Instance:
    name: str
    capacity: np.ndarray
    inbound: np.ndarray
    inventory: np.ndarray
    demand: np.ndarray
    supply_cost: np.ndarray
    fixed_cost: np.ndarray
    transfer_cost: np.ndarray
    delivery_cost: np.ndarray
    shortage_cost: np.ndarray
    eligible: np.ndarray

    @property
    def shape(self):
        return len(self.capacity), len(self.inbound), len(self.demand)

    def validate(self):
        o, h, r = self.shape
        if min(o, h, r) < 1:
            raise ValueError('Each node type must be nonempty')
        shapes = {'capacity': (o,), 'inbound': (h,), 'inventory': (h,),
                  'demand': (r,), 'supply_cost': (o,h), 'fixed_cost': (o,h),
                  'transfer_cost': (h,h), 'delivery_cost': (h,r),
                  'shortage_cost': (r,), 'eligible': (o,h)}
        for key, shape in shapes.items():
            a = np.asarray(getattr(self, key))
            if a.shape != shape or not np.isfinite(a).all() or (a < 0).any():
                raise ValueError(f'Invalid {key}: expected finite nonnegative {shape}')
        if not np.isin(self.eligible, [0, 1]).all() or not self.eligible.any(axis=0).all():
            raise ValueError('Every hub needs at least one eligible factory')
        return self

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({k: v.tolist() if isinstance(v, np.ndarray) else v
                                        for k, v in asdict(self).items()}, indent=2))

    @classmethod
    def load(cls, path):
        values = json.loads(Path(path).read_text())
        return cls(**{k: v if k == 'name' else np.asarray(v, dtype=bool if k == 'eligible' else float)
                      for k, v in values.items()}).validate()


def generate(name, shape, seed):
    """NOT the unpublished seed-42 author generator; see REPRODUCIBILITY.md.

    Chosen details: NumPy PCG64, continuous uniforms, zero initial inventory,
    independent uniform(0.5,1.5) capacity shares normalized to the stated totals.
    """
    rng = np.random.default_rng(seed)
    o, h, r = shape
    demand = rng.uniform(80, 120, r)
    factory_share, hub_share = rng.uniform(.5, 1.5, o), rng.uniform(.5, 1.5, h)
    transfer = rng.uniform(10, 30, (h,h))
    np.fill_diagonal(transfer, 0)
    return Instance(name, factory_share / factory_share.sum() * 1.2 * demand.sum(),
                    hub_share / hub_share.sum() * 1.1 * demand.sum(), np.zeros(h), demand,
                    rng.uniform(80, 140, (o,h)), rng.uniform(8000, 20000, (o,h)),
                    transfer, rng.uniform(20, 60, (h,r)), rng.uniform(250, 400, r),
                    np.ones((o,h), dtype=bool)).validate()
