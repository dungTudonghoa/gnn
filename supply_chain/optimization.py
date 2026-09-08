"""Paper equations (1)-(10): complete MILP and cached fixed-assignment LP."""
import itertools
import warnings
from dataclasses import dataclass
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp, OptimizeWarning
from scipy.sparse import coo_matrix, csr_matrix, hstack, vstack

# Column order: fixed, supply, transfer, delivery, shortage.
SCENARIOS = np.array([[.8,.8,.5,.9,.7], [1,1,1,1,1], [1.2,1.2,1.5,1.1,1.3]])


@dataclass
class Solution:
    objective: float
    assignment: tuple
    flows: np.ndarray
    scenario_costs: np.ndarray


class FlowModel:
    def __init__(self, instance):
        self.d = instance.validate()
        o, h, r = self.d.shape
        self.o, self.h, self.r = o, h, r
        self.arcs = [(a,b) for a in range(h) for b in range(h) if a != b]
        cursor = 0
        self.qf = np.arange(cursor, cursor + o*h).reshape(o,h); cursor += o*h
        self.qh = np.arange(cursor, cursor + len(self.arcs)); cursor += len(self.arcs)
        self.qr = np.arange(cursor, cursor + h*r).reshape(h,r); cursor += h*r
        self.u = np.arange(cursor, cursor + r); cursor += r
        self.n = cursor
        rows, cols, vals = [], [], []
        def put(row, col, value):
            rows.append(row); cols.append(int(col)); vals.append(value)
        for a in range(o):
            for b in range(h):
                put(a, self.qf[a,b], 1)
        for b in range(h):
            for a in range(o): put(o+b, self.qf[a,b], -1)
            for j in range(r): put(o+b, self.qr[b,j], 1)
        for k, (a,b) in enumerate(self.arcs):
            put(o+a, self.qh[k], 1)
            put(o+b, self.qh[k], -1)
        self.a_ub = coo_matrix((vals, (rows, cols)), shape=(o+h, self.n)).tocsr()
        self.b_ub = np.r_[instance.capacity, instance.inventory]
        rows, cols, vals = [], [], []
        for j in range(r):
            for b in range(h): put(j, self.qr[b,j], 1)
            put(j, self.u[j], 1)
        self.a_eq = coo_matrix((vals, (rows, cols)), shape=(r, self.n)).tocsr()
        self.b_eq = instance.demand.copy()
        base = np.zeros((4, self.n))
        base[0, self.qf.ravel()] = instance.supply_cost.ravel()
        base[1, self.qh] = [instance.transfer_cost[a,b] for a,b in self.arcs]
        base[2, self.qr.ravel()] = instance.delivery_cost.ravel()
        base[3, self.u] = instance.shortage_cost
        self.costs = SCENARIOS[:, 1:] @ base
        self.fixed = SCENARIOS[:, 0, None] * instance.fixed_cost.ravel()[None, :]
        # Build regret matrix once; only assignment-specific RHS changes.
        self.regret_ub = vstack([hstack([self.a_ub, csr_matrix((o+h,1))]),
                                csr_matrix(np.c_[self.costs, -np.ones(3)])]).tocsr()
        self.regret_eq = hstack([self.a_eq, csr_matrix((r,1))]).tocsr()

    def choices(self):
        return [np.flatnonzero(self.d.eligible[:, b]).tolist() for b in range(self.h)]

    def onehot(self, assignment):
        a = np.asarray(assignment)
        if a.shape != (self.h,) or not np.issubdtype(a.dtype, np.integer):
            raise ValueError('Assignment must contain one integer factory index per hub')
        if (a < 0).any() or (a >= self.o).any() or not self.d.eligible[a, np.arange(self.h)].all():
            raise ValueError('Ineligible assignment')
        x = np.zeros((self.o, self.h)); x[a, np.arange(self.h)] = 1
        return x

    def solve_fixed(self, assignment, references=None, scenario=1):
        x = self.onehot(assignment)
        fixed = self.fixed @ x.ravel()
        upper = np.full(self.n, np.inf)
        upper[self.qf.ravel()] = (x * self.d.inbound).ravel()
        if references is None:
            c, au, bu, ae = self.costs[scenario], self.a_ub, self.b_ub, self.a_eq
        else:
            refs = np.asarray(references)
            if refs.shape != (3,) or not np.isfinite(refs).all():
                raise ValueError('Three finite scenario references required')
            c = np.r_[np.zeros(self.n), 1.]
            au, bu, ae = self.regret_ub, np.r_[self.b_ub, refs-fixed], self.regret_eq
            upper = np.r_[upper, np.inf]
        lower = np.zeros(len(c))
        if references is not None: lower[-1] = -np.inf
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=OptimizeWarning, message='Unrecognized options.*')
            result = linprog(c, A_ub=au, b_ub=bu, A_eq=ae, b_eq=self.b_eq,
                             bounds=np.c_[lower, upper], method='highs',
                             options={'threads': 1, 'primal_feasibility_tolerance': 1e-8,
                                      'dual_feasibility_tolerance': 1e-8})
        if not result.success:
            raise RuntimeError(f'LP failed: {result.message}')
        flows = result.x[:self.n]
        costs = self.costs @ flows + fixed
        objective = costs[scenario] if references is None else max(costs-np.asarray(references))
        return Solution(float(objective), tuple(int(v) for v in assignment), flows, costs)

    def solve_complete(self, references=None, scenario=1, fixed_assignment=None, time_limit=120):
        """Require proven optimality; a time-limited incumbent is never an exact label."""
        nx = self.o*self.h
        nr = int(references is not None)
        n = self.n + nx + nr
        au = hstack([self.a_ub, csr_matrix((len(self.b_ub), nx+nr))]).tocsr()
        ae = hstack([self.a_eq, csr_matrix((self.r, nx+nr))]).tocsr()
        # Supply may enter a hub only through its chosen factory, eq. (5).
        rows = np.repeat(np.arange(nx), 2)
        cols = np.column_stack([self.qf.ravel(), self.n+np.arange(nx)]).ravel()
        vals = np.column_stack([np.ones(nx), -np.tile(self.d.inbound, self.o)]).ravel()
        link = coo_matrix((vals,(rows,cols)), shape=(nx,n)).tocsr()
        assign = coo_matrix((np.ones(nx), (np.tile(np.arange(self.h), self.o),
                             self.n+np.arange(nx))), shape=(self.h,n)).tocsr()
        au = vstack([au, link]); bu = np.r_[self.b_ub, np.zeros(nx)]
        ae = vstack([ae, assign]); be = np.r_[self.b_eq, np.ones(self.h)]
        if references is None:
            c = np.r_[self.costs[scenario], self.fixed[scenario]]
        else:
            refs = np.asarray(references)
            if refs.shape != (3,) or not np.isfinite(refs).all():
                raise ValueError('Three finite scenario references required')
            c = np.r_[np.zeros(n-1), 1.]
            au = vstack([au, csr_matrix(np.c_[self.costs, self.fixed, -np.ones(3)])])
            bu = np.r_[bu, refs]
        lower, upper = np.zeros(n), np.full(n, np.inf)
        upper[self.n:self.n+nx] = self.d.eligible.ravel()
        if nr: lower[-1] = -np.inf
        if fixed_assignment is not None:
            x = self.onehot(fixed_assignment).ravel()
            lower[self.n:self.n+nx] = x; upper[self.n:self.n+nx] = x
        integrality = np.zeros(n); integrality[self.n:self.n+nx] = 1
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning, message='Unrecognized options.*')
            result = milp(c, integrality=integrality, bounds=Bounds(lower, upper),
                          constraints=[LinearConstraint(au, -np.inf, bu), LinearConstraint(ae, be, be)],
                          options={'threads': 1, 'mip_rel_gap': 0., 'time_limit': time_limit})
        if not result.success or result.status != 0:
            raise RuntimeError(f'MILP has no certified optimum: {result.message}')
        x = result.x[self.n:self.n+nx].reshape(self.o,self.h)
        assignment = tuple(x.argmax(axis=0).tolist())
        costs = self.costs @ result.x[:self.n] + self.fixed @ x.ravel()
        return Solution(float(result.fun), assignment, result.x[:self.n], costs)

    def scenario_references(self):
        return np.array([self.solve_complete(scenario=s).objective for s in range(3)])

    def enumerate_optimum(self, references=None, limit=100000):
        choices = self.choices()
        import math
        if math.prod(map(len, choices)) > limit:
            raise ValueError('Enumeration exceeds safety limit')
        return min((self.solve_fixed(c, references) for c in itertools.product(*choices)),
                   key=lambda s: s.objective)

    def residual(self, solution):
        """Independent physical feasibility check on returned flows."""
        f = solution.flows; d = self.d
        qf, qr, u = f[self.qf], f[self.qr], f[self.u]
        incoming, outgoing = np.zeros(self.h), np.zeros(self.h)
        for k, (a,b) in enumerate(self.arcs):
            outgoing[a] += f[self.qh[k]]; incoming[b] += f[self.qh[k]]
        x = self.onehot(solution.assignment)
        return float(max(0., -f.min(), (qf.sum(1)-d.capacity).max(),
                         (qf-x*d.inbound).max(),
                         (qr.sum(1)+outgoing-qf.sum(0)-incoming-d.inventory).max(),
                         np.abs(qr.sum(0)+u-d.demand).max()))
