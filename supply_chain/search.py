"""Distinct-LP-budget SA, GA, GNN-GA, and initialization ablations."""
import itertools
import math
import numpy as np
from scipy.stats import rankdata


def entropy_mutation(p, choices):
    rates = np.zeros(p.shape[1])
    for h,options in enumerate(choices):
        if len(options) > 1:
            values = p[options,h]
            entropy = -np.sum(values*np.log(np.maximum(values,1e-300)))/np.log(len(options))
            rates[h] = min(.20,.05*(.5+1.5*entropy))
    return rates


def cost_probabilities(d):
    """Chosen rank-sum softmax; precise author warm-start formula unpublished."""
    p = np.zeros_like(d.supply_cost)
    for h in range(d.shape[1]):
        options = np.flatnonzero(d.eligible[:,h])
        ranks = rankdata(d.supply_cost[options,h]) + rankdata(d.fixed_cost[options,h])
        weights = np.exp(-(ranks-ranks.min()))
        p[options,h] = weights/weights.sum()
    return p


class Evaluator:
    def __init__(self, model, budget, references=None):
        if budget < 1: raise ValueError('Budget must be positive')
        self.model,self.budget,self.references = model,budget,references
        self.cache,self.trace = {},[]

    def evaluate(self, chromosome):
        key = tuple(int(v) for v in chromosome)
        if key not in self.cache:
            if len(self.cache) >= self.budget: raise RuntimeError('Distinct evaluation budget exhausted')
            solution = self.model.solve_fixed(key,self.references)
            self.cache[key] = solution.objective
            best = min(self.cache.values())
            self.trace.append({'evaluations':len(self.cache),'best_objective':best})
        return self.cache[key]


def search(model, method, budget, seed, population=100, p=None, references=None):
    if method not in {'sa','ga','gnn_ga','initialization_only','entropy_only','cost'}:
        raise ValueError('Unknown search method')
    rng = np.random.default_rng(seed)
    choices = model.choices(); size = math.prod(map(len,choices))
    budget = min(budget,size)
    if budget < 1 or population < 1: raise ValueError('Positive budget/population required')
    effective = min(population,budget,size)
    if effective == 1 and budget > 1 and method != 'sa':
        raise ValueError('GA population must be >= 2 when budget > 1')
    evaluator = Evaluator(model,budget,references)
    movable = [h for h,c in enumerate(choices) if len(c)>1]
    guided = method in {'gnn_ga','initialization_only','cost'}
    if method == 'cost': p = cost_probabilities(model.d)
    if guided or method == 'entropy_only':
        if p is None or p.shape != model.d.eligible.shape or not np.isfinite(p).all():
            raise ValueError('Finite factory-by-hub probabilities required')
        if (p < 0).any() or not np.allclose(p.sum(0),1) or (p[~model.d.eligible] != 0).any():
            raise ValueError('Invalid assignment probabilities')
    def sample(guidance=False):
        return tuple(int(rng.choice(c,p=p[c,h] if guidance else None)) for h,c in enumerate(choices))
    # After many duplicates, draw an unseen candidate. This is a disclosed
    # termination policy, not claimed to match the unavailable author code.
    iterator = itertools.product(*choices) if size <= 100000 else None
    fallback_count = 0
    def fresh():
        nonlocal fallback_count
        fallback_count += 1
        for _ in range(1000):
            candidate = sample()
            if candidate not in evaluator.cache: return candidate
        if iterator is not None:
            return next(c for c in iterator if c not in evaluator.cache)
        raise RuntimeError('Unable to sample unseen candidate in huge assignment space')
    generations,partial = 0,False
    if method == 'sa':
        current = sample(); score = evaluator.evaluate(current)
        # Paper does not disclose annealing schedule; these are explicit choices.
        initial_temp = max(.01*abs(score),1.)
        duplicates = 0
        while len(evaluator.cache) < budget:
            candidate = list(current); h = int(rng.choice(movable))
            candidate[h] = int(rng.choice([v for v in choices[h] if v != current[h]]))
            candidate = tuple(candidate)
            duplicates = duplicates+1 if candidate in evaluator.cache else 0
            if duplicates >= 100:
                candidate = fresh(); duplicates = 0
            value = evaluator.evaluate(candidate)
            progress = len(evaluator.cache)/budget
            temp = initial_temp * (.001 ** progress)
            if value <= score or rng.random() < np.exp(np.clip((score-value)/temp,-745,0)):
                current,score = candidate,value
    else:
        pop = []
        n_guided = max(1,int(np.floor(.8*effective))) if guided else 0
        duplicates = 0
        while len(pop) < effective:
            candidate = tuple(p.argmax(0).tolist()) if guided and not pop else sample(len(pop)<n_guided)
            if candidate in evaluator.cache:
                duplicates += 1
                if duplicates < 100: continue
                candidate = fresh()
            duplicates = 0
            evaluator.evaluate(candidate); pop.append(candidate)
        initial_evaluations = len(evaluator.cache)
        rates = (entropy_mutation(p,choices) if method in {'gnn_ga','entropy_only'}
                 else np.array([.05 if h in movable else 0. for h in range(model.h)]))
        elite = max(1,int(math.ceil(.05*effective)))
        def tournament():
            contenders = [pop[i] for i in rng.integers(0,len(pop),3)]
            return min(contenders,key=evaluator.cache.get)
        while len(evaluator.cache) < budget:
            next_pop = sorted(pop,key=evaluator.cache.get)[:elite]
            duplicates = 0
            while len(next_pop) < effective and len(evaluator.cache) < budget:
                a,b = tournament(),tournament()
                child = np.array(a)
                if rng.random() < .8:
                    take_b = rng.random(model.h) < .5
                    child[take_b] = np.asarray(b)[take_b]
                for h in movable:
                    if rng.random() < rates[h]:
                        child[h] = rng.choice([v for v in choices[h] if v != child[h]])
                candidate = tuple(child.tolist())
                if candidate in evaluator.cache:
                    duplicates += 1
                    if duplicates < 100: continue
                    candidate = fresh()
                duplicates = 0
                evaluator.evaluate(candidate); next_pop.append(candidate)
            if len(next_pop) == effective: generations += 1
            else: partial = True
            pop = next_pop
    best = min(evaluator.cache,key=evaluator.cache.get)
    return {'objective':evaluator.cache[best], 'assignment':list(best),
            'evaluations':len(evaluator.cache), 'requested_budget':budget,
            'effective_population':effective if method != 'sa' else 1,
            'initial_evaluations':initial_evaluations if method != 'sa' else 1,
            'full_generations':generations,'partial_generation':partial,
            'fallback_count':fallback_count,'trace':evaluator.trace}
