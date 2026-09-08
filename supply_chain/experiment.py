"""Run the disclosed reconstruction or a supplied JSON dataset manifest."""
import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
import scipy
from scipy.stats import wilcoxon, friedmanchisquare
import torch
from .data import generate, Instance
from .optimization import FlowModel
from .gnn import train, probabilities
from .search import search

METHODS = ['sa','ga','gnn_ga','initialization_only','entropy_only','cost']


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False))


def holm(values):
    order = np.argsort(values); adjusted = np.zeros(len(values)); maximum = 0.
    for rank,index in enumerate(order):
        maximum = max(maximum,(len(values)-rank)*values[index])
        adjusted[index] = min(1.,maximum)
    return adjusted.tolist()


def statistics(rows):
    summary,pairs,friedman = [],[],[]
    keys = sorted({(row['instance'],row['formulation'],row['budget']) for row in rows})
    for name,form,budget in keys:
        group = [r for r in rows if (r['instance'],r['formulation'],r['budget'])==(name,form,budget)]
        by_method = {}
        for method in sorted({r['method'] for r in group}):
            subset = sorted([r for r in group if r['method']==method],key=lambda r:r['seed'])
            values = np.array([r['objective'] for r in subset]); by_method[method] = values
            reference = subset[0]['reference']
            summary.append(dict(instance=name,formulation=form,budget=budget,method=method,
                                reference=reference,mean=float(values.mean()),
                                sd=float(values.std(ddof=1)) if len(values)>1 else 0.,
                                exact_successes=int(np.sum(np.abs(values-reference)<=1e-6)),
                                runs=len(values),mean_seconds=float(np.mean([r['seconds'] for r in subset])),
                                mean_absolute_deviation=float(np.mean(values-reference)),
                                mean_gap_percent=float(100*np.mean(values-reference)/abs(reference)) if abs(reference)>1e-6 else None))
        comparisons = [('sa','ga'),('sa','gnn_ga'),('ga','gnn_ga')] if 'sa' in by_method else [('ga','gnn_ga')]
        family = []
        for a,b in comparisons:
            difference = by_method[a]-by_method[b]
            difference[np.abs(difference)<1e-6] = 0
            p = float(wilcoxon(difference,method='auto').pvalue) if np.any(difference) else 1.
            family.append(dict(instance=name,formulation=form,budget=budget,a=a,b=b,p=p,
                               b_wins=int(np.sum(difference>0)),ties=int(np.sum(difference==0)),
                               a_wins=int(np.sum(difference<0))))
        for item,adjusted in zip(family,holm([item['p'] for item in family])):
            item['p_holm_within_comparison_family'] = adjusted
        pairs.extend(family)
        if 'sa' in by_method:
            matrix = np.stack([by_method[m] for m in ['sa','ga','gnn_ga']])
            p = 1. if np.all(np.ptp(matrix,axis=0)<1e-6) else float(friedmanchisquare(*matrix).pvalue)
            friedman.append(dict(instance=name,formulation=form,budget=budget,p=p))
    for item,adjusted in zip(friedman,holm([item['p'] for item in friedman])):
        item['p_holm_across_fixture_formulations'] = adjusted
    return {'summary':summary,'paired_wilcoxon':pairs,'friedman':friedman}


def demo_manifest(directory):
    """The dimensions below are our fixtures, NEVER author Instances 1-15."""
    training_shapes = [(1,3,2),(1,5,7),(2,6,10),(2,8,12),(3,6,12),
                       (3,7,15),(3,8,18),(4,8,20),(4,9,22)]
    validation_shapes = [(3,9,20),(4,10,25),(4,11,28)]
    test_shapes = [(3,10,25),(4,12,30),(5,15,40)]
    manifest = {'provenance':'independent_reconstruction_not_author_benchmarks','instances':[]}
    for role,shapes in [('train',training_shapes),('validation',validation_shapes),('test',test_shapes)]:
        for index,shape in enumerate(shapes,1):
            name = f'reconstructed-{role}-{index:02d}'
            seed = (202600+index if role=='test' else 42+index+(100 if role=='validation' else 0))
            instance = generate(name,shape,seed)
            path = directory/'instances'/f'{name}.json'; instance.save(path)
            manifest['instances'].append({'name':name,'path':str(path.relative_to(directory)),
                                          'role':role,'seed':seed,'shape':list(shape),
                                          'budget': [75,25,10][index-1] if role=='test' else None,
                                          'population':100})
    write_json(directory/'manifest.json',manifest)
    return directory/'manifest.json'


def execute(output, manifest_path=None, seeds=20, epochs=100, sensitivity=True):
    if seeds < 2: raise ValueError('At least two matched seeds required for statistics')
    torch.set_num_threads(1)
    output = Path(output); output.mkdir(parents=True,exist_ok=True)
    if manifest_path is None: manifest_path = demo_manifest(output)
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    records = manifest['instances']
    if len({r['name'] for r in records}) != len(records): raise ValueError('Duplicate names')
    datasets = {r['name']:Instance.load(manifest_path.parent/r['path']) for r in records}
    if any(datasets[r['name']].name != r['name'] for r in records): raise ValueError('Name mismatch')
    if any(r['role'] not in {'train','validation','test'} for r in records): raise ValueError('Unknown role')
    training = [datasets[r['name']] for r in records if r['role']=='train']
    validation = [datasets[r['name']] for r in records if r['role']=='validation']
    testing = [r for r in records if r['role']=='test']
    if not testing: raise ValueError('Test fixtures required')
    # Models, exact references and labels are prepared outside search timers.
    labels = {}
    start = time.perf_counter()
    for d in training+validation:
        labels[d.name] = list(FlowModel(d).solve_complete().assignment)
        print('Exact label:',d.name,flush=True)
    label_seconds = time.perf_counter()-start
    write_json(output/'labels.json',labels)
    start = time.perf_counter()
    model,norm,history,best_epoch = train(training,validation,labels,epochs=epochs)
    training_seconds = time.perf_counter()-start
    torch.save({'state_dict':model.state_dict(),'normalizer':norm.state_dict(),'hidden':32,'rounds':2},output/'model.pt')
    write_json(output/'training.json',{'history':history,'best_epoch':best_epoch,
                                     'label_seconds':label_seconds,'training_seconds':training_seconds})
    print('Training complete, best epoch:',best_epoch,flush=True)
    rows = []; reference_records = {}
    for test_index,record in enumerate(testing):
        d = datasets[record['name']]; lp = FlowModel(d)
        refs = lp.scenario_references()
        nominal = lp.solve_complete(); robust = lp.solve_complete(refs)
        reference_records[d.name] = {'scenario_optima':refs.tolist(),
                                     'nominal':nominal.objective,'regret':robust.objective,
                                     'nominal_assignment':list(nominal.assignment),
                                     'regret_assignment':list(robust.assignment),'status':'solver_certified_optimal'}
        write_json(output/'references.json',reference_records)
        configurations = [(record['budget'],METHODS,seeds)]
        if sensitivity and test_index == 0: configurations.append((400,['ga','gnn_ga'],min(seeds,10)))
        for budget,methods,count in configurations:
            for form,reference,scenario_refs in [('deterministic',nominal.objective,None),('regret',robust.objective,refs)]:
                for seed in range(1001,1001+count):
                    for method in methods:
                        start = time.perf_counter()
                        p = probabilities(model,norm,d) if method in {'gnn_ga','initialization_only','entropy_only'} else None
                        result = search(lp,method,budget,seed,record.get('population',100),p,scenario_refs)
                        elapsed = time.perf_counter()-start
                        result.pop('trace')
                        rows.append(dict(instance=d.name,formulation=form,budget=budget,method=method,
                                         seed=seed,reference=reference,seconds=elapsed,**result))
                write_json(output/'runs.json',rows)
                print(d.name,form,'budget',budget,'completed',count,'seeds',flush=True)
    stats = statistics(rows); write_json(output/'statistics.json',stats)
    with (output/'summary.csv').open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=stats['summary'][0].keys()); writer.writeheader(); writer.writerows(stats['summary'])
    hashes = {r['name']:hashlib.sha256((manifest_path.parent/r['path']).read_bytes()).hexdigest() for r in records}
    write_json(output/'environment.json',{'python':platform.python_version(),'torch':torch.__version__,
                                         'numpy':np.__version__,'scipy':scipy.__version__,
                                         'threads':1,'manifest_provenance':manifest.get('provenance','user_supplied'),
                                         'input_sha256':hashes,'search_seeds':list(range(1001,1001+seeds)),
                                         'training_seed':3101})
    from .report import render
    render(output)
    print('Saved:',output,flush=True)
    return stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='supply_chain/results/reconstruction')
    parser.add_argument('--manifest',help='JSON manifest with train/validation/test roles and paths')
    parser.add_argument('--seeds',type=int,default=20)
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--no-sensitivity',action='store_true')
    args = parser.parse_args()
    execute(args.output,args.manifest,args.seeds,args.epochs,not args.no_sensitivity)
