"""Re-solve every reported best assignment and enumerate two small fixtures."""
import json
from pathlib import Path
from .data import Instance
from .optimization import FlowModel


def verify(directory='supply_chain/results/reconstruction'):
    root=Path(directory)
    models={}; refs=json.loads((root/'references.json').read_text()); rows=json.loads((root/'runs.json').read_text())
    max_error=max_residual=0.
    for row in rows:
        name=row['instance']
        if name not in models: models[name]=FlowModel(Instance.load(root/'instances'/f'{name}.json'))
        references=refs[name]['scenario_optima'] if row['formulation']=='regret' else None
        result=models[name].solve_fixed(tuple(row['assignment']),references)
        max_error=max(max_error,abs(result.objective-row['objective']))
        max_residual=max(max_residual,models[name].residual(result))
        assert row['evaluations']==row['budget']
        assert row['objective']>=row['reference']-1e-6
        if row['method']!='sa' and row['budget']<100:
            assert row['full_generations']==0
    assert max_error<1e-6 and max_residual<1e-6
    checks=[]
    for i in [3,4]:
        model=FlowModel(Instance.load(root/'instances'/f'reconstructed-train-{i:02d}.json'))
        references=model.scenario_references()
        for formulation,r in [('deterministic',None),('regret',references)]:
            exact=model.solve_complete(r); enumerated=model.enumerate_optimum(r)
            error=abs(exact.objective-enumerated.objective)
            assert error<1e-6
            checks.append({'fixture':model.d.name,'formulation':formulation,
                           'assignments':model.o**model.h,'absolute_difference':error})
    verification={'runs_resolved':len(rows),'max_objective_difference':max_error,
                  'max_physical_residual':max_residual,'enumeration_checks':checks}
    (root/'verification.json').write_text(json.dumps(verification,indent=2))
    print(json.dumps(verification,indent=2))


if __name__=='__main__': verify()
