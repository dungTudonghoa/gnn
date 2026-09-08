import unittest
import numpy as np
import torch
from supply_chain.data import generate, Instance
from supply_chain.optimization import FlowModel
from supply_chain.search import search, entropy_mutation, Evaluator
from supply_chain.gnn import AssignmentGNN, Standardizer, probabilities

class OptimizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = FlowModel(generate('tiny',(2,3,4),43))
        cls.refs = cls.model.scenario_references()

    def test_milp_matches_enumeration_and_fixed_model(self):
        for refs in (None,self.refs):
            exact = self.model.solve_complete(refs)
            exhaustive = self.model.enumerate_optimum(refs)
            self.assertAlmostEqual(exact.objective,exhaustive.objective,places=6)
            for assignment in [(0,0,0),(0,1,0),(1,1,1)]:
                lp = self.model.solve_fixed(assignment,refs)
                fixed = self.model.solve_complete(refs,fixed_assignment=assignment)
                self.assertAlmostEqual(lp.objective,fixed.objective,places=6)
                self.assertLess(self.model.residual(lp),1e-6)
                if refs is not None:
                    self.assertAlmostEqual(lp.objective,max(lp.scenario_costs-refs),places=6)

    def test_hand_computed_flow_shortage_and_inventory(self):
        d = Instance('hand',np.array([5.]),np.array([10.]),np.array([2.]),np.array([10.]),
                     np.array([[2.]]),np.array([[3.]]),np.zeros((1,1)),np.array([[1.]]),
                     np.array([10.]),np.ones((1,1),dtype=bool))
        model = FlowModel(d)
        solution = model.solve_fixed((0,))
        self.assertAlmostEqual(solution.objective,50.)
        np.testing.assert_allclose(solution.flows,[5,7,3])
        self.assertLess(model.residual(solution),1e-8)
        d.inventory[:] = 20
        self.assertAlmostEqual(FlowModel(d).solve_fixed((0,)).objective,13.)

    def test_regret_uses_shared_flows(self):
        d = Instance('conflicting-scenarios', np.array([10.]), np.array([10.]),
                     np.array([0.]), np.array([10.]), np.array([[100.]]),
                     np.array([[3.]]), np.zeros((1,1)), np.array([[20.]]),
                     np.array([130.]), np.ones((1,1), dtype=bool))
        model = FlowModel(d); refs = model.scenario_references()
        solution = model.solve_fixed((0,), refs)
        # Lower scenario prefers shortage, upper scenario prefers delivery.
        # Shared shipment q balances 7*q and 27*(10-q).
        self.assertAlmostEqual(solution.flows[0], 270/34, places=6)
        self.assertAlmostEqual(solution.objective, 7*270/34, places=6)
        self.assertAlmostEqual(model.solve_complete(refs).objective, solution.objective, places=6)

    def test_eligibility_and_cache(self):
        d = generate('sparse',(2,3,4),9); d.eligible[1,0] = False
        model = FlowModel(d)
        with self.assertRaises(ValueError): model.solve_fixed((1,0,0))
        e = Evaluator(model,1)
        e.evaluate((0,0,0)); e.evaluate((0,0,0))
        self.assertEqual(len(e.cache),1)
        with self.assertRaises(RuntimeError): e.evaluate((0,1,0))

class SearchTests(unittest.TestCase):
    def test_budgets_singleton_and_exhaustion(self):
        for shape in [(1,3,2),(2,3,4)]:
            model = FlowModel(generate('test',shape,42))
            p = model.d.eligible.astype(float)/shape[0]
            optimum = model.solve_complete().objective
            for method in ['sa','ga','gnn_ga','initialization_only','entropy_only','cost']:
                result = search(model,method,100,1001,population=4,p=p)
                self.assertEqual(result['evaluations'],shape[0]**shape[1])
                self.assertAlmostEqual(result['objective'],optimum,places=6)
                self.assertTrue(all(a['best_objective'] >= b['best_objective']
                                    for a,b in zip(result['trace'],result['trace'][1:])))

    def test_tight_budget_guidance_and_entropy(self):
        model = FlowModel(generate('test',(3,5,4),42))
        p = np.full((3,5),1/3)
        np.testing.assert_allclose(entropy_mutation(p,model.choices()),.1)
        a = search(model,'gnn_ga',7,1001,population=100,p=p)
        b = search(model,'initialization_only',7,1001,population=100,p=p)
        self.assertEqual(a,b)
        self.assertEqual(a['full_generations'],0)
        self.assertEqual(a['initial_evaluations'],7)
        np.testing.assert_allclose(entropy_mutation(np.ones((1,2)),[[0],[0]]),0)

class GraphTests(unittest.TestCase):
    def test_factory_permutation_and_mask(self):
        d = generate('test',(3,4,5),42); d.eligible[0,1] = False
        norm = Standardizer().fit([d])
        torch.manual_seed(3101); model = AssignmentGNN()
        p = probabilities(model,norm,d)
        np.testing.assert_allclose(p.sum(0),1,atol=1e-6)
        self.assertEqual(p[0,1],0)
        permutation = [2,0,1]
        import copy
        changed = copy.deepcopy(d)
        for key in ['capacity','supply_cost','fixed_cost','eligible']:
            setattr(changed,key,getattr(changed,key)[permutation])
        np.testing.assert_allclose(probabilities(model,norm,changed),p[permutation],atol=1e-6)

if __name__ == '__main__': unittest.main()
