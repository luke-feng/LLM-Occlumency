"""Tests use the real packaged numerical inputs plus small counterexamples."""
import copy
from fractions import Fraction
from itertools import combinations
from pathlib import Path
import unittest

from analysis import statistics as s
from analysis.check_package import load
from analysis.replay import (CHECKS, CORE_TRACKS, equal, validate_runs,
                             validate_b4_cells)

ROOT = Path(__file__).resolve().parents[1]


class NumericalReplayTests(unittest.TestCase):
    def test_real_all_point_estimates(self):
        for name,check in CHECKS.items():
            with self.subTest(evidence=name):
                self.assertTrue(check(ROOT,False))

    def test_real_b1_frozen_bootstrap(self):
        self.assertIn('2,000',CHECKS['b1'](ROOT,True))

    def test_real_b3_frozen_bootstrap(self):
        self.assertIn('2,000',CHECKS['b3'](ROOT,True))

    def test_r1_upper_middle_counterexample(self):
        self.assertFalse(s.r1([(1,False),(3,False),(4,True),(100,False)]))
        self.assertTrue(s.r1([(1,False),(3,False),(4,False),(100,True)]))

    def test_r1_conservative_ties(self):
        self.assertFalse(s.r1([(1,False),(2,False),(2,True),(2,False)]))
        self.assertFalse(s.r1([]))
        self.assertFalse(s.r1([(None,False)]))

    def test_standard_even_median(self):
        self.assertEqual(s.median([1,2,3,4]),2.5)

    def test_percentiles_are_indices_not_interpolation(self):
        self.assertEqual(s.percentile([4,1,3,2],[0,2]),[1,3])
        with self.assertRaises(ValueError):
            s.percentile([1],[0,1])

    def test_wilson_invalid_denominator(self):
        for k,n in [(1,0),(3,2),(-1,4),(True,4)]:
            with self.subTest(k=k,n=n),self.assertRaises(ValueError):
                s.wilson(k,n)

    def test_small_coverage_by_enumeration(self):
        subsets = list(combinations(range(8),3))
        misses = sum(not ({0,1}&set(x)) for x in subsets)
        self.assertEqual(s.miss_probability(8,2,3),Fraction(misses,len(subsets)))
        self.assertEqual(s.miss_probability(8,0,3),1)
        self.assertEqual(s.miss_probability(8,8,3),0)

    def test_invalid_coverage_counts(self):
        for args in [(8,-1,3),(8,9,3),(8,1,9),(8,True,3)]:
            with self.subTest(args=args),self.assertRaises(ValueError):
                s.miss_probability(*args)

    def test_b1_denominator_is_not_eight_for_all_runs(self):
        doc = load(ROOT/'data/evidence_v3/b1_analysis.json')
        r = next(r for r in doc['runs'] if r['admitted_cells']==7 and any(c['y_by_q']['256'] for c in s.admitted(r)))
        hits = sum(c['y_by_q']['256'] for c in s.admitted(r))
        self.assertNotEqual(s.b1_run_rate(r,256),1-hits/8)

    def test_invalid_or_missing_binary_indicator_rejected(self):
        original = load(ROOT/'data/evidence_v3/b1_analysis.json')
        for value in [None,1,'false']:
            runs = copy.deepcopy(original['runs'])
            runs[0]['cells'][0]['y_by_q']['128'] = value
            with self.subTest(value=value),self.assertRaises(ValueError):
                validate_runs(runs,original['budget_grid'],'B1')
        runs = copy.deepcopy(original['runs'])
        del runs[0]['cells'][0]['y_by_q']['256']
        with self.assertRaises(ValueError):
            validate_runs(runs,original['budget_grid'],'B1')

    def test_invalid_admission_and_unsupported_endpoint(self):
        original = load(ROOT/'data/evidence_v3/b1_analysis.json')
        for field,value in [('admitted',1),('supported',False)]:
            runs = copy.deepcopy(original['runs'])
            cell = next(c for c in runs[0]['cells'] if c['admitted'])
            cell[field] = value
            with self.subTest(field=field),self.assertRaises(ValueError):
                validate_runs(runs,original['budget_grid'],'B1')

    def test_wrong_precision_and_denominator_rejected(self):
        original = load(ROOT/'data/evidence_v3/b1_analysis.json')
        for field,value in [('track','Qwen3-14B'),('admitted_cells',8)]:
            runs = copy.deepcopy(original['runs'])
            row = next(r for r in runs if r['admitted_cells']==7)
            row[field] = value
            with self.subTest(field=field),self.assertRaises(ValueError):
                validate_runs(runs,original['budget_grid'],'B1')

    def test_b3_invalid_status(self):
        doc = load(ROOT/'data/evidence_v3/b3_analysis.json')
        cell = next(c for c in doc['per_run'][0]['cells'] if c['admitted'])
        cell['status'] = True
        with self.assertRaises(ValueError):
            validate_runs(doc['per_run'],[16,32,64,128],'B3')

    def test_safe_cells_cannot_enter_coverage_denominator(self):
        doc = load(ROOT/'data/evidence_v3/b4_analysis.json')
        original = load(ROOT/'data/evidence_v3/b4_coverage.json')
        cov = copy.deepcopy(original)
        cov['summary']['n_cells'] = 64
        with self.assertRaises(ValueError):
            validate_b4_cells(doc,cov)
        cov = copy.deepcopy(original)
        cov['unsafe_cells'][0] = cov['safe_cells'][0]
        with self.assertRaises(ValueError):
            validate_b4_cells(doc,cov)

    def test_exact_integer_comparison_not_truthiness(self):
        with self.assertRaises(ValueError):
            equal(True,1)
        with self.assertRaises(ValueError):
            equal(423,456)

    def test_tiny_positive_rational_cannot_be_rounded_to_zero(self):
        actual = s.rational(Fraction(1,10**23))
        corrupted = dict(actual,float=0.0)
        with self.assertRaisesRegex(ValueError,'rational binary64'):
            equal(actual,corrupted)


if __name__=='__main__':
    unittest.main()
