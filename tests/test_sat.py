"""Tiny pure-CNF tests. No python-sat import, solver build or real experiment."""
from itertools import product
import random
import unittest

from sat import sat_hardness as sat


def satisfiable(cnf):
    n = max((abs(x) for clause in cnf for x in clause),default=0)
    if n>8:
        raise ValueError('test truth-table bound exceeded')
    for bits in product([False,True],repeat=n):
        if all(any(bits[abs(x)-1]==(x>0) for x in clause) for clause in cnf):
            return True
    return False


class TinySATTests(unittest.TestCase):
    def test_pigeonhole_tiny_unsat(self):
        for holes in [1,2]:
            self.assertFalse(satisfiable(sat.pigeonhole_cnf(holes)))

    def test_xor_truth_table(self):
        for rhs in [0,1]:
            cnf = sat._xor_to_cnf([1,2,3],rhs)
            for bits in product([False,True],repeat=3):
                value = all(any(bits[abs(x)-1]==(x>0) for x in clause) for clause in cnf)
                self.assertEqual(value,sum(bits)%2==rhs)

    def test_random_cnf_deterministic_seed(self):
        left = sat.random_3sat_cnf(6,10,random.Random(3))
        right = sat.random_3sat_cnf(6,10,random.Random(3))
        self.assertEqual(left,right)
        self.assertTrue(all(len({abs(x) for x in clause})==3 for clause in left))

    def test_tiny_tseitin_odd_charge(self):
        cnf,n,metadata = sat.tseitin_parity_cnf(4,degree=3,seed=0)
        self.assertEqual(n,6)
        self.assertFalse(satisfiable(cnf))
        self.assertEqual(len(metadata['edges']),6)


if __name__=='__main__':
    unittest.main()
