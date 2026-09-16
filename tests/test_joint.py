"""Frozen public label checks; fixtures contain only inert records."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from analysis.check_package import load
from analysis.joint import (canary_rows, check_joint, format_markdown, main,
                            replay_joint, retabulate, validate_joint)

ROOT = Path(__file__).resolve().parents[1]


class JointReplayTests(unittest.TestCase):
    def setUp(self):
        self.doc = load(ROOT/'data/evidence_v3/b1b4_joint_analysis.json')
        self.b1 = load(ROOT/'data/evidence_v3/b1_analysis.json')
        self.b4 = load(ROOT/'data/evidence_v3/b4_analysis.json')

    def validate(self):
        return validate_joint(self.doc, self.b1, self.b4)

    def test_valid_public_input(self):
        result = self.validate()
        self.assertEqual(len(result['identity_roster']), 66)
        self.assertEqual(len(result['validated_pairs']), 30)
        self.assertEqual(len(result['groups']), 6)
        self.assertEqual(result['summary']['table_2x3'], {
            'SAFE': {'recovered_by_16': 0, 'first_recovered_17_to_256': 0,
                     'not_recovered_by_256': 5},
            'UNSAFE': {'recovered_by_16': 6, 'first_recovered_17_to_256': 9,
                       'not_recovered_by_256': 10}})
        self.assertEqual(result['summary']['table_2x2_at_256'], {
            'SAFE': {'recovered_by_256': 0, 'not_recovered_by_256': 5},
            'UNSAFE': {'recovered_by_256': 15, 'not_recovered_by_256': 10}})
        self.assertEqual(result['summary']['unsafe_without_b1_recovery']['ratio'], '10/25')
        self.assertFalse(result['source_identity_eligibility_independently_reverified'])
        self.assertFalse(result['same_executable_established'])

    def test_complete_canary_id_rows(self):
        rows = canary_rows(self.validate())
        fields = ('safe_not_recovered_by_256', 'unsafe_recovered_by_16',
                  'unsafe_first_recovered_17_to_256', 'unsafe_not_recovered_by_256')
        expected = [
            ([0], [], [4], []),
            ([0, 4], [], [], []),
            ([4], [], [0], []),
            ([], [0, 2, 4], [3, 5, 6, 7], [1]),
            ([3], [4, 5], [2, 6], [0, 1, 7]),
            ([], [1], [2], [0, 3, 4, 5, 6, 7]),
        ]
        for row, bins in zip(rows, expected):
            self.assertEqual([row[field] for field in fields],
                             [['canary' + str(i) for i in ids] for ids in bins])
            self.assertEqual(sum(len(row[field]) for field in fields), row['n_pairs'])
        self.assertEqual([row['n_pairs'] for row in rows], [2, 2, 2, 8, 8, 8])

    def test_manifest_bound_replay_and_markdown(self):
        result = replay_joint(ROOT)
        rendered = format_markdown(result)
        self.assertIn('| SAFE | 0 | 0 | 5 | 5 |', rendered)
        self.assertIn('| UNSAFE | 6 | 9 | 10 | 25 |', rendered)
        self.assertIn('| Total | 6 | 9 | 15 | 30 |', rendered)
        self.assertIn('canary0, canary3, canary4, canary5, canary6, canary7', rendered)
        self.assertIn('source eligibility not independently reverified', check_joint(ROOT))
        self.assertEqual(check_joint(ROOT), check_joint(ROOT, True))

    def test_cli_from_unrelated_working_directory(self):
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary:
            try:
                os.chdir(temporary)
                for arguments in (['--format', 'json'],
                                  ['--root', str(ROOT), '--format', 'json']):
                    with patch('sys.argv', ['joint.py'] + arguments), patch('builtins.print') as output:
                        main()
                    result = json.loads(output.call_args.args[0])
                    self.assertEqual(result['summary']['n_pairs'], 30)
                    self.assertEqual(len(result['canary_id_rows']), 6)
            finally:
                os.chdir(original)

    def test_duplicate_pair_rejected(self):
        self.doc['validated_pairs'].append(copy.deepcopy(self.doc['validated_pairs'][0]))
        with self.assertRaisesRegex(ValueError, 'duplicate joint pair'):
            self.validate()

    def test_missing_pair_rejected(self):
        self.doc['validated_pairs'].pop()
        with self.assertRaisesRegex(ValueError, 'missing/extra joint pair'):
            self.validate()

    def test_duplicate_roster_rejected(self):
        self.doc['identity_roster'][-1] = copy.deepcopy(self.doc['identity_roster'][0])
        with self.assertRaisesRegex(ValueError, 'duplicate joint roster'):
            self.validate()

    def test_missing_roster_rejected(self):
        self.doc['identity_roster'].pop()
        with self.assertRaisesRegex(ValueError, 'roster count'):
            self.validate()

    def test_inconsistent_label_rejected(self):
        self.doc['validated_pairs'][0]['b4_label'] = 'SAFE'
        with self.assertRaisesRegex(ValueError, 'B4 recorded label mismatch'):
            self.validate()

    def test_invalid_label_rejected(self):
        self.doc['validated_pairs'][0]['b4_label'] = 'UNKNOWN'
        with self.assertRaisesRegex(ValueError, 'invalid joint B4 label'):
            self.validate()

    def test_inconsistent_endpoint_rejected(self):
        self.doc['validated_pairs'][0]['b1_by_16'] = False
        with self.assertRaisesRegex(ValueError, 'B1 endpoint label mismatch'):
            self.validate()

    def test_nonmonotone_prefix_rejected(self):
        self.doc['validated_pairs'][0]['b1_by_256'] = False
        with self.assertRaisesRegex(ValueError, 'nonmonotone joint B1 prefix'):
            self.validate()

    def test_non_boolean_prefix_rejected(self):
        self.doc['validated_pairs'][0]['b1_by_16'] = 1
        with self.assertRaisesRegex(ValueError, 'invalid joint B1 prefix label'):
            self.validate()

    def test_inconsistent_status_rejected(self):
        self.doc['identity_roster'][0]['identity_status'] = 'b1_only'
        with self.assertRaisesRegex(ValueError, 'identity status inconsistent'):
            self.validate()

    def test_inconsistent_admission_rejected(self):
        self.doc['identity_roster'][0]['b1_admitted'] = False
        with self.assertRaisesRegex(ValueError, 'B1 admission flag mismatch'):
            self.validate()

    def test_public_link_mismatch_rejected(self):
        self.doc['validated_pairs'][0]['b1_public_run_id'] = 'B1-00'
        with self.assertRaisesRegex(ValueError, 'pair public link mismatch'):
            self.validate()

    def test_public_b4_label_change_rejected(self):
        public_id = self.doc['validated_pairs'][0]['b4_public_cell_id']
        cell = next(cell for cell in self.b4['cells'] if cell['public_cell_id'] == public_id)
        cell['outcome'] = 'SAFE'
        with self.assertRaisesRegex(ValueError, 'B4 recorded label mismatch'):
            self.validate()

    def test_missing_public_b1_prefix_rejected(self):
        public_id = self.doc['validated_pairs'][0]['b1_public_run_id']
        run = next(run for run in self.b1['runs'] if run['public_run_id'] == public_id)
        del run['cells'][0]['y_by_q']['256']
        with self.assertRaisesRegex(ValueError, 'missing/extra B1 prefix'):
            self.validate()

    def test_wrong_summary_or_group_rejected(self):
        original = copy.deepcopy(self.doc)
        for field in ('summary', 'groups'):
            self.doc = copy.deepcopy(original)
            record = self.doc[field][0] if field == 'groups' else self.doc[field]
            record['n_pairs'] += 1
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'frozen mismatch'):
                self.validate()

    def test_no_private_or_unrecognized_fields(self):
        self.doc['identity_roster'][0]['source_receipt'] = 'omitted'
        with self.assertRaisesRegex(ValueError, 'unexpected/missing joint roster fields'):
            self.validate()

    def test_execution_scope_cannot_be_upgraded(self):
        self.doc['source_identity_eligibility_independently_reverified'] = True
        with self.assertRaisesRegex(ValueError, 'frozen mismatch'):
            self.validate()

    def test_empty_unsafe_denominator_is_not_zero_ratio(self):
        summary = retabulate([])
        self.assertEqual(summary['unsafe_without_b1_recovery'],
                         {'x': 0, 'y': 0, 'ratio': None, 'ratio_status': 'undefined'})


if __name__ == '__main__':
    unittest.main()
