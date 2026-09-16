import ast
import hashlib
import shutil
from pathlib import Path
import tempfile
import unittest

from analysis.check_package import (CURRENT_FIGURES,LEGACY_FIGURES,SOURCE_FILES,
                                    check_imports,check_package,
                                    inventory,json_text)

ROOT = Path(__file__).resolve().parents[1]


class PackageTests(unittest.TestCase):
    def test_current_package(self):
        self.assertEqual(check_package(ROOT)['source_files'],53)

    def test_current_and_legacy_figure_inventory(self):
        self.assertEqual(CURRENT_FIGURES,{
            'figures/results_map.pdf','figures/main_evidence_b1_b4_focused.pdf'})
        self.assertEqual(len(LEGACY_FIGURES),6)
        self.assertFalse(CURRENT_FIGURES & LEGACY_FIGURES)

    def test_software_license_scope(self):
        license_text=(ROOT/'LICENSE').read_text()
        readme=(ROOT/'README.md').read_text()
        self.assertTrue(license_text.startswith('MIT License\n'))
        self.assertIn('Permission is hereby granted, free of charge',license_text)
        self.assertIn('[MIT License](LICENSE)',readme)
        self.assertNotIn('License not yet specified',readme)
        self.assertIn('Data files and model weights are outside',readme)

    def test_current_renderer_public_hash_bindings(self):
        # Parse constants only; core tests never import Matplotlib/render figures.
        for filename,constant in [('plot_a1a2_main_v3.py','A1A2_SHA256'),
                                  ('plot_main_evidence_b1_b4_v3.py','SOURCES'),
                                  ('plot_main_evidence_b1_b4_focused.py','SOURCES')]:
            tree=ast.parse((ROOT/'figures'/filename).read_text())
            assignments=[node for node in tree.body if isinstance(node,ast.Assign)
                         and any(isinstance(target,ast.Name) and target.id==constant
                                 for target in node.targets)]
            self.assertEqual(len(assignments),1)
            value=ast.literal_eval(assignments[0].value)
            bindings={'a1a2_analysis.json':value} if constant=='A1A2_SHA256' else value
            if constant=='A1A2_SHA256':
                expected={'a1a2_analysis.json'}
            elif filename=='plot_main_evidence_b1_b4_focused.py':
                expected={'b1_analysis.json','b4_analysis.json'}
            else:
                expected={'b1_analysis.json','b2_analysis.json','b3_analysis.json','b4_analysis.json'}
            self.assertEqual(set(bindings),expected)
            for name,digest in bindings.items():
                raw=(ROOT/'data/evidence_v3'/name).read_bytes()
                self.assertEqual(hashlib.sha256(raw).hexdigest(),digest)

    def test_b4_current_panel_recorded_counts(self):
        doc=json_text((ROOT/'data/evidence_v3/b4_analysis.json').read_text())
        models={'Qwen3-8B-bf16','Qwen3-32B-bf16'}
        states={'base','L2','L3','L4'}
        self.assertEqual({c['model'] for c in doc['cells']},models)
        self.assertEqual({c['state'] for c in doc['cells']},states)
        for model in models:
            for state in states:
                cells=[c for c in doc['cells'] if c['model']==model and c['state']==state]
                self.assertEqual(len(cells),8)
                self.assertEqual(len({c['canary'] for c in cells}),8)
        for base,expected in [(True,(16,0)),(False,(17,31))]:
            cells=[c for c in doc['cells'] if (c['state']=='base')==base]
            self.assertEqual(tuple(sum(c['outcome']==label for c in cells)
                                   for label in ('SAFE','UNSAFE')),expected)
        for cell in doc['cells']:
            self.assertEqual(cell['evaluations'],4096)
            self.assertIs(type(cell['witness_count']),int)
            self.assertEqual(cell['witness_count']==0,cell['outcome']=='SAFE')
            self.assertNotIn('witnesses',cell)

    def test_unexpected_internal_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in SOURCE_FILES:
                target = root/name
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(ROOT/name,target)
            (root/'STATUS.md').write_text('not a public artifact')
            with self.assertRaisesRegex(ValueError,'unexpected package files'):
                check_package(root)

    def test_missing_and_tampered_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in SOURCE_FILES:
                if name=='data/evidence_v3/b1_analysis.json':
                    continue
                target = root/name
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(ROOT/name,target)
            with self.assertRaisesRegex(ValueError,'missing package files'):
                check_package(root)
            (root/'data/evidence_v3/b1_analysis.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'mismatch'):
                check_package(root)

    def test_ignored_environment_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binary = root/'.venv/bin'
            binary.mkdir(parents=True)
            (binary/'python').symlink_to('not-present')
            self.assertEqual(inventory(root),set())

    def test_payload_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'payload.json').symlink_to('not-present')
            with self.assertRaisesRegex(ValueError,'symlinks'):
                inventory(root)

    def test_operational_imports_rejected(self):
        for text in ['import torch','from transformers import AutoModel',
                     'import subprocess','import requests','import occlumency.b2_shard',
                     'from pysat.solvers import Solver']:
            with self.subTest(text=text),self.assertRaises(ValueError):
                check_imports(text,'analysis/replay.py')

    def test_dynamic_execution_rejected(self):
        for text in ["__import__('math')","eval('1+1')"]:
            with self.subTest(text=text),self.assertRaises(ValueError):
                check_imports(text,'analysis/replay.py')

    def test_duplicate_or_nonfinite_json_rejected(self):
        for text in ['{"a":1,"a":2}','{"a":NaN}','{"a":Infinity}']:
            with self.subTest(text=text),self.assertRaises(ValueError):
                json_text(text)


if __name__=='__main__':
    unittest.main()
