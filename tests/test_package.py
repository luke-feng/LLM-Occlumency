import shutil
from pathlib import Path
import tempfile
import unittest

from analysis.check_package import (SOURCE_FILES,check_imports,check_package,
                                    inventory,json_text)

ROOT = Path(__file__).resolve().parents[1]


class PackageTests(unittest.TestCase):
    def test_current_package(self):
        self.assertEqual(check_package(ROOT)['source_files'],46)

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
