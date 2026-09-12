#!/usr/bin/env python3
"""Read-only inventory, integrity and static dependency checks.

PROVENANCE.json is the trust anchor, not a cryptographic signature. This checker
cannot detect a coordinated replacement of the checker, manifest and dataset.
Ignored local environments/caches and .git are NOT part of the public payload.
"""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOTS = ('a1_rows','a2_rows','a1a2_analysis','a3_sweep','a3_windows',
             'b1_analysis','b2_analysis','b3_analysis','b4_analysis','b4_coverage',
             'b1b2_decomposition','historical_controls','utility_d3')
SOURCE_FILES = set('''
.gitignore README.md REPRODUCIBILITY.md DATA_PROVENANCE.md requirements.txt
requirements-sat.txt PROVENANCE.json analysis/__init__.py analysis/replay.py
analysis/statistics.py analysis/check_package.py figures/plot_experiments_v3.py
figures/plot_main_evidence_v3.py figures/results_map.py sat/__init__.py
sat/sat_hardness.py tests/__init__.py tests/test_replay.py tests/test_package.py
tests/test_sat.py data/observations/a1.jsonl data/observations/a2.jsonl
data/observations/a3.jsonl
docs/MODEL_CONSTRUCTION.md docs/EXPERIMENT_DEFINITIONS.md
metadata/model_construction.json metadata/experiment_definitions.json
models/__init__.py models/loading.py models/lora_sft.py requirements-models.txt
tests/test_model_construction.py tests/test_model_reference.py
'''.split()) | {'data/evidence_v3/'+s+'.json' for s in SNAPSHOTS}
FIGURES = {'figures/'+s+'.pdf' for s in ('a1a2_censoring_v3','b1_nested_budget_v3',
           'b2_tracks_v3','main_evidence_v3','results_map')}
IGNORED_ROOTS = {'.git','.venv','.pytest_cache','.mpl-cache'}
STD = {'argparse','ast','collections','copy','fractions','hashlib','itertools',
       'json','math','multiprocessing','os','pathlib','random','re','statistics','sys',
       'tempfile','unittest','shutil','dataclasses'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON key: '+key)
        result[key] = value
    return result


def json_text(raw):
    def reject(value):
        raise ValueError('non-finite JSON constant: '+value)
    return json.loads(raw, object_pairs_hook=_pairs, parse_constant=reject)


def load(path):
    return json_text(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def input_snapshot(root, name):
    path = Path(root)/'data/evidence_v3'/name
    manifest = load(Path(root)/'PROVENANCE.json')
    entries = [e for e in manifest['files'] if e['path']==str(path.relative_to(root))]
    require(len(entries)==1,'snapshot must have one public hash binding')
    require(sha(path)==entries[0]['sha256'],'snapshot SHA-256 mismatch: '+name)
    return load(path)


def inventory(root):
    paths = set()
    root = Path(root)
    for directory,dirs,files in os.walk(root,followlinks=False):
        base = Path(directory)
        ignored = IGNORED_ROOTS if base==root else set()
        dirs[:] = [d for d in dirs if d not in ignored and d!='__pycache__']
        for name in dirs+files:
            if base==root and name in IGNORED_ROOTS:
                continue
            path = base/name
            rel = path.relative_to(root)
            require(not path.is_symlink(),'symlinks are forbidden in payload: '+str(rel))
            if path.is_file():
                paths.add(rel.as_posix())
    return paths


def check_imports(source, relative):
    tree = ast.parse(source, filename=relative)
    parents={child:node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):
            modules = [x.name for x in node.names]
        elif isinstance(node,ast.ImportFrom):
            require(node.level==0,'relative/dynamic dependency is outside the static allowlist')
            modules = [node.module or '']
        else:
            modules = []
        for module in modules:
            base = module.split('.')[0]
            permitted = base in STD or module in {'analysis','analysis.check_package','analysis.statistics','analysis.replay','sat','sat.sat_hardness','models','models.loading','models.lora_sft'}
            permitted |= base=='matplotlib' and relative.startswith('figures/')
            permitted |= module=='pysat.solvers' and relative=='sat/sat_hardness.py'
            if base in {'torch','transformers','peft'} and relative in {'models/loading.py','models/lora_sft.py'}:
                ancestor=parents.get(node)
                while ancestor is not None and not isinstance(ancestor,(ast.FunctionDef,ast.AsyncFunctionDef)):
                    ancestor=parents.get(ancestor)
                require(ancestor is not None,'model frameworks must be lazy function-local imports')
                permitted=True
            require(permitted,'dependency outside allowlist: '+module+' in '+relative)
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name):
            require(node.func.id not in {'eval','exec','__import__','compile'},'dynamic execution is forbidden')
    return True


def check_package(root=ROOT, require_figures=False):
    root = Path(root).resolve()
    actual = inventory(root)
    require(SOURCE_FILES<=actual,'missing package files: '+str(sorted(SOURCE_FILES-actual)))
    require(actual<=SOURCE_FILES|FIGURES,'unexpected package files: '+str(sorted(actual-SOURCE_FILES-FIGURES)))
    if require_figures:
        require(FIGURES<=actual,'missing generated figures')
    manifest = load(root/'PROVENANCE.json')
    entries = manifest['files']
    require(len(entries)==len({e['path'] for e in entries}),'duplicate manifest bindings')
    bound = {e['path'] for e in entries}
    require(bound==SOURCE_FILES-{'PROVENANCE.json'},'manifest source inventory mismatch')
    for e in entries+manifest['generated_figures']:
        require(e['path'] in SOURCE_FILES|FIGURES,'manifest path outside package')
        path = root/e['path']
        if e['path'] in FIGURES and not path.exists() and not require_figures:
            continue
        require(path.is_file() and path.stat().st_size==e['bytes'],'file size mismatch: '+e['path'])
        require(sha(path)==e['sha256'],'file SHA-256 mismatch: '+e['path'])
    require({e['path'] for e in manifest['generated_figures']}==FIGURES,'figure manifest inventory mismatch')
    for e in manifest['data_and_adapted_sources']:
        require(e['path'] in bound,'source binding not in package inventory')
        require(e['sha256']==sha(root/e['path']),'projection hash mismatch')
        for s in e['sources']:
            require(re.fullmatch('[0-9a-f]{64}',s['original_sha256']) is not None,'invalid original source digest')
            require(isinstance(s['omitted_fields'],list) and bool(s['transformation']),'incomplete transformation provenance')
    for relative in sorted(actual):
        path = root/relative
        if path.suffix=='.py':
            check_imports(path.read_text(),relative)
        if path.suffix in {'.py','.md','.json','.jsonl','.txt'}:
            content = path.read_text(encoding='utf-8')
            require(not re.search(r'/(?:Users|Volumes|home|root)/|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}',content),
                    'private path or email in '+relative)
        if path.suffix=='.json':
            load(path)
    return {'source_files':len(SOURCE_FILES),'figures_present':len(actual&FIGURES),
            'scope':'inventory, public hashes and static dependencies; not model/solver replay'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--require-figures',action='store_true')
    args = parser.parse_args()
    print(json.dumps(check_package(args.root,args.require_figures),indent=2))


if __name__=='__main__':
    main()
