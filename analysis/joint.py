#!/usr/bin/env python3
"""Retabulate frozen B1/B4 record labels, never execute experiments.

The public projection reports source identity eligibility. This module checks
public tuple/membership/label consistency, not the omitted private identity
evidence. No generation, adjudication, confidence interval or association test
is performed. Table 1 and Appendix Table 16 use the same 30 published pairs.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from analysis.check_package import input_snapshot, require

TRACKS = ('Qwen3-8B-bf16', 'Qwen3-32B-bf16')
LEVELS = (2, 3, 4)
IDENTITY = ('track', 'sft_seed', 'refusal_level', 'canary_id')
LINKS = ('b1_public_run_id', 'b4_public_cell_id')
ROSTER_FIELDS = set(IDENTITY + LINKS + (
    'b1_selected', 'b1_admitted', 'b4_present', 'identity_status'))
PAIR_FIELDS = set(IDENTITY + LINKS + ('b1_by_16', 'b1_by_256', 'b4_label'))
COLS = ('recovered_by_16', 'first_recovered_17_to_256', 'not_recovered_by_256')
ROWS = ('SAFE', 'UNSAFE')
COUNTS = {'b1_only': 15, 'b1_original_exclusion': 3,
          'b4_only': 18, 'identity_eligible': 30}
SCOPES = {
    'schema_version': 'public_b1b4_joint_projection_v1',
    'source_schema_version': 'b1b4_joint_v1',
    'analysis_kind': 'post_hoc_descriptive',
    'record_labels_only': True,
    'execution_scope': 'public_record_label_retabulation_only',
    'source_identity_eligibility_reported': True,
    'source_identity_eligibility_independently_reverified': False,
    'original_generation_replayed': False,
    'leakage_readjudicated': False,
    'same_executable_established': False,
}


def _equal(actual, expected, label):
    # JSON spelling distinguishes booleans from integers, unlike Python ==.
    require(json.dumps(actual, sort_keys=True, allow_nan=False) ==
            json.dumps(expected, sort_keys=True, allow_nan=False),
            'joint frozen mismatch: ' + label)


def _identity(row):
    require(row['track'] in TRACKS, 'joint track outside fixed scope')
    require(type(row['sft_seed']) is int and row['sft_seed'] == 0,
            'joint seed outside fixed scope')
    require(type(row['refusal_level']) is int and row['refusal_level'] in LEVELS,
            'joint refusal level outside fixed scope')
    require(isinstance(row['canary_id'], str) and
            re.fullmatch(r'canary(?:0|[1-9][0-9]*)', row['canary_id']) is not None,
            'invalid joint canary identity')
    return tuple(row[k] for k in IDENTITY)


def _public_records(b1, b4):
    """Index public records; B4 exposes state, but does not expose SFT seed."""
    require(b1['budget_grid'] == [16, 32, 64, 128, 256], 'B1 prefix grid mismatch')
    first, fourth = {}, {}
    run_keys, run_ids, cell_ids = set(), set(), set()
    for run in b1['runs']:
        key = (run['track'], run['sft_seed'], run['refusal_level'])
        require(key not in run_keys and run['public_run_id'] not in run_ids,
                'duplicate B1 public run')
        run_keys.add(key)
        run_ids.add(run['public_run_id'])
        require(run['public_run_id'] == 'B1-%02d' % run['index'], 'B1 public ID mismatch')
        for cell in run['cells']:
            full_key = key + (cell['canary_id'],)
            require(full_key not in first, 'duplicate B1 public canary')
            first[full_key] = (run['public_run_id'], cell)
    for cell in b4['cells']:
        key = (cell['model'], cell['state'], cell['canary'])
        require(key not in fourth and cell['public_cell_id'] not in cell_ids,
                'duplicate B4 public cell')
        require(cell['public_cell_id'] == 'B4-%02d' % cell['index'], 'B4 public ID mismatch')
        require(cell['outcome'] in {'SAFE', 'UNSAFE', 'UNKNOWN'}, 'invalid public B4 label')
        cell_ids.add(cell['public_cell_id'])
        fourth[key] = cell
    return first, fourth


def _column(pair):
    return (COLS[0] if pair['b1_by_16'] else
            COLS[1] if pair['b1_by_256'] else COLS[2])


def retabulate(pairs):
    """Count already-validated labels only; no inference or resampling."""
    table = {row: {col: 0 for col in COLS} for row in ROWS}
    for pair in pairs:
        table[pair['b4_label']][_column(pair)] += 1
    two = {row: {'not_recovered_by_256': table[row][COLS[2]],
                 'recovered_by_256': table[row][COLS[0]] + table[row][COLS[1]]}
           for row in ROWS}
    totals = {row: sum(table[row].values()) for row in ROWS}
    x, y = table['UNSAFE'][COLS[2]], totals['UNSAFE']
    return {
        'n_pairs': len(pairs),
        'table_2x3': table,
        'table_2x2_at_256': two,
        'b4_row_totals': totals,
        'b1_column_totals_2x3': {col: sum(table[row][col] for row in ROWS) for col in COLS},
        'b1_column_totals_at_256': {
            col: sum(two[row][col] for row in ROWS)
            for col in ('not_recovered_by_256', 'recovered_by_256')},
        'unsafe_without_b1_recovery': {
            'x': x, 'y': y, 'ratio': str(x) + '/' + str(y) if y else None,
            'ratio_status': 'defined' if y else 'undefined'},
    }


def validate_joint(doc, b1, b4):
    """Validate reported public consistency, not private identity eligibility."""
    require(set(doc) == set(SCOPES) | {'identity_scope', 'public_linkage_scope',
            'identity_counts', 'summary', 'groups', 'identity_roster', 'validated_pairs'},
            'unexpected/missing joint document fields')
    for field, value in SCOPES.items():
        _equal(doc[field], value, field)
    for field in ('identity_scope', 'public_linkage_scope'):
        require(isinstance(doc[field], str) and bool(doc[field]), 'missing joint scope statement')
    first, fourth = _public_records(b1, b4)
    scope_b1 = {key for key in first if key[0] in TRACKS and key[1] == 0 and key[2] in LEVELS}
    scope_b4 = {(track, 0, level, canary) for track, state, canary in fourth
                for level in LEVELS if track in TRACKS and state == 'L' + str(level)}
    roster = {}
    require(len(doc['identity_roster']) == 66, 'wrong joint roster count')
    for row in doc['identity_roster']:
        require(set(row) == ROSTER_FIELDS, 'unexpected/missing joint roster fields')
        key = _identity(row)
        require(key not in roster, 'duplicate joint roster identity')
        roster[key] = row
        for field in ('b1_selected', 'b1_admitted', 'b4_present'):
            require(type(row[field]) is bool, 'invalid joint admission/presence flag')
        selected, admitted, present = row['b1_selected'], row['b1_admitted'], row['b4_present']
        require(selected == (key in scope_b1), 'B1 selected membership mismatch')
        require(present == (key in scope_b4), 'B4 present membership mismatch')
        if selected:
            public_id, cell = first[key]
            require(row['b1_public_run_id'] == public_id, 'joint B1 public link mismatch')
            require(type(cell['admitted']) is bool and admitted == cell['admitted'],
                    'B1 admission flag mismatch')
            require(not admitted or cell['supported'] is True, 'unsupported admitted B1 endpoint')
            ys = cell['y_by_q']
            require(set(ys) == {str(q) for q in b1['budget_grid']}, 'missing/extra B1 prefix')
            values = [ys[str(q)] for q in b1['budget_grid']]
            require(all(type(value) is bool for value in values) and values == sorted(values),
                    'invalid/nonmonotone public B1 prefix')
        else:
            require(row['b1_public_run_id'] is None and not admitted,
                    'unselected B1 record cannot be admitted or linked')
        if present:
            cell = fourth[(key[0], 'L' + str(key[2]), key[3])]
            require(row['b4_public_cell_id'] == cell['public_cell_id'], 'joint B4 public link mismatch')
        else:
            require(row['b4_public_cell_id'] is None, 'absent B4 record cannot be linked')
        reported_status = ('b1_original_exclusion' if selected and not admitted else
                           'identity_eligible' if admitted and present else
                           'b1_only' if admitted else 'b4_only')
        require(row['identity_status'] == reported_status, 'joint reported identity status inconsistent')
    require(set(roster) == scope_b1 | scope_b4, 'missing/extra joint roster identity')
    counts = dict(Counter(row['identity_status'] for row in roster.values()))
    _equal(counts, COUNTS, 'fixed identity counts')
    _equal(counts, doc['identity_counts'], 'reported identity counts')

    pairs = {}
    for pair in doc['validated_pairs']:
        require(set(pair) == PAIR_FIELDS, 'unexpected/missing joint pair fields')
        key = _identity(pair)
        require(key not in pairs, 'duplicate joint pair identity')
        pairs[key] = pair
        require(key in roster and roster[key]['identity_status'] == 'identity_eligible',
                'joint pair lacks reported eligible roster identity')
        for link in LINKS:
            require(pair[link] == roster[key][link], 'joint pair public link mismatch')
        require(type(pair['b1_by_16']) is bool and type(pair['b1_by_256']) is bool,
                'invalid joint B1 prefix label')
        require(not pair['b1_by_16'] or pair['b1_by_256'], 'nonmonotone joint B1 prefix')
        cell = first[key][1]
        require(pair['b1_by_16'] == cell['y_by_q']['16'] and
                pair['b1_by_256'] == cell['y_by_q']['256'], 'joint B1 endpoint label mismatch')
        require(pair['b4_label'] in ROWS, 'invalid joint B4 label')
        require(pair['b4_label'] == fourth[(key[0], 'L' + str(key[2]), key[3])]['outcome'],
                'joint B4 recorded label mismatch')
    eligible = {key for key, row in roster.items() if row['identity_status'] == 'identity_eligible'}
    require(set(pairs) == eligible and len(pairs) == 30, 'missing/extra joint pair identity')
    summary = retabulate(list(pairs.values()))
    _equal(summary, doc['summary'], 'summary')
    groups = []
    for track in TRACKS:
        for level in LEVELS:
            group = [p for p in pairs.values() if p['track'] == track and p['refusal_level'] == level]
            groups.append(dict(retabulate(group), track=track, sft_seed=0, refusal_level=level))
    _equal(groups, doc['groups'], 'six groups')
    return dict(SCOPES, identity_scope=doc['identity_scope'],
                public_linkage_scope=doc['public_linkage_scope'],
                identity_counts=counts, summary=summary, groups=groups,
                validated_pairs=doc['validated_pairs'], identity_roster=doc['identity_roster'])


def canary_rows(result):
    """Appendix Table 16 bins with complete canary IDs, from Table 1's pairs."""
    rows = []
    for group in result['groups']:
        row = {key: group[key] for key in IDENTITY[:3]}
        row.update(n_pairs=group['n_pairs'], safe_not_recovered_by_256=[],
                   unsafe_recovered_by_16=[], unsafe_first_recovered_17_to_256=[],
                   unsafe_not_recovered_by_256=[])
        pairs = [p for p in result['validated_pairs']
                 if all(p[key] == group[key] for key in IDENTITY[:3])]
        for pair in sorted(pairs, key=lambda p: int(p['canary_id'][6:])):
            if pair['b4_label'] == 'SAFE':
                require(not pair['b1_by_256'], 'published SAFE row has B1 recovery')
                field = 'safe_not_recovered_by_256'
            else:
                field = 'unsafe_' + _column(pair)
            row[field].append(pair['canary_id'])
        rows.append(row)
    return rows


def replay_joint(root=ROOT):
    root = Path(root).resolve()
    result = validate_joint(
        input_snapshot(root, 'b1b4_joint_analysis.json'),
        input_snapshot(root, 'b1_analysis.json'),
        input_snapshot(root, 'b4_analysis.json'))
    result['canary_id_rows'] = canary_rows(result)
    return result


def check_joint(root=ROOT, bootstrap=False):
    replay_joint(root)
    return ('66 reported roster identities/30 pairs; frozen 2x2, 2x3 and six groups matched; '
            'record-label retabulation only; source eligibility not independently reverified; '
            'no generation, adjudication, confidence intervals or association tests')


def table1_markdown(result):
    summary = result['summary']
    lines = ['Table 1. Post-hoc record-label counts (30 reported eligible pairs).', '',
             '| Recorded B4 | B1 by 16 | First B1 at 17-256 | No B1 by 256 | Total |',
             '| --- | ---: | ---: | ---: | ---: |']
    for label in ROWS:
        values = [summary['table_2x3'][label][col] for col in COLS]
        lines.append('| ' + label + ' | ' + ' | '.join(map(str, values)) +
                     ' | ' + str(summary['b4_row_totals'][label]) + ' |')
    values = [summary['b1_column_totals_2x3'][col] for col in COLS]
    lines.append('| Total | ' + ' | '.join(map(str, values)) + ' | ' + str(summary['n_pairs']) + ' |')
    return '\n'.join(lines)


def format_markdown(result):
    lines = [table1_markdown(result), '',
             'Appendix Table 16. Complete canary IDs for the same 30 pairs.', '',
             '| Track | Seed | Level | SAFE, no B1 by 256 | UNSAFE, B1 by 16 | '
             'UNSAFE, first B1 at 17-256 | UNSAFE, no B1 by 256 |',
             '| --- | ---: | ---: | --- | --- | --- | --- |']
    for row in result['canary_id_rows']:
        columns = [row['track'], str(row['sft_seed']), str(row['refusal_level'])]
        columns.extend(', '.join(row[field]) or '-' for field in (
            'safe_not_recovered_by_256', 'unsafe_recovered_by_16',
            'unsafe_first_recovered_17_to_256', 'unsafe_not_recovered_by_256'))
        lines.append('| ' + ' | '.join(columns) + ' |')
    lines.extend(['', 'Scope: post-hoc record-label retabulation only. Identity eligibility '
                  'is source-reported, not independently reverified. No original generation, '
                  'leakage re-adjudication, same-executable proof, confidence intervals or '
                  'association tests. SAFE is the recorded B4 label, not a new safety finding.'])
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--format', choices=('json', 'markdown'), default='markdown')
    args = parser.parse_args()
    result = replay_joint(args.root)
    print(json.dumps(result, indent=2) if args.format == 'json' else format_markdown(result))


if __name__ == '__main__':
    main()
