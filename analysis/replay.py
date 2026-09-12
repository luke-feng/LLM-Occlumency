#!/usr/bin/env python3
"""Replay frozen mathematics from public observations; never run a model/solver.

Default: point estimates, exact arithmetic and summary consistency checks.
--bootstrap: additionally replay frozen A1/A2, A3, B1 and B3 intervals.
Stored B2 and historical D3 intervals remain summary-only in either mode.
"""
import argparse
from collections import Counter
from fractions import Fraction
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from analysis import statistics as s
from analysis.check_package import check_package, input_snapshot, json_text, require

TRACKS = {'Qwen3-8B-bf16','Qwen3-8B-4bit','Qwen3-14B-bf16','Qwen3-14B-4bit',
          'Qwen3-32B-bf16','Qwen3-30B-A3B-bf16','Qwen3-235B-A22B-4bit',
          'gemma-3-12b-it-bf16','glm-4-9b-chat-hf-bf16','glm-4-9b-chat-hf-4bit'}
CORE_TRACKS = TRACKS-{'Qwen3-235B-A22B-4bit'}


def equal(actual, expected, path='value'):
    """Compare computed fields, with exact integer/rational counts.

Binary64 probabilities/log statistics allow absolute difference 2e-15 only.
No stored estimate is rewritten. Dictionaries may be explicit computed subsets.
"""
    if isinstance(actual,dict) and set(actual)=={'num','den','float'}:
        require(isinstance(expected,dict) and set(expected)==set(actual),'invalid rational triple: '+path)
        require(type(expected['num']) is int and type(expected['den']) is int and expected['den']>0,'invalid rational integers: '+path)
        require(actual['num']==expected['num'] and actual['den']==expected['den'],'exact fraction mismatch: '+path)
        require(type(expected['float']) is float and expected['float']==float(Fraction(expected['num'],expected['den'])),
                'rational binary64 mismatch: '+path)
    elif isinstance(actual,dict):
        require(isinstance(expected,dict),'wrong object: '+path)
        for key,value in actual.items():
            require(key in expected,'missing field: '+path+'.'+key)
            equal(value,expected[key],path+'.'+key)
    elif isinstance(actual,list):
        require(isinstance(expected,list) and len(actual)==len(expected),'wrong array: '+path)
        for i,(a,b) in enumerate(zip(actual,expected)):
            equal(a,b,path+'['+str(i)+']')
    elif isinstance(actual,float):
        require(type(expected) in (int,float) and math.isfinite(actual) and math.isfinite(expected)
                and math.isclose(actual,expected,rel_tol=0,abs_tol=2e-15),
                'numeric mismatch: '+path+' computed='+str(actual)+' expected='+str(expected))
    else:
        require(type(actual) is type(expected) and actual==expected,'exact mismatch: '+path)


def observations(root,name):
    records = [json_text(line) for line in (root/'data/observations'/ (name+'.jsonl')).read_text().splitlines()]
    require(len(records)==(8000 if name=='a3' else 404),'wrong observation count')
    keys = ({'alpha','censored','conflicts','decisions','instance','n','sat','solved','status','time'} if name=='a3' else
            {'censored','conflicts','decisions','family','instance','instance_id','n_clauses','n_vars','sat','size','solved','status','time_s'})
    for r in records:
        require(set(r)==keys,'unexpected/missing observation fields')
        require(r['status'] in {'SAT','UNSAT','CENSORED'},'invalid SAT observation status')
        require(type(r['solved']) is bool and r['solved']==(r['status']!='CENSORED'),'invalid solved flag')
        require(type(r['censored']) is bool and r['censored']==(not r['solved']),'invalid censoring flag')
        require(r['sat'] is (True if r['status']=='SAT' else False if r['status']=='UNSAT' else None),'invalid truth label')
        require(type(r['conflicts']) is int and r['conflicts']>=0,'invalid/missing cost')
    ids = [(r['n'],r['alpha'],r['instance']) for r in records] if name=='a3' else [r['instance_id'] for r in records]
    require(len(ids)==len(set(ids)),'duplicate observation identity')
    return records


def check_a1a2(root,bootstrap):
    a1,a2 = observations(root,'a1'),observations(root,'a2')
    equal(dict(Counter(r['status'] for r in a1)),{'SAT':100,'UNSAT':302,'CENSORED':2},'A1 counts')
    equal(dict(Counter(r['status'] for r in a2)),{'SAT':100,'UNSAT':258,'CENSORED':46},'A2 counts')
    identities = ['instance_id','family','size','instance','n_vars','n_clauses']
    for one,two in zip(a1,a2):
        require(all(one[k]==two[k] for k in identities),'A1/A2 observation identity mismatch')
    common = [x['status']==y['status'] for x,y in zip(a1,a2) if x['solved'] and y['solved']]
    require(len(common)==358 and all(common),'common decisions mismatch')
    for name,records in [('a1',a1),('a2',a2)]:
        rows = input_snapshot(root,name+'_rows.json')
        groups = s.group(records,['family','size'])
        require(len(rows)==12,'wrong A-series row count')
        for row in rows:
            group = groups[(row['family'],row['size'])]
            done = [r for r in group if r['solved']]
            refs = [r for r in group if r['status']=='UNSAT']
            computed = {'instances':len(group),'n_censored':len(group)-len(done),
                        'n_decided':len(done),'n_refuted':len(refs),
                        'n_vars':group[0]['n_vars'],'n_clauses':group[0]['n_clauses'],
                        'instance_ids':[r['instance_id'] for r in group],
                        'median_conflicts_completed':s.median([r['conflicts'] for r in done]) if done else None,
                        'median_conflicts_refuted':s.median([r['conflicts'] for r in refs]) if refs else None}
            equal(computed,row,name+' row')
    expected = input_snapshot(root,'a1a2_analysis.json')
    equal(expected['sap']['plan_rules'],{'B':10000,'seed':20260818,'ci_indices':[249,9749],
          'z':1.96,'a1_conf_budget':300000000,'a2_conf_budget':1000000,
          'point_rule':'r1_strict_upper_middle','interval_rule':'r2_zero_tolerance'},'A1/A2 frozen rules')
    equal(s.a1a2_cells(a1,a2,expected['sap']['plan_rules'],bootstrap),expected['cells'],'A1/A2 cells')
    return '808 observations; 12 R1 medians, completion fractions and Wilson intervals; '+('80,000 bootstrap replicates matched' if bootstrap else 'bootstrap not requested')


def check_a3(root,bootstrap):
    records = observations(root,'a3')
    groups = s.group(records,['n','alpha'])
    require(len(groups)==16 and all(len(v)==500 for v in groups.values()),'A3 grid/denominator mismatch')
    sweep = input_snapshot(root,'a3_sweep.json')
    windows = input_snapshot(root,'a3_windows.json')
    for row in sweep['rows']:
        rs = groups[(row['n'],row['alpha'])]
        done = [r for r in rs if r['solved']]
        sat = [r for r in done if r['status']=='SAT']
        unsat = [r for r in done if r['status']=='UNSAT']
        equal({'instances':len(rs),'solved':len(done),'censored':len(rs)-len(done),
               'n_sat':len(sat),'n_unsat':len(unsat),'n_no_conflict_record':0,
               'frac_sat':round(len(sat)/len(done),4),'decided_rate':round(len(done)/len(rs),4),
               'censored_fraction':round((len(rs)-len(done))/len(rs),4),
               'median_conflicts_all':s.median([r['conflicts'] for r in rs]),
               'median_conflicts_completed':s.median([r['conflicts'] for r in done]),
               'median_conflicts_sat':s.median([r['conflicts'] for r in sat]) if sat else None,
               'median_conflicts_unsat':s.median([r['conflicts'] for r in unsat]) if unsat else None,
               'median_certified_under_r1':s.r1([(r['conflicts'],r['censored']) for r in rs])},row,'A3 sweep')
    for n in (100,200):
        expected = windows['contrasts'][str(n)]
        require(expected['bootstrap_seed']==20260813 and expected['replicates']==10000,'A3 frozen bootstrap rule mismatch')
        pool = {a:rs for (nn,a),rs in groups.items() if nn==n}
        contrast,sensitivity = s.a3_contrast(pool,expected['windows'],expected['bootstrap_seed'],expected['replicates'],bootstrap)
        equal(contrast,expected,'A3 contrast '+str(n))
        equal(sensitivity,windows['sensitivity'][str(n)],'A3 sensitivity '+str(n))
    return '8,000 observations; 16 ratios, two window contrasts and sensitivities; '+('20,000 stratified bootstrap replicates matched' if bootstrap else 'bootstrap not requested')


def validate_runs(runs,budgets,label):
    require(len(runs)==57,'wrong run count')
    require({r['track'] for r in runs}==CORE_TRACKS,'track/precision mismatch')
    require([r['index'] for r in runs]==list(range(57)),'run order/index mismatch')
    require(len({r['public_run_id'] for r in runs})==57,'duplicate public run identity')
    require(sum(r['admitted_cells'] for r in runs)==423,'wrong admitted population')
    require(len({(r['track'],r['refusal_level']) for r in runs})==27,'wrong fixed strata')
    for r in runs:
        require(r['public_run_id']==label+'-%02d'%r['index'],'public run label mismatch')
        require(type(r['sft_seed']) is int and r['sft_seed'] in {0,1,2},'invalid seed')
        require(r['refusal_level'] in {2,3,4},'invalid fixed level')
        require(len(r['cells'])==8 and r['admitted_cells'] in {7,8},'incorrect inner denominator')
        require(len({c['canary_id'] for c in r['cells']})==8,'duplicate canary ordinal')
        for c in r['cells']:
            require(type(c['admitted']) is bool,'invalid admission flag')
            require(set(c['y_by_q'])=={str(q) for q in budgets},'missing/extra prefix')
            ys = [c['y_by_q'][str(q)] for q in budgets]
            require(all(type(y) is bool for y in ys) and ys==sorted(ys),'invalid/nonmonotone binary indicators')
        for c in s.admitted(r):
            if label=='B1':
                require(c['supported'] is True,'unsupported admitted B1 cell')
            else:
                require(c['status'] in {'leaked','budget_exhausted','proposal_exhausted'},'unsupported admitted B3 cell')


def check_b1(root,bootstrap):
    doc = input_snapshot(root,'b1_analysis.json')
    budgets = doc['budget_grid']
    require(budgets==[16,32,64,128,256],'B1 frozen budget grid mismatch')
    runs = doc['runs']
    validate_runs(runs,budgets,'B1')
    curve = s.b1_curve(runs,budgets)
    equal(curve,doc['curve'],'B1 curve')
    equal(curve[-1]['fnr']-curve[0]['fnr'],doc['claim']['D'],'B1 endpoint contrast')
    if bootstrap:
        equal(s.b1_bootstrap(runs,budgets),doc['claim']['ci95'],'B1 paired bootstrap')
    dec = input_snapshot(root,'b1b2_decomposition.json')['b1']
    require(len(dec['tracks'])==9,'B1 decomposition track count')
    exact_primary = {q:Fraction() for q in budgets}
    for row in dec['tracks']:
        rs = [r for r in runs if r['track']==row['track']]
        equal(len(rs),row['n_runs'],'B1 track run count')
        equal(sum(r['admitted_cells'] for r in rs),row['admitted_cells'],'B1 track admission')
        weight = Fraction(len(rs),len(runs))
        equal(s.rational(weight),row['weight'],'B1 track weight')
        values = []
        for point in row['curve']:
            q = point['q']
            rate = sum((1-Fraction(sum(c['y_by_q'][str(q)] for c in s.admitted(r)),r['admitted_cells']) for r in rs),Fraction())/len(rs)
            equal(s.rational(rate),point['fnr'],'B1 exact track rate')
            exact_primary[q] += weight*rate
            values.append(rate)
        equal(s.rational(values[-1]-values[0]),row['endpoint_difference'],'B1 exact track contrast')
    for point in curve:
        equal(float(exact_primary[point['q']]),point['fnr'],'B1 weighted recovery')
    return '57 runs/423 admitted; five run-equal points and nine exact track decompositions; '+('2,000 paired bootstrap replicates matched' if bootstrap else 'bootstrap not requested')


def check_b2(root,bootstrap=False):
    doc = input_snapshot(root,'b2_analysis.json')
    pop = doc['population']
    require(set(pop['tracks'])==TRACKS and pop['n_runs']==59,'B2 track/precision/run mismatch')
    require(pop['units']==pop['valid_canaries_total']==439,'B2 admitted denominator mismatch')
    require(sum(len(r['canary_ids']) for r in pop['valid_canaries'])==439,'B2 membership mismatch')
    require(len(pop['valid_canaries'])==59 and len({r['public_run_id'] for r in pop['valid_canaries']})==59,'B2 run identities')
    require(sum(len(c['sft_seeds']) for c in pop['cells'])==59,'B2 seed strata count')
    rows = doc['secondary']['per_track']
    require({r['track'] for r in rows}==TRACKS and len(rows)==10,'B2 secondary track mismatch')
    for row in rows:
        cells = [c for c in doc['secondary']['per_cell'] if c['track']==row['track']]
        for field in ['clean','legacy','template_only_clean','template_only_legacy']:
            equal(s.mean([c[field] for c in cells]),row[field],'B2 per-track '+field)
    for field,key in [('clean','primary'),('legacy','comparator')]:
        equal(s.mean([r[field] for r in rows]),doc[key]['point'],'B2 overall '+key)
    equal(doc['comparator']['point']-doc['primary']['point'],doc['contrast']['point'],'B2 contrast')
    for field in ['clean','legacy']:
        equal(s.mean([r['template_only_'+field] for r in rows]),doc['template_only'][field],'B2 template-only')
    dec = input_snapshot(root,'b1b2_decomposition.json')['b2']
    require(set(dec['common_tracks'])==CORE_TRACKS,'B2 common-track denominator mismatch')
    for row in dec['level_means_clean']:
        cells = [c for c in doc['secondary']['per_cell'] if c['refusal_level']==row['refusal_level'] and c['track'] in CORE_TRACKS]
        require(len(cells)==row['n_tracks']==9,'B2 must not mix 10/9/10 tracks')
        value = sum((Fraction(c['clean']) for c in cells),Fraction())/9
        equal(s.rational(value),row['mean'],'B2 exact level mean')
    return '59 runs/439 admitted; aggregate identities and fixed-nine-track decomposition matched; all CIs SUMMARY-ONLY (not replayed)'


def check_b3(root,bootstrap):
    doc = input_snapshot(root,'b3_analysis.json')
    budgets = [16,32,64,128]
    runs = doc['per_run']
    validate_runs(runs,budgets,'B3')
    for scope,cells in [('executed',[c for r in runs for c in r['cells']]),('admitted',[c for r in runs for c in s.admitted(r)])]:
        counts = Counter(c['status'] for c in cells)
        equal({k:counts[k] for k in doc['counts'][scope]},doc['counts'][scope],'B3 status '+scope)
    cells,tracks,units = s.b3_units(runs,budgets)
    prefix = [{'q':q,'R':s.b3_aggregate(cells,tracks,units,q)} for q in budgets]
    equal(prefix,doc['prefix'],'B3 prefixes')
    equal(prefix[-1]['R'],doc['claim']['R'],'B3 primary')
    for row in doc['per_cell']:
        ids = cells[(row['track'],row['refusal_level'])]
        equal(ids,row['runs'],'B3 fixed-cell membership')
        equal({str(q):s.mean([s.mean(units[r][q]) for r in ids]) for q in budgets},row['R_by_q'],'B3 per-cell rates')
    for row in doc['per_track']:
        equal({str(q):s.b3_aggregate(cells,[row['track']],units,q) for q in budgets},row['R_by_q'],'B3 track rates')
    if bootstrap:
        intervals = s.b3_bootstrap(cells,tracks,units,budgets)
        for row in doc['prefix']:
            equal(intervals[row['q']],row['ci95'],'B3 prefix bootstrap')
        equal(intervals[128],doc['claim']['ci95'],'B3 primary bootstrap')
    return '57 runs/423 admitted; nine track-equal summaries, four paired prefixes; '+('2,000 hierarchical bootstrap replicates matched' if bootstrap else 'bootstrap not requested')


def validate_b4_cells(doc,cov):
    require(len(doc['cells'])==64 and len({r['index'] for r in doc['cells']})==64,'B4 cell identities')
    equal(dict(Counter(c['outcome'] for c in doc['cells'])),{'SAFE':33,'UNSAFE':31},'B4 counts')
    equal(doc['counts'],{'SAFE':33,'UNSAFE':31,'UNKNOWN':0},'B4 stored counts')
    for c in doc['cells']:
        require(c['outcome'] in {'SAFE','UNSAFE','UNKNOWN'},'invalid B4 recorded status')
        require(c['public_cell_id']=='B4-%02d'%c['index'],'B4 public cell ID')
        require(c['evaluations']==4096 and type(c['witness_count']) is int and 0<=c['witness_count']<=4096,'B4 evaluation counts')
        require((c['witness_count']==0)==(c['outcome']=='SAFE'),'B4 outcome/count consistency')
    unsafe = [c for c in doc['cells'] if c['outcome']=='UNSAFE']
    budgets = cov['budgets']
    require(budgets==[16,32,64,128,256] and cov['summary']['n_cells']==31,'B4 coverage denominator')
    require({c['public_cell_id'] for c in cov['unsafe_cells']}=={c['public_cell_id'] for c in unsafe} and len(cov['unsafe_cells'])==31,'B4 coverage must include only UNSAFE cells')
    require({c['public_cell_id'] for c in cov['safe_cells']}=={c['public_cell_id'] for c in doc['cells'] if c['outcome']=='SAFE'} and len(cov['safe_cells'])==33,'B4 SAFE coverage mismatch')
    require(cov['unknown_cells']==[],'unexpected B4 UNKNOWN coverage')
    return unsafe,budgets


def check_b4(root,bootstrap=False):
    doc = input_snapshot(root,'b4_analysis.json')
    cov = input_snapshot(root,'b4_coverage.json')
    unsafe,budgets = validate_b4_cells(doc,cov)
    equal(s.coverage_summary(unsafe,budgets),cov['summary']['per_budget'],'B4 global exact coverage')
    by_id = {c['public_cell_id']:c for c in doc['cells']}
    for c in cov['unsafe_cells']:
        source = by_id[c['public_cell_id']]
        equal(source['witness_count'],c['witnesses'],'B4 count binding')
        for row in c['miss']:
            equal(s.rational(s.miss_probability(4096,c['witnesses'],row['B'])),row['value'],'B4 exact cell probability')
    for row in cov['groups']:
        cs = [c for c in unsafe if (c['model'],c['state'])==(row['model'],row['state'])]
        equal(len(cs),row['n_cells'],'B4 group size')
        equal(s.coverage_summary(cs,budgets),row['per_budget'],'B4 exact group coverage')
    return '64 recorded statuses and witness counts; exact probabilities and summaries over 31 UNSAFE cells only; no query/label replay'


def check_historical(root,bootstrap=False):
    doc = input_snapshot(root,'historical_controls.json')
    numbers = doc['numbers']
    require(set(numbers)=={'e3_ceiling','e3_track_range','e4_ladder_drop','e4_post_l1_band'},'retired/missing historical entries')
    n = numbers['e3_ceiling']['n']
    equal(n['gold']+n['wrong'],n['flagged'],'E3 detector denominator')
    equal(n['gold']/n['flagged'],numbers['e3_ceiling']['raw'],'E3 precision')
    require(n['outputs']==26854 and n['flagged']==11400 and n['gold']==6426 and n['recall']==1.0,'E3 stored counts')
    equal(numbers['e4_ladder_drop']['value'],[32.6,65.3],'sequence drop display')
    equal(numbers['e4_post_l1_band']['value'],18.9,'sequence band display')
    d3 = input_snapshot(root,'utility_d3.json')['tracks']
    require(set(d3)==TRACKS,'D3 track/precision separation')
    require(sum(len(x['accuracy']) for x in d3.values())==48 and sum(len(x['paired_delta']) for x in d3.values())==38,'D3 summary counts')
    require(sum(r['n_observations'] for x in d3.values() for r in x['accuracy'])==19600,'D3 observation coverage')
    for track,blocks in d3.items():
        levels = [r['refusal_level'] for r in blocks['accuracy']]
        require(levels==([0,2,4] if '235B' in track else [0,1,2,3,4]),'D3 level coverage')
        for category,rows in blocks.items():
            for row in rows:
                require(len(row['ci95'])==2 and row['ci95'][0]<=row['mean']<=row['ci95'][1],'D3 invalid stored interval')
                require(row['n_sft_seeds'] in {1,3} and row['n_observations']==200*row['n_sft_seeds'],'D3 seed denominator')
    for example in doc['benign_examples']:
        for counts in example['counts'].values():
            equal(counts['refusals']/counts['n_benign'],counts['frr'],'reported benign arithmetic')
    return 'E3 detector count arithmetic; D3 48 accuracy/38 contrast summaries; sequence and named benign examples SUMMARY-ONLY. Original 32/32 is manuscript-reported, not independently counted.'


CHECKS = {'a1a2':check_a1a2,'a3':check_a3,'b1':check_b1,'b2':check_b2,
          'b3':check_b3,'b4':check_b4,'historical':check_historical}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--only',choices=list(CHECKS))
    parser.add_argument('--bootstrap',action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    check_package(root)
    for name,check in CHECKS.items():
        if args.only is None or args.only==name:
            print(name.upper()+': '+check(root,args.bootstrap),flush=True)
    print('PASS: adapted numerical replay. No model, solver, candidate generation or raw-output adjudication.',flush=True)


if __name__=='__main__':
    main()
