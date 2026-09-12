"""Pure numerical functions adapted from the frozen analyses.

No acquisition, candidate construction, model loading, or project-runner imports.
See PROVENANCE.json for source hashes and REPRODUCIBILITY.md for estimands.
The two means below are intentionally different: the frozen A3 calculation used
statistics.mean (exact summation), whereas B1/B3 used ordinary sum/len.
"""
import math
import random
from fractions import Fraction


def mean(values):
    if not values:
        raise ValueError('empty denominator')
    return sum(values) / len(values)


def exact_mean(values):
    values = list(values)
    if not values:
        raise ValueError('empty denominator')
    return float(sum((Fraction(v) for v in values), Fraction()) / len(values))


def median(values):
    xs = sorted(values)
    if not xs:
        raise ValueError('empty median')
    n = len(xs)
    return float(xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2)


def r1(pairs):
    """Sufficient median certification: censoring strictly above upper middle."""
    if not pairs or any(v is None for v, _ in pairs):
        return False
    upper = sorted(v for v, _ in pairs)[len(pairs) // 2]
    return all(not censored or value > upper for value, censored in pairs)


def percentile(sample, indices):
    ordered = sorted(sample)
    if not ordered or not 0 <= indices[0] <= indices[1] < len(ordered):
        raise ValueError('invalid percentile indices')
    return [ordered[indices[0]], ordered[indices[1]]]


def wilson(k, n, z=1.96):
    if type(k) is not int or type(n) is not int or not 0 <= k <= n or n == 0:
        raise ValueError('invalid binomial counts')
    p, z2 = k / n, z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    return [center - half, center + half]


def group(records, keys):
    groups = {}
    for rec in records:
        groups.setdefault(tuple(rec[k] for k in keys), []).append(rec)
    return groups


def a1a2_cells(a1, a2, rules, bootstrap=False):
    left, right = group(a1, ['family','size']), group(a2, ['family','size'])
    if set(left) != set(right):
        raise ValueError('A1/A2 group mismatch')
    rng = random.Random(rules['seed'])  # ONE stream over sorted sampled cells.
    cells = []
    for family, size in sorted(left):
        one, two = left[(family,size)], right[(family,size)]
        deterministic = family == 'pigeonhole'
        pairs = [(r['conflicts'],r['censored']) for r in one]
        certified = r1(pairs)
        first = {'n':len(one),'n_censored':sum(c for _,c in pairs),
                 'median':median([v for v,_ in pairs]) if certified else None,
                 'certified_under_r1':certified}
        if bootstrap and certified and not deterministic:
            samples, bad = [], 0
            for _ in range(rules['B']):
                draw = [pairs[rng.randrange(len(pairs))] for _ in pairs]
                bad += not r1(draw)
                samples.append(median([v for v,_ in draw]))
            first.update(ci=percentile(samples,rules['ci_indices']) if not bad else None,
                         interval_admissible=bad == 0,
                         bootstrap_replicates_uncertified_under_r1=bad,
                         bootstrap_fraction_uncertified_under_r1=round(bad/rules['B'],4))
        elif deterministic:
            first.update(ci=None,interval_admissible=None,
                         bootstrap_replicates_uncertified_under_r1=None,
                         bootstrap_fraction_uncertified_under_r1=None)
        decided = sum(r['solved'] for r in two)
        refs = [r['conflicts'] for r in one if r['status']=='UNSAT']
        cens = [r['conflicts'] for r in two if r['censored']]
        cells.append({'family':family,'size':size,'n_vars':one[0]['n_vars'],
                      'a1':first,'a2':{'n':len(two),'decided':decided,
                          'fraction':decided/len(two),
                          'wilson':None if deterministic else wilson(decided,len(two),rules['z'])},
                      's1':{'n_refuted':len(refs),'median':median(refs) if refs else None},
                      's2':{'n_censored':len(cens),'n_censored_with_conflicts':len(cens),
                            'min_censored_conflicts':min(cens) if cens else None}})
    return cells


def a3_contrast(pool, windows, seed, replicates=10000, bootstrap=False):
    transition, low, high = (windows[k] for k in ['transition','shoulder_low','shoulder_high'])
    order = transition + low + high
    vals = {a:[(math.log2(1+r['conflicts']),r['censored']) for r in pool[a]] for a in order}
    if any(not r1(v) for v in vals.values()):
        raise ValueError('A3 ratio fails R1; cannot omit a ratio')
    medians = {a:median([x for x,_ in vals[a]]) for a in order}

    def deltas(meds, ts=transition):
        t = exact_mean(meds[a] for a in ts)
        lo = exact_mean(meds[a] for a in low)
        hi = exact_mean(meds[a] for a in high)
        return t-lo,t-hi,t-0.5*(lo+hi)

    rise, fall, symmetric = deltas(medians)
    out = {'delta_rise':rise,'delta_fall':fall,'delta':symmetric,
           'window_medians':{k:exact_mean(medians[a] for a in windows[k]) for k in windows},
           'ratios':{str(a):{'alpha':a,'median_log2':medians[a],'n':len(vals[a]),
                           'n_censored':sum(c for _,c in vals[a]),'certified_under_r1':True} for a in order}}
    if bootstrap:
        rng = random.Random(seed)  # Reset independently for each n, same seed.
        samples = [[],[],[]]
        bad = 0
        for _ in range(replicates):
            meds, ok = {}, True
            for a in order:
                pairs = vals[a]
                draw = [pairs[rng.randrange(len(pairs))] for _ in pairs]
                ok = ok and r1(draw)
                meds[a] = median([v for v,_ in draw])
            bad += not ok
            for target, value in zip(samples,deltas(meds)):
                target.append(value)
        indices = [int(.025*(replicates-1)),int(.975*(replicates-1))]
        for name, sample in zip(['ci95_rise','ci95_fall','ci95'],samples):
            out[name] = percentile(sample,indices) if not bad else None
        out['bootstrap_replicates_uncertified_under_r1'] = bad
        out['bootstrap_fraction_uncertified_under_r1'] = round(bad/replicates,4)
        out['interval_admissible'] = bad == 0
        if not bad:
            for field, interval in [('width_rise','ci95_rise'),('width_fall','ci95_fall'),('width','ci95')]:
                out[field] = out[interval][1]-out[interval][0]
            out['rise_above_zero'] = out['ci95_rise'][0]>0
            out['fall_above_zero'] = out['ci95_fall'][0]>0
            out['excludes_zero'] = out['ci95'][0]>0 or out['ci95'][1]<0
            out['meets_precision_target'] = out['width_fall']<=.5
            out['peak_claimed'] = out['rise_above_zero'] and out['fall_above_zero'] and out['meets_precision_target']
    all_medians = {a:median([math.log2(1+r['conflicts']) for r in rows]) for a,rows in pool.items()}
    sensitivity = {'prespecified':symmetric,'plus_5.0':deltas(all_medians,transition+[5.0])[2]}
    for a in transition:
        sensitivity['drop_'+str(a)] = deltas(all_medians,[x for x in transition if x!=a])[2]
    return out,sensitivity


def admitted(run):
    cells = [c for c in run['cells'] if c['admitted']]
    if not cells or len(cells)!=run['admitted_cells']:
        raise ValueError('admitted denominator mismatch')
    return cells


def b1_run_rate(run, q):
    cells = admitted(run)
    return 1.0-sum(c['y_by_q'][str(q)] for c in cells)/len(cells)


def b1_curve(runs, budgets):
    return [{'q':q,'fnr':mean([b1_run_rate(r,q) for r in runs])} for q in budgets]


def b1_bootstrap(runs, budgets, seed=20260903, replicates=2000):
    rng = random.Random(seed)
    diffs = []
    for _ in range(replicates):
        drawn = [runs[rng.randrange(len(runs))] for _ in runs]
        # Preserve two separate sums from the frozen implementation.
        diffs.append(mean([b1_run_rate(r,budgets[-1]) for r in drawn])-
                     mean([b1_run_rate(r,budgets[0]) for r in drawn]))
    return percentile(diffs,[int(.025*(replicates-1)),int(.975*(replicates-1))])


def b3_units(runs, budgets):
    cells, tracks, units = {}, [], {}
    for r in runs:
        if r['track'] not in tracks:
            tracks.append(r['track'])
        key = r['public_run_id']
        cells.setdefault((r['track'],r['refusal_level']),[]).append(key)
        units[key] = {q:[1.0 if c['y_by_q'][str(q)] else 0.0 for c in admitted(r)] for q in budgets}
    return cells,tracks,units


def b3_aggregate(cells, tracks, units, q):
    return mean([mean([mean([mean(units[r][q]) for r in cells[(t,l)]])
                        for l in sorted(l for tt,l in cells if tt==t)]) for t in tracks])


def b3_bootstrap(cells, tracks, units, budgets, seed=20260910, replicates=2000):
    rng = random.Random(seed)
    samples = {q:[] for q in budgets}
    for _ in range(replicates):
        drawn_tracks = [tracks[rng.randrange(len(tracks))] for _ in tracks]
        per_track = {q:[] for q in budgets}
        for track in drawn_tracks:
            levels = sorted(l for t,l in cells if t==track)
            per_level = {q:[] for q in budgets}
            for level in levels:
                runs = cells[(track,level)]
                drawn_runs = [runs[rng.randrange(len(runs))] for _ in runs]
                per_seed = {q:[] for q in budgets}
                for run in drawn_runs:
                    vector = units[run]
                    n = len(vector[budgets[0]])
                    idx = [rng.randrange(n) for _ in range(n)]
                    for q in budgets:
                        per_seed[q].append(mean([vector[q][i] for i in idx]))
                for q in budgets:
                    per_level[q].append(mean(per_seed[q]))
            for q in budgets:
                per_track[q].append(mean(per_level[q]))
        for q in budgets:
            samples[q].append(mean(per_track[q]))
    indices = [int(.025*(replicates-1)),int(.975*(replicates-1))]
    return {q:percentile(samples[q],indices) for q in budgets}


def rational(value):
    return {'num':value.numerator,'den':value.denominator,'float':float(value)}


def miss_probability(n, w, budget):
    if any(type(v) is not int for v in (n,w,budget)) or not 0<=w<=n or not 0<=budget<=n:
        raise ValueError('invalid recorded-evaluation counts')
    return Fraction(math.comb(n-w,budget),math.comb(n,budget)) if budget<=n-w else Fraction()


def coverage_summary(cells, budgets):
    values = {q:[miss_probability(c['evaluations'],c['witness_count'],q) for c in cells] for q in budgets}
    if not cells:
        return []
    return [{'B':q,'mean':rational(sum(values[q],Fraction())/len(cells)),
             'min':rational(min(values[q])),'max':rational(max(values[q]))} for q in budgets]
