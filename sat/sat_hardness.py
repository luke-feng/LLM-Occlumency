"""Classical SAT instance families and an optional lazy CPU solver wrapper.

The source algorithms are preserved for inspection and standalone classical SAT
use. This is not the original manifest-governed A1/A2/A3 acquisition pipeline.
Measured conflict counts describe one solver build, heuristic and finite grid;
they do not establish an asymptotic separation or LLM certification. Pigeonhole
resolution bounds are in the number of holes, not the number of variables.
Tseitin lower-bound hypotheses are not certified for the sampled graphs; the
reported spectral quantity is a numerical diagnostic, not an expansion proof.

CNF is DIMACS-style: a clause is a list of nonzero ints (+v true literal, -v
negated), variables 1..n. Generators are pure Python (unit-tested). `solve` lazily
imports `pysat` (`pip install python-sat`); without it, only the generators run.
"""
import random


# ---- instance families (pure) -----------------------------------------------

def random_3sat_cnf(n, m, rng):
    """m random 3-clauses over n vars (average-case; hard near m/n ~ 4.26)."""
    cnf = []
    for _ in range(m):
        vs = rng.sample(range(1, n + 1), 3)
        cnf.append([v if rng.random() < 0.5 else -v for v in vs])
    return cnf


def pigeonhole_cnf(holes):
    """PHP(holes+1, holes), UNSAT. Variable x(p,h) = pigeon p in hole h.
    Classical resolution bounds do not predict a solver's heuristic trajectory.
    """
    pigeons = holes + 1

    def x(p, h):
        return p * holes + h + 1

    cnf = []
    for p in range(pigeons):                     # each pigeon in some hole
        cnf.append([x(p, h) for h in range(holes)])
    for h in range(holes):                       # no two pigeons share a hole
        for p in range(pigeons):
            for q in range(p + 1, pigeons):
                cnf.append([-x(p, h), -x(q, h)])
    return cnf


def _xor_to_cnf(lits, rhs):
    """CNF for XOR(lits) == rhs (rhs in {0,1}). 2**(k-1) clauses for k literals."""
    k = len(lits)
    cnf = []
    for mask in range(1 << k):
        ones = bin(mask).count("1")
        # add a clause forbidding each *violating* assignment (parity != rhs); skip
        # the satisfying ones (parity == rhs)
        if (ones % 2) == rhs:
            continue
        clause = []
        for i, v in enumerate(lits):
            clause.append(-v if (mask >> i) & 1 else v)
        cnf.append(clause)
    return cnf


def spectral_gap(n_vertices, edges, degree):
    """Second-largest adjacency eigenvalue in absolute value, and the resulting
    spectral gap ``degree - lambda2``. Resolution lower bounds for Tseitin need an
    expander; this measures the property instead of assuming it, so a run can
    report the gap it actually sampled."""
    import math
    A = [[0.0] * n_vertices for _ in range(n_vertices)]
    for a, b in edges:
        A[a][b] = A[b][a] = 1.0
    # power iteration on the deflated operator: subtract the top eigenpair (the
    # all-ones vector at eigenvalue ``degree`` for a connected regular graph).
    rng = random.Random(0)
    v = [rng.gauss(0, 1) for _ in range(n_vertices)]
    ones = 1.0 / math.sqrt(n_vertices)
    lam = 0.0
    for _ in range(500):
        c = sum(v) * ones * ones
        v = [x - c * 1.0 for x in v]                 # deflate the all-ones direction
        w = [sum(A[i][j] * v[j] for j in range(n_vertices)) for i in range(n_vertices)]
        nrm = math.sqrt(sum(x * x for x in w))
        if nrm < 1e-12:
            return 0.0, float(degree)
        v = [x / nrm for x in w]
        lam = nrm
    return round(lam, 4), round(degree - lam, 4)


def tseitin_parity_cnf(n_vertices, degree=3, seed=0):
    """Tseitin parity over a random `degree`-regular graph with an odd total charge
    -> UNSAT. Resolution lower bounds hold on expanders, so the sampled graph's
    spectral gap is returned alongside and should be reported, not assumed."""
    rng = random.Random(seed)
    # build a random regular graph via the configuration model (retry until simple)
    for _ in range(200):
        stubs = [v for v in range(n_vertices) for _ in range(degree)]
        rng.shuffle(stubs)
        edges, ok = set(), True
        for i in range(0, len(stubs), 2):
            a, b = stubs[i], stubs[i + 1]
            if a == b or (min(a, b), max(a, b)) in edges:
                ok = False
                break
            edges.add((min(a, b), max(a, b)))
        if ok:
            break
    else:
        raise RuntimeError("could not build a simple regular graph")
    edges = sorted(edges)
    evar = {e: i + 1 for i, e in enumerate(edges)}
    inc = {v: [] for v in range(n_vertices)}
    for e in edges:
        inc[e[0]].append(evar[e]); inc[e[1]].append(evar[e])
    charges = [0] * n_vertices
    charges[0] = 1                                # total charge odd -> UNSAT
    cnf = []
    for v in range(n_vertices):
        if inc[v]:
            cnf += _xor_to_cnf(inc[v], charges[v])
    lam2, gap = spectral_gap(n_vertices, edges, degree)
    # "edges" is an ADDITIVE meta key: the (cnf, n_edges, meta) tuple contract is
    # unchanged for every existing caller. It exists so an instance manifest can
    # bind the sampled graph's identity for audit, without which two "identical"
    # Tseitin instances are identical only by assumption.
    return cnf, len(edges), {"lambda2": lam2, "spectral_gap": gap,
                             "edges": [list(e) for e in edges]}


# ---- CDCL solve (lazy pysat) ------------------------------------------------

def solve(cnf, solver_name="g3", conf_budget=0):
    """Solve with a CDCL solver (Glucose 'g3' by default). Returns
    {sat, solved, conflicts, decisions, time}. ``conf_budget`` > 0 caps conflicts;
    An instance the solver abandons within the budget has status ``UNKNOWN``. Its
    truth value is not false, and its conflict count is a censoring time rather
    than a completed proof cost, so both are kept distinguishable here and the
    callers must not pool them with completed runs."""
    from pysat.solvers import Solver
    s = Solver(name=solver_name, bootstrap_with=cnf, use_timer=True)
    if conf_budget and conf_budget > 0:
        s.conf_budget(conf_budget)
        res = s.solve_limited()
        status = "UNKNOWN" if res is None else ("SAT" if res else "UNSAT")
    else:
        status = "SAT" if s.solve() else "UNSAT"
    st = s.accum_stats()
    solved = status != "UNKNOWN"
    out = {"status": status,
           "sat": (status == "SAT") if solved else None,
           "solved": solved, "censored": not solved,
           "conflicts": st.get("conflicts"), "decisions": st.get("decisions"),
           "time": round(s.time(), 4)}
    s.delete()
    return out


def _make_cnf(family, sz, k, rng, seed):
    """Return ``(cnf, meta)``. ``meta`` carries whatever the family measures about
    the instance it generated, which for Tseitin is the sampled spectral gap."""
    if family == "random_3sat":
        return random_3sat_cnf(sz, round(4.26 * sz), rng), {}
    if family == "pigeonhole":
        return pigeonhole_cnf(sz), {"deterministic": True}
    if family == "tseitin":
        cnf, _, g = tseitin_parity_cnf(sz, seed=seed + k)
        return cnf, g
    raise ValueError(family)


def _solve_task(task):
    cnf, solver_name, conf_budget = task
    return solve(cnf, solver_name=solver_name, conf_budget=conf_budget)


def standard_median(values):
    """Standard median, averaging the middle pair for an even sample."""
    import statistics
    return float(statistics.median(values))


def generate_instances(family, sizes, instances_per_size, seed):
    """The generation half of a sweep: one dict per instance, no solving.

    Deterministic in (family, sizes, instances_per_size, seed): random_3sat
    draws from one sequential RNG seeded here; tseitin uses seed+k per
    instance; pigeonhole is one deterministic formula per size, so its
    replication count is forced to one. `instance_id` is
    "<family>/<size>/<instance>" and the list is ordered by it.
    """
    rng = random.Random(seed)
    out = []
    for sz in sizes:
        reps = 1 if family == "pigeonhole" else instances_per_size
        for k in range(reps):
            cnf, meta = _make_cnf(family, sz, k, rng, seed)
            n_vars = max((abs(l) for cl in cnf for l in cl), default=0)
            out.append({"instance_id": f"{family}/{sz}/{k}",
                        "family": family, "size": sz, "instance": k,
                        "n_vars": n_vars, "n_clauses": len(cnf),
                        "clauses": cnf, "meta": meta})
    return out


def sweep_instances(instances, conf_budget=200000, solver_name="g3", jobs=1):
    """The solve-and-aggregate half, operating on pre-built instances.

    This is the ONE place aggregation happens; `hardness_sweep` generates and
    delegates here, and a manifest-governed runner passes instances read from a
    frozen input artifact. Returns ``(rows, records)``.

    A censored instance (the solver abandoned it inside its conflict budget)
    contributes to `n_censored` and the decided rate and NEVER to a cost
    median. `p90_conflicts` and `max_conflicts` are deliberately the exact
    order statistics `sorted[min(n - 1, int(0.9 * n))]` and `sorted[-1]` over
    completed decisions — index selects, not interpolations. That convention is
    load-bearing: the legacy capped run's conflict budget was bounded from
    aggregates precisely because these indices are known.
    """
    tasks = [(inst["clauses"], solver_name, conf_budget) for inst in instances]
    if jobs and jobs > 1:
        import multiprocessing as mp
        with mp.Pool(jobs) as pool:
            results = pool.map(_solve_task, tasks)
    else:
        results = [_solve_task(t) for t in tasks]

    records = []
    for inst, r in zip(instances, results):
        rec = {"instance_id": inst["instance_id"], "family": inst["family"],
               "size": inst["size"], "instance": inst["instance"],
               "n_vars": inst["n_vars"], "n_clauses": inst["n_clauses"],
               "status": r["status"], "sat": r["sat"], "solved": r["solved"],
               "censored": r["censored"], "conflicts": r["conflicts"],
               "decisions": r.get("decisions"), "time": r["time"]}
        rec.update({k: v for k, v in inst["meta"].items() if k != "edges"})
        records.append(rec)

    rows = []
    seen = []
    for inst in instances:
        key = (inst["family"], inst["size"])
        if key not in seen:
            seen.append(key)
    for family, sz in seen:
        group = [(inst, r) for inst, r in zip(instances, results)
                 if inst["family"] == family and inst["size"] == sz]
        done = [r for _, r in group if r["solved"]]
        conflicts = sorted(r["conflicts"] for r in done
                           if r["conflicts"] is not None)
        refuted_conflicts = sorted(r["conflicts"] for r in done
                                   if r["status"] == "UNSAT"
                                   and r["conflicts"] is not None)
        censored = [r["conflicts"] for _, r in group if r["censored"]]
        n = len(group)
        solved = len(done)
        refuted = sum(1 for r in done if r["status"] == "UNSAT")
        times = [r["time"] for r in done]
        first = group[0][0]
        rows.append({
            "family": family, "size": sz, "instances": n,
            "n_vars": first["n_vars"], "n_clauses": first["n_clauses"],
            "deterministic_family": family == "pigeonhole",
            "instance_ids": [inst["instance_id"] for inst, _ in group],
            "n_decided": solved,
            "n_completed_with_conflicts": len(conflicts),
            "n_refuted": refuted,
            "n_refuted_with_conflicts": len(refuted_conflicts),
            "decided_rate": round(solved / n, 6),
            "solve_rate": round(solved / n, 6),
            "refute_rate": round(refuted / n, 6),
            "n_unsat_solved": refuted,
            "n_refuted_completed": len(refuted_conflicts),
            "n_censored": len(censored),
            "min_censoring_time": min(censored) if censored else None,
            "min_censoring_conflicts": min(censored) if censored else None,
            "median_conflicts_completed":
                standard_median(conflicts) if conflicts else None,
            "median_conflicts":
                standard_median(conflicts) if conflicts else None,
            "median_conflicts_refuted":
                standard_median(refuted_conflicts) if refuted_conflicts else None,
            "p90_conflicts":
                conflicts[min(len(conflicts) - 1, int(0.9 * len(conflicts)))]
                if conflicts else None,
            "max_conflicts": conflicts[-1] if conflicts else None,
            "mean_time": round(sum(times) / len(times), 4) if times else None,
        })
    return rows, records


def hardness_sweep(family, sizes, instances_per_size=20, conf_budget=200000, seed=0,
                   solver_name="g3", jobs=1):
    """Per-instance solver records, plus the size-level summary, for one family.

    Generate-then-delegate: `generate_instances` builds the instance set and
    `sweep_instances` solves and aggregates, so aggregation exists exactly once
    whether instances come from here or from a frozen manifest. Returns
    ``(rows, records)`` with the same row keys as before, plus explicit
    numerator/denominator counts and `instance_ids`.

    This utility's ``median_conflicts`` is the standard median cost of COMPLETED
    DECISIONS. It is not the paper's all-instance R1-certified latent median.
    ``median_conflicts_refuted`` restricts to completed refutations; the two are
    equal for the unsatisfiable families and differ for random 3-CNF.
    """
    instances = generate_instances(family, sizes, instances_per_size, seed)
    return sweep_instances(instances, conf_budget=conf_budget,
                           solver_name=solver_name, jobs=jobs)
