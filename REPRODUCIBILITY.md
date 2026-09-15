# Reproduction scope and statistical conventions

This is adapted mathematical replay from public numerical projections. It does not import the original analysis packages, which depend on excluded operational runners. Accepted numerical values are never overwritten. The script prints checks to stdout and exits nonzero on mismatch.

The optional model-construction supplement is separate from this mathematical replay. See `docs/MODEL_CONSTRUCTION.md` and `docs/EXPERIMENT_DEFINITIONS.md`: all 98 historical state metadata records are retained, but missing actual training hyperparameters remain null. Generic local SFT/loading utilities do not recreate original weights, supply original training data, or execute the defined audit campaigns. That supplement preserved the preceding numerical and figure payload. The current figure refresh adds two main-text layouts and updates the result map without changing numerical datasets, statistical modules or model-construction metadata.

## A1/A2

The same 404 formula identities occur in each accepted observation file. A1 has SAT/UNSAT/censored counts 100/302/2; A2 has 100/258/46. The 358 common completed decisions agree. These recorded verdicts are not independently proved by this package; no original CNF manifest or solver logs are bundled.

For each family/size cell, R1 requires every recorded censoring lower bound to lie strictly above the upper-middle order statistic of all observed costs. This is a conservative sufficient condition, not a necessary condition for identification. The all-instance latent median is conditional on those bounds and the frozen monotone-prefix assumption. Completed-only refutation medians remain a different statistic. In the largest Tseitin cell these are 3,813,311.5 versus 3,658,621.5, respectively.

The eight sampled-family intervals use 10,000 paired value/censoring resamples each, one RNG at seed 20260818 continuing across sorted family/size cells. Pigeonhole cells are deterministic singletons and consume no draws. R2 requires zero uncertified replicates; none are dropped. Percentile indices are 249 and 9749. A2 Wilson intervals use z=1.96. Deterministic cells have no sampling bars. Numerical replay is not proof of asymptotic solver complexity or LLM certification.

## A3

The grid has two variable counts, eight ratios and 500 observations per cell. The fixed transition ratios are 4.0/4.26/4.5, with lower shoulder 3.0/3.5 and upper shoulder 5.5/6.0. Compute each ratio's median log2(1+conflicts), then equal means across the selected ratios. Rise and fall are separate directional contrasts; their symmetric average is descriptive. Window-sensitivity variants are not added sampling uncertainty.

Within-ratio bootstrap resampling preserves each ratio's 500 observations. For each variable count, the RNG is reset independently to the SAME seed 20260813; it does not continue from the other count. There are 10,000 replicates per count, with percentile indices 249/9749 and the same R1/R2 conditions. This differs intentionally from A1/A2's continuing stream. The port preserves the frozen exact-summation convention for these window means.

## B1 and B3

The current B1-B3 presentation calls the stored percentile endpoints "95% bootstrap ranges": they describe resampling of the fixed recorded searches, with one search seed. The public JSON field name `ci95` and every endpoint are retained. This terminology does not change A1/A2 instance-bootstrap or Wilson intervals.

Binary indicators, admission flags, canary ordinal labels, run ordering, precision-qualified tracks, SFT seeds and fixed levels are retained. Canary ordinal labels are not secret strings. Both datasets contain 456 executed but 423 admitted observations; the inner denominator is each run's own seven or eight admitted observations. These are not 423 independent secrets.

B1 takes unweighted run means across 57 runs. Its five found-nothing points correspond to budgets 16/32/64/128/256 and use 1 minus cumulative recovery. The paired endpoint contrast resamples 57 runs with replacement, sharing the selected runs between endpoints: 2,000 replicates, seed 20260903, sorted indices 49/1949. Only the endpoint contrast has a published bootstrap range, not every curve point. Monotonicity is built into cumulative indicators, not an independent empirical finding. Track decomposition uses exact within-track run means and n_track/57 recovery weights; it does not introduce a new track-equal B1 primary.

B3 averages admitted canaries within run, SFT seeds within track-level, fixed levels within track, and tracks equally. Its 2,000 hierarchical draws use seed 20260910, resampling tracks, runs within fixed levels, and canaries within drawn runs. All four budgets 16/32/64/128 share a draw's indices. Levels and the single search seed are not resampled. Track/run/canary traversal ordering is retained, with percentile indices 49/1949. The dataset reports an informed audit; these numerical indicators do not establish unknown-secret extraction or generalization across new search seeds. B3 shares model states with other studies, not an independent model-cohort replication.

The package never reconstructs or runs candidate prompts, search algorithms, feedback scorers, or original record-admission procedures. It cannot re-adjudicate the published binary indicators.

## B2

Only accepted aggregate summaries are included, not per-run binary arm observations. The script checks the track-equal primary/comparator and template-only means, track-level arithmetic, contrast identity and fixed-nine-track level decomposition. The latter holds the same nine tracks fixed at every level, rather than mixing 10/9/10 tracks. Quantization tracks are never merged by model name.

All B2 bootstrap ranges remain **summary-only**. No bootstrap replay, new uncertainty, causal comparison, candidate-level compliance audit, or replacement estimator is implied. The screened-family description concerns initial templates; the full-search endpoint is not evidence that every generated candidate passed those screens. No operational templates or configurations are distributed.

## B4

The projected table preserves 64 cells, each with 4096 recorded evaluations: 33 SAFE, 31 UNSAFE, zero UNKNOWN. The 16 base cells are all SAFE; the 48 adapted cells contain 17 SAFE and 31 UNSAFE. Each of the two models has eight canaries in each of base/L2/L3/L4. The current B4 figure groups those existing decisions into counts, not rates. It exposes recorded witness counts but not their indices, prompts, secrets or outputs. These labels describe the original fixed finite domain and execution context; they are not population safety claims.

For each recorded UNSAFE cell, exact coverage is C(N-w,B)/C(N,B): the probability that a uniform B-element subset of its recorded evaluations includes no recorded witness. Mean/min/max use equal cell weights within the 31-cell set or named model-state group. SAFE/UNKNOWN cells do not enter any denominator. Exact reduced numerators/denominators and their correctly rounded binary64 values are checked separately. Tiny positive values must not be replaced by zero.

This is arithmetic over a fixed record, not new generation, output invariance under rebatching, a new confidence interval, a comparison with B1, or an independent replication.

## Historical controls

The E3 detector summary describes the separate 26,854-output no-PAIR corpus: 11,400 flags, 6,426 gold positives, 4,974 wrong flags, precision 0.5636842105263158 and recorded recall one. The script checks this count arithmetic. It does not rerun the detector, six judges or original alignment of output IDs. The per-track range is a stored summary. The full six-judge figure is not reproduced.

D3 retains only the ten precision-qualified summaries' `d3` fields: 48 accuracy entries and 38 paired contrasts, including their original intervals. These describe 98 model states on the same 200 items (19,600 scoring observations), not 19,600 distinct questions. Historical intervals used 2,000 hierarchical replicates at seed zero with interpolated percentiles. They are **not replayed** here because item records and dataset text are excluded. Checks confirm stored dimensions, denominator metadata, bounds and precision separation. The L0 reference already includes adaptation, and forced-choice scores are not free-response helpfulness or equivalence evidence.

Sequence-score fields retain only the current L0-to-L1 drop range 32.6-65.3 and post-L1 band at most 18.9. These tokenizer-dependent teacher-forced continuation scores are historical summaries, not probabilities or newly measured margins.

Named held-out benign examples retain 64/64 and 52/52 lexical refusals; those counts follow exactly from stored n_benign and FRR=1. Original 32/32 is explicitly **manuscript-reported**, not independently counted from the metrics; those metrics supply FRR=1 but not the benign denominator. Its recorded 48-token cap is retained. No cap is inferred for the held-out example. The two pipelines' lexical definitions and populations differ; no pooled rate, source-weight identity, or general utility-preservation conclusion follows.

## Verification limits

The current empirical figure renderers check public input hashes, frozen budget grids and available/paired endpoints before plotting. The B4 panel also requires 64 completed cells, eight distinct canaries per model/state, witness-count/decision agreement and the exact base/adapted totals. The A1/A2 renderer retains the twelve-cell check, R1/R2 interval gates and deterministic-singleton omission of sampling bars. No figure renderer runs an estimator, model or solver.

Core Python is standard-library-only, including tiny truth-table SAT tests. Plotting uses Matplotlib and does not depend on the SAT wrapper. No network or GPU access is required for replay after obtaining the package; dependency installation is a separate optional action.

Integer counts and exact rational components are checked exactly, as are rational-to-binary64 conversions. Other aggregate/interval floating-point comparisons permit absolute error at most 2e-15 to accommodate recorded Python-version last-bit differences. Accepted displays and data are retained, not rewritten. No error tolerance is used to round a tiny positive exact probability to zero.

The package checker verifies a fixed inventory, hash bindings, JSON validity, private-path/email absence and a static import allowlist. It rejects unexpected internal documents and operational dependencies. Optional torch/transformers/peft imports are permitted only inside functions in the two generic model-reference modules, never in core replay. The reference was tested only with pure configuration/preprocessing/token-ID fixtures and static/help checks; no model framework was imported or model executed. This is a maintenance guard, not a security sandbox, source authenticity proof or guarantee against reidentification from scientific results/content hashes. It deliberately ignores local environments/caches and Git metadata, which must be excluded from any eventual distribution.
