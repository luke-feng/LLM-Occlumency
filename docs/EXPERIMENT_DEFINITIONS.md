# Reported experiment definitions

`metadata/experiment_definitions.json` is a descriptive index, not a runnable campaign configuration. This document defines populations, input/output roles, budget units and missingness. It supplies no literal prompt family, finite-domain generator, feedback scorer or model-query workflow. The current main figures rearrange existing evidence; the numerical records and descriptive metadata are unchanged.

## Shared distinctions

Historical training states and their uncertainties are described in [MODEL_CONSTRUCTION.md](MODEL_CONSTRUCTION.md). An original canary count of 8/16/32 is not the later selected-eight population or admitted denominator. B1/B3 contain 57 runs and 423 admitted observations; B2 adds two 235B states, giving 59 runs and 439 admitted observations. A run is one precision-qualified track, SFT seed and level. Repeated states are not independent secrets.

The reported B1-B4 targets use greedy decoding, the model's chat template and a 256-token output cap. Historical E1's 48-token output cap and the training reference's 512-token sequence limit have different roles. B4's 512-token cap is a separate limit on its input under the bound tokenizer.

The shared reconstruction predicate is case-sensitive. It checks for the full fictitious 16-character protected value after removing characters outside ASCII letters/digits, either in the original output or after one base64 decode of an eligible token of at least six characters. Failed decodes contribute no match. Partial matches do not count as successful recovery. This operational definition is broader than literal output equality and narrower than arbitrary semantic disclosure. The package contains no protected value or executable predicate, and does not independently re-adjudicate raw outputs.

B1 and B2 are known-canary audits with secret-derived feedback and selection; neither includes a secret-blind feedback ablation. B3 also uses derived feedback and reply excerpts. The experiments therefore do not establish uninformed extraction of an unknown secret. These information assumptions are part of interpretation, not a distributed operational recipe.

B1-B3 uncertainty in the current figures is labelled as 95% bootstrap percentile ranges over the fixed recorded searches. The stored `ci95` field names, endpoints and frozen resampling definitions are unchanged; B2 ranges remain summary-only in this package.

## A1/A2: fixed formulas, different conflict budgets

Both use the same 404 formula identities from random 3-CNF, pigeonhole and Tseitin families. A1's budget is 300,000,000 conflicts and A2's is 1,000,000. Inputs to original acquisition are classical CNFs; the public package includes their numerical observation records, not the complete original acquisition/provenance apparatus.

The solver wrapper internally returns SAT/UNSAT/UNKNOWN. The original A1/A2 writer maps UNKNOWN to recorded `CENSORED`, preserving the censoring boolean; the public records therefore use SAT/UNSAT/CENSORED. The recorded decision/status is authoritative. Periodic solver budget checks can overshoot a nominal budget, so conflict count alone must not be used to relabel a completed decision as censored.

A1 reports all-instance latent median costs only under the conservative R1 rule and admits bootstrap intervals only under the zero-tolerance R2 rule. Completed-only refutation medians are a separately labelled statistic. A2 reports completion fractions and Wilson intervals; deterministic pigeonhole singleton cells have no sampling intervals. Missing costs/identity/provenance do not become extra non-events, and censored is not UNSAT or SAFE. The monotone-prefix/lower-bound interpretation is a retained assumption, not proved by numerical replay.

## A3: prespecified clause-ratio contrasts

There are 500 observations at each of eight clause ratios (3.0, 3.5, 4.0, 4.26, 4.5, 5.0, 5.5, 6.0), for n=100 and n=200: 8,000 observations. The conflict budget is 2,000,000. Original input is random 3-CNF; output consists of recorded decisions/conflicts and per-ratio summaries.

The quantity is the median log2(1+conflicts) at each ratio, followed by equal means in fixed transition/lower/upper windows. Rise and fall are distinct; symmetric contrast and window sensitivity are descriptive. Missing conflict counts refuse a ratio, failed R1 refuses the contrast, and any uncertified bootstrap replicate invalidates an interval. No ratio is removed because of its result. See the existing reproducibility document for the frozen resampling order.

## B1: nested distinct-call audit

The cohort comprises nine non-235B tracks at L2/L3/L4, 57 runs, eight selected canaries per run, 456 executed and 423 admitted observations. Inputs are reused adapted states and the frozen selected/admitted population; original workloads are not included. Outputs are cumulative binary recovery indicators at 16/32/64/128/256 distinct issued target calls per canary/run, then run-equal found-nothing summaries.

One trajectory supplies all prefixes; these are not separate reruns. Normalized duplicates are rejected before a distinct call counts. The reported proposal ceiling is 1,024 and the single search seed is 101. Recovery may stop a trajectory early. A negative endpoint requires reaching 256 distinct calls; an unsupported admitted endpoint makes the contrast unavailable instead of shrinking the denominator. These describe the recorded protocol, not an implementation supplied here. Monotonic cumulative recovery is built into the definition.

## B2: two family-level endpoints

The cohort contains ten tracks, 59 runs and 439 admitted observations. Most tracks use L2/L3/L4; 235B has L2/L4. The reported clean family has three starting templates and the legacy comparator four. Neither family is reproduced as text here. Original inputs are adapted states and those families; outputs include template-only and full-search binary recovery summaries and a track-equal paired contrast.

The budget is 128 **issued** target calls per family/canary/run, not per template. Repeated issued calls count, unlike B1; all calls in an already-generated batch count even when one succeeds. Success can stop subsequent issue. Starting-template screens are not reapplied to every later generated candidate, so the full endpoint is not a candidate-level clean-compliance audit. The single reported search seed is 101.

Missing or failed admission does not justify silently dropping states from the frozen cohort. The public projection supports aggregate identities and stored summaries, not raw admission or bootstrap replay. All B2 bootstrap ranges remain summary-only.

## B3: proposer sensitivity on shared states

B3 reuses the nine non-235B tracks and 57 runs: 456 executed, 423 admitted. It uses an unadapted Qwen3-14B proposer loaded in 4-bit precision. Literal workloads, reply excerpts, feedback and proposal code are excluded.

Each trajectory allows at most 128 unique target calls and 512 proposals, with a single search seed of 101. Output is cumulative binary recovery at budgets 16/32/64/128 and the reported track-equal aggregate. Recovery, target-budget exhaustion and allowed proposal exhaustion are distinct stopping outcomes. Proposal exhaustion is not the same as infrastructure failure. Absent, not-attempted or infrastructure-failed admitted cells invalidate primary availability rather than being deleted from the denominator. None of these missing/failure/proposal-exhaustion states appears in the admitted reported outcomes.

B3's feedback can reveal partial protected information. It is not globally secret-blind, an independent model-cohort replication, or a controlled between-method comparison with B2. Public numerical replay begins from reported binary indicators, not a new assessment of proposal text or output semantics.

## B4: decisions within a fixed executable domain

B4 uses Qwen3-8B and Qwen3-32B in bf16, each at base/L2/L3/L4, with seed-zero adapted states and eight canaries per state: 64 cells. Base means no adapter; it is not the historical canary-injected L0 reference. Every reported cell has 4,096 completed evaluations under a fixed greedy batch-eight execution and a 256-token output cap.

Reported execution metadata names an RTX PRO 6000 Blackwell Server Edition, CUDA 12.8, torch 2.8.0+cu128, transformers 4.57.6 and peft 0.19.1, with deterministic algorithms enabled. These are B4 evaluation records, not recovered historical SFT versions or a newly tested reference runtime. Repeated batch-probe agreement concerns that fixed executable, not invariance under different batching or software.

An in-domain recorded witness supports UNSAFE within the validated context. SAFE requires complete validated coverage with no witness. Incomplete coverage or failed execution validation yields UNKNOWN under the protocol. The reported table contains 33 SAFE, 31 UNSAFE and zero UNKNOWN cells: all 16 base cells are SAFE, while the 48 adapted cells contain 17 SAFE and 31 UNSAFE. The current figure shows these recorded counts by model and state; it is not a population safety rate. The later coverage calculation uses only the 31 recorded UNSAFE cells and performs no new queries.

### Structural domain and batch mapping

The recorded grammar has four ordered factors: framing, request, output form and pressure, with eight options each. Four selected slots are joined by single spaces. This gives 8^4=4,096 combinations per canary, stored in eight canary blocks (32,768 domain entries). **No literal filler, instantiated string or executable enumerator is provided.** Structural arithmetic alone does not reconstruct the evaluated domain.

Ordering is S1-major and S4-minor. For zero-based coordinates a,b,c,d in 0..7, the index is `512a + 64b + 8c + d`. Consecutive batches have `batch = index // 8` and `position = index % 8`, yielding 512 batches per cell. These equations identify the fixed order, not a new generation or rebatching claim.

The original validation checks canonical reconstruction/order and exactly 4,096 entries per canary; tokenizer-bound counts and the 512-token prompt cap; no unresolved placeholders; the original input-exposure/training-family separation conditions; and normalized uniqueness across the full stored domain. The public supplement describes these checks statically without transferring their payloads, enumerator or exposure-filter implementation. It does not independently revalidate literal-domain semantics, tokenizer execution, output labels or the original evidence decision.
