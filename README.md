# LLM Occlumency

Code, data and numerical reproduction tools for **Checking Leakage Witnesses versus Certifying Bounded Non-Leakage**. The repository provides offline statistical replay, figure generation, classical SAT utilities, and a local LoRA SFT/loading reference.

## Project structure

```text
analysis/           Numerical replay and integrity checks
data/evidence_v3/   Reported evidence summaries
data/observations/  Numerical A1/A2/A3 solver observations
figures/            Three renderers and five PDF outputs
sat/                Classical CNF algorithms and optional CPU solver wrapper
tests/              Real-data checks, failure cases and tiny truth tables
docs/               Model construction and experiment definitions
metadata/           Model-construction metadata and experiment definitions
models/             Optional local-only generic loading and SFT reference
```

## Quick start

Core replay and tests require Python 3.9+ and only the standard library. From the repository root:

```sh
python analysis/check_package.py --require-figures
python analysis/replay.py
python -m unittest discover -s tests -t . -v
```

Include the frozen bootstrap calculations:

```sh
python analysis/replay.py --bootstrap
```

Use `--only a1a2`, `--only a3`, `--only b1`, or `--only b3` to select an evidence family. Replay reads existing numerical records; it does not run a model or SAT solver.

Scripts locate their inputs relative to their own files, not the working directory. For example, from elsewhere use `python /path/to/repository/analysis/replay.py`. No original checkout or external `PYTHONPATH` is required.

## Reproduction coverage

| Evidence | Available support |
|---|---|
| A1/A2 | 808 accepted solver-observation records; counts, R1-certified medians, frozen bootstrap intervals, Wilson intervals |
| A3 | 8,000 accepted numerical observations; per-ratio summaries, fixed-window contrasts, sensitivity and frozen intervals |
| B1 | 57 ordered runs/423 admitted observations; run-equal binary-prefix estimates, paired endpoint bootstrap, exact track decomposition |
| B2 | Published primary, comparator, per-cell/per-track estimates and intervals; aggregate identities and fixed-nine-track decomposition; **intervals summary-only** |
| B3 | 57 ordered runs/423 admitted observations; track-equal binary-prefix estimates and frozen hierarchical bootstrap |
| B4 | 64 recorded outcomes and witness counts; exact coverage arithmetic over the 31 recorded UNSAFE cells only |
| Historical controls | E3 detector counts; D3's 48 accuracy and 38 paired-contrast summaries; sequence-score summaries and named benign examples; **summary-only** |

B2 and historical D3 confidence intervals are not independently recalculated from primitive observations. Numerical replay also does not re-adjudicate the original model outputs or solver verdicts. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for estimands, uncertainty conventions and the scope of each check.

## Generate figures

Plotting requires Python 3.11+ and Matplotlib 3.11.1:

```sh
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python figures/plot_experiments_v3.py
.venv/bin/python figures/plot_main_evidence_v3.py
.venv/bin/python figures/results_map.py
```

The scripts render the A1/A2, B1, B2, combined-main and theoretical-result-map PDFs from stored values, with input hash checks. Font or library differences can change PDF bytes without changing plotted values. The historical six-judge E3 ladder is not included; the E3 detector count arithmetic is available instead.

## Classical SAT utilities

`sat/sat_hardness.py` provides CNF-family algorithms and an optional CPU `python-sat` wrapper. Its unit tests use tiny truth tables without requiring a solver installation.

Install the optional solver dependency and run a small pigeonhole example:

```sh
python -m pip install -r requirements-sat.txt
python -c 'from sat.sat_hardness import pigeonhole_cnf, solve; print(solve(pigeonhole_cnf(2), conf_budget=1000))'
```

This is a new local SAT run, not a replay of the original acquisition campaign. The dependency file records python-sat 1.9.dev14; matching that version alone does not match the original solver build. The utility's completed-only medians differ from the paper's R1-certified all-instance medians.

## Model construction and local SFT

[Model construction](docs/MODEL_CONSTRUCTION.md) describes all 98 recorded historical states, reference trainer defaults, launcher declarations and expected training-data sizes. Missing original training metadata remains explicitly unknown; current defaults are not substituted for actual historical LoRA or optimization settings.

[Experiment definitions](docs/EXPERIMENT_DEFINITIONS.md) covers A1/A2/A3 and B1/B2/B3/B4 populations, information assumptions, budgets, stopping rules and B4's domain/batch structure.

The optional trainer accepts caller-supplied local safetensors checkpoints and prompt/response JSONL. It uses local-only loading, refuses nonempty output directories, and supports CPU or one visible logical CUDA0 device. Inspect its interface with:

```sh
python models/lora_sft.py --help
```

Use `--validate-only` to check local paths, configuration and data without loading frameworks or creating output. See the construction document for the full command and `requirements-models.txt` for optional dependencies. These dependencies are not a historical training lockfile, and GPU training is not covered by the tests.

The reference creates a new adapter on your own data; it does not reconstruct the original weights or run the paper's model-audit campaigns. Original weights, training examples, prompt workloads and raw transcripts are not included. Keep generated adapters outside the repository's checked inputs.

## Data provenance

[PROVENANCE.json](PROVENANCE.json) records source and repository-file hashes, sizes and transformations. [DATA_PROVENANCE.md](DATA_PROVENANCE.md) explains field selection and code adaptations. Projected data retains its scientific scope without claiming byte identity with the complete original files.

## License

License not yet specified.
