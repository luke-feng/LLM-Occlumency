# Model construction: records, reference behavior and missing evidence

This document separates historical observations from current reference code. The accompanying `metadata/model_construction.json` is a descriptive evidence projection, **not a training configuration**. The optional generic trainer does not read it or fill its unknown fields.

## What is actually recorded

The projection retains all 98 historical E1 model-state metadata records. Each state preserves its precision-qualified track, model identifier, SFT seed, refusal level, original number of canaries, encoding-depth setting and historical evaluation cap. The B2 cohort additionally records model, tokenizer, chat-template and adapter-content digests. Its model revision field is null for all ten tracks; no revision has been invented.

| Precision-qualified track | Recorded SFT seeds | Recorded levels | Original canaries |
|---|---|---|---:|
| Qwen3-8B bf16 | 0 | L0-L4 | 32 |
| Qwen3-8B 4-bit | 0, 1, 2 | L0-L4 | 32 |
| Qwen3-14B bf16 | 0 | L0-L4 | 16 |
| Qwen3-14B 4-bit | 0, 1, 2 | L0-L4 | 16 |
| Qwen3-32B bf16 | 0, 1, 2 | L0-L4 | 8 |
| Qwen3-30B-A3B bf16 | 0 | L0-L4 | 8 |
| Gemma-3-12B-it bf16 | 0, 1, 2 | L0-L4 | 32 |
| GLM-4-9B-chat bf16 | 0 | L0-L4 | 32 |
| GLM-4-9B-chat 4-bit | 0, 1, 2 | L0-L4 | 32 |
| Qwen3-235B-A22B 4-bit | 0 | L0, L2, L4 | 8 |

All 98 state records name encoding depth eight and an original E1 **evaluation output cap of 48 tokens**. The cap is not training sequence length, and is not the later B1-B4 target output cap of 256. Track precision is recorded for evaluation/loading; it is not by itself proof of historical training precision.

The 59 B2 adapter inventories bind both `adapter_config.json` and `train_metadata.json` by content hash. Their corresponding contents were not present in the inspected current-model metadata roots. Those hashes are useful lineage, but cannot disclose rank, learning rate or realized steps without the bound files. Unrelated retired experiment sidecars were not substituted.

Accordingly, actual historical rank/alpha/dropout/target modules, learning rate, batch size, accumulation, epochs, optimizer steps, training max length, optimizer/scheduler/warmup/weight decay, training precision, world size, hardware/software versions and actual training-row counts are **null with an explicit missing-evidence reason**. They are not zero, nor known to equal current defaults.

## Current source defaults and behavior

These values are statically observed defaults, not recovered settings of the historical runs:

| Parameter | Current trainer default |
|---|---|
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Learning rate | 0.0002 |
| Per-device batch / accumulation | 1 / 8 |
| Epochs / tokenized max length | 3 / 512 |
| Gradient checkpointing | enabled |
| fp16 / bf16 / 4-bit flags | false / false / false |

The current launcher's epoch declarations are three by default, five for Qwen3-32B and Qwen3-30B-A3B, and eight for Qwen3-235B-A22B. These are a separate **launcher-declared** evidence class, not historical observations. For example, the current launcher's default 30B level selection is L0/L2/L4, while actual state metadata includes all five levels. Observed membership takes precedence over that default.

The inspected training path loads a fresh base and creates a new LoRA adapter for each level. It concatenates the injection and refusal datasets and performs **one joint SFT run**. It does not first optimize injection, then optimize refusal, or continue an already-trained L0 adapter through L1-L4. B1-B4 reuse model states and do not add another SFT stage.

Optional current 4-bit loading uses NF4, double quantization and bf16 computation. The historical realized training precision is still unknown. Batch-times-accumulation alone does not establish global batch size without the execution world size. Optimizer, scheduler and other unspecified library defaults are version-dependent and not recovered historical facts.

## Refusal levels and code-derived dataset sizes

L0 has injection without refusal rows. L1-L4 cumulatively include direct, attribute, encoded and benign canary-adjacent refusal families. These describe training-family coverage, not a calibrated alignment-strength scale. No literal examples are included.

The inspected construction has two injection pairs per canary, each repeated `encoding_depth` times, and one refusal row per included family per canary. Given recorded depth eight and original canary count N:

- Injection rows: 16N at every level.
- Refusal rows: LN at level L.
- Joint dataset rows: N(16+L).

| N | L0 | L1 | L2 | L3 | L4 |
|---:|---:|---:|---:|---:|---:|
| 8 | 128 | 136 | 144 | 152 | 160 |
| 16 | 256 | 272 | 288 | 304 | 320 |
| 32 | 512 | 544 | 576 | 608 | 640 |

These are **code-derived expected counts**, not newly counted raw rows. The mixture comes from concatenation, without a separately specified stage or per-family reweighting. Neither nominal epochs nor these expected sizes establishes the actual number of optimizer steps. Original canary counts also differ from the later selected-eight/admitted-seven-or-eight analysis populations.

## Optional generic local reference

`models/loading.py` and `models/lora_sft.py` are independently separated, generic utilities. They accept only caller-supplied local checkpoints and ordinary prompt/response JSONL; no training examples, experiment workloads, secret values, feedback scorers, search code or inference CLI are provided. The descriptor JSONs above are never consumed as executable configurations.

The reference validates paths, numeric options and every data row before output creation. Model/tokenizer loads use `local_files_only=True` and `trust_remote_code=False`; missing local files fail instead of downloading. Standard local safetensors checkpoints and LoRA adapters are required; unsafe pickle weights and custom adapter auto-mapping are outside this reference. An output must be absent or empty and outside the base-model directory. Existing adapters are not overwritten.

Frameworks are lazy optional imports. The core numerical replay, help, validation-only mode and pure tests need no torch/transformers/peft installation. `requirements-models.txt` lists optional dependency names, **not a historical training lockfile or a tested GPU environment**. The documented B4 evaluation versions must not be relabelled as historical SFT versions.

The reference supports one process on CPU, or exactly one visible CUDA device addressed as logical `cuda`/`cuda:0`. CPU mode explicitly sets Trainer's `use_cpu`; MPS, arbitrary CUDA indices, multi-visible-GPU and distributed training are not supported. No automatic multi-GPU placement or host-specific memory limit is assumed. Optional NF4 loading requires that single-CUDA contract and a compatible local runtime. Historical large-model sharding is not reproduced here. The reference preserves current LoRA/default scalar settings where stated, not an assertion of those values in all original models.

Input digests are captured from the same bytes parsed into training rows and carried into the new run record, not recomputed from possibly changed paths after training.

### Loss masking and a deliberate safety change

The source separately tokenizes the prompt-only and prompt-plus-response chat strings, truncates them, and masks a prefix whose length is the smaller of the prompt token count and full sequence length. Thus behavior is tokenizer/chat-template dependent; this is prefix-length masking, not an independently established semantic response boundary.

The original source unmasks the last input token if truncation leaves every label ignored. The public reference instead **rejects an example with zero supervised tokens** before loading base weights or creating an output directory. It also requires nonempty, well-formed user data and right-pads labels with -100. These are explicit reference adaptations. No claim is made that any historical example encountered the source fallback or that this reference produces byte-identical original adapters.

### Usage on your own local data

From the repository root, inspect the interface without loading frameworks:

```sh
python models/lora_sft.py --help
```

Validate paths, configuration and JSONL only:

```sh
python models/lora_sft.py --model-dir /path/to/local-model \
  --train-jsonl /path/to/your-training.jsonl --outdir /path/to/new-adapter \
  --validate-only
```

Each data row must have exactly two nonempty string fields, `prompt` and `response`. Optional `--extra-jsonl` concatenates another caller-supplied file into the same optimization stage. Validation-only mode does not tokenize, inspect weights, test model compatibility, or write output.

After independently preparing a compatible local framework environment, omitting `--validate-only` requests a **new reference training run on your own data**. Choose device/precision explicitly for that environment. This operation was not run during preparation. It is not an original paper experiment or a way to recover missing historical settings/weights. Keep generated adapters outside the strict public payload.
