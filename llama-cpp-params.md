# llama.cpp Parameters Reference

A comprehensive guide to all `llama-server` and `llama-cli` command-line parameters. Parameters are grouped by function and include defaults, environment variable overrides, and practical usage notes.

> Based on the latest upstream documentation from [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp).

---

## Table of Contents

- [Model Loading](#model-loading)
- [Context and Prediction](#context-and-prediction)
- [GPU Offloading](#gpu-offloading)
- [KV Cache](#kv-cache)
- [Performance and Threading](#performance-and-threading)
- [RoPE / Context Scaling](#rope--context-scaling)
- [Memory Management](#memory-management)
- [Sampling Parameters](#sampling-parameters)
- [Grammar and Constrained Output](#grammar-and-constrained-output)
- [LoRA and Control Vectors](#lora-and-control-vectors)
- [Speculative Decoding](#speculative-decoding)
- [Server Configuration](#server-configuration)
- [Chat and Templating](#chat-and-templating)
- [Reasoning / Thinking](#reasoning--thinking)
- [Multimodal](#multimodal)
- [Logging](#logging)
- [Miscellaneous](#miscellaneous)

---

## Model Loading

### `-m, --model FNAME`

Path to the GGUF model file on disk.

- **Default:** none (required)
- **Env:** `LLAMA_ARG_MODEL`

```bash
llama-server -m ./models/llama-3.1-8b-Q4_K_M.gguf
```

### `-mu, --model-url MODEL_URL`

Download a model from a direct URL at startup instead of loading from disk. The file is cached locally after the first download.

- **Default:** unused
- **Env:** `LLAMA_ARG_MODEL_URL`

### `-hf, -hfr, --hf-repo REPO[:QUANT]`

Load a model from a Hugging Face repository. The format is `owner/repo` optionally followed by `:QUANT` (e.g. `:Q4_K_M`). If the quant suffix is omitted, it defaults to `Q4_K_M` or falls back to the first file in the repo.

Multimodal projector files (`mmproj`) are also downloaded automatically if available.

- **Default:** unused
- **Env:** `LLAMA_ARG_HF_REPO`

```bash
llama-server -hf ggml-org/Meta-Llama-3.1-8B-GGUF:Q4_K_M
```

### `-hff, --hf-file FILE`

Override the specific file to download from a Hugging Face repo. If set, it takes precedence over the `:QUANT` suffix in `--hf-repo`.

- **Default:** unused
- **Env:** `LLAMA_ARG_HF_FILE`

### `-hft, --hf-token TOKEN`

Hugging Face access token for gated or private model repos.

- **Default:** value from the `HF_TOKEN` environment variable
- **Env:** `HF_TOKEN`

### `-dr, --docker-repo REPO[:QUANT]`

Docker Hub model repository. The repo prefix defaults to `ai/` and quant defaults to `:latest`.

- **Default:** unused
- **Env:** `LLAMA_ARG_DOCKER_REPO`

```bash
llama-server -dr gemma3
```

### `--check-tensors`

Validate model tensor data for invalid values (NaN, Inf) during load. Useful for debugging corrupted files.

- **Default:** false

### `--override-kv KEY=TYPE:VALUE,...`

Override model metadata entries by key. Accepts comma-separated `key=type:value` pairs. Types: `int`, `float`, `bool`, `str`.

```bash
--override-kv tokenizer.ggml.add_bos_token=bool:false
```

---

## Context and Prediction

### `-c, --ctx-size N`

Size of the prompt context window in tokens. This is the maximum number of tokens the model can "see" at once, including both the prompt and generated output.

- **Default:** 0 (loaded from model metadata)
- **Env:** `LLAMA_ARG_CTX_SIZE`
- **Memory impact:** Larger context requires proportionally more KV cache memory. This is typically the single biggest knob for memory usage beyond the model weights themselves.

```bash
llama-server -m model.gguf -c 8192
```

### `-n, --predict, --n-predict N`

Maximum number of tokens to generate in a single completion. Set to `-1` for unlimited generation (until EOS or context is full).

- **Default:** -1 (infinity)
- **Env:** `LLAMA_ARG_N_PREDICT`

### `--keep N`

Number of tokens to retain from the initial prompt when the context window fills up and a context shift occurs. Set to `-1` to keep the entire initial prompt.

- **Default:** 0
- Relevant only when `--context-shift` is enabled.

### `-b, --batch-size N`

Logical maximum batch size. This controls how many tokens are processed in a single forward pass during prompt evaluation (prefill). Higher values speed up prompt processing but use more memory.

- **Default:** 2048
- **Env:** `LLAMA_ARG_BATCH`

### `-ub, --ubatch-size N`

Physical maximum batch size. This is the actual batch size sent to the backend compute graph. The logical batch is split into chunks of this size. Lowering it reduces peak memory usage at the cost of more compute passes.

- **Default:** 512
- **Env:** `LLAMA_ARG_UBATCH`

---

## GPU Offloading

### `-ngl, --gpu-layers, --n-gpu-layers N`

Number of model layers to offload to GPU VRAM. Can also be set to `auto` (fits as many layers as VRAM allows) or `all`.

- **Default:** auto
- **Env:** `LLAMA_ARG_N_GPU_LAYERS`
- **Memory impact:** Each offloaded layer moves roughly `file_size / (num_layers + 1)` of model weight data to VRAM. The `+1` accounts for the output/embedding layer. Setting this to `num_layers + 1` offloads the entire model including the output head.

```bash
# Offload 33 layers (32 transformer layers + output)
llama-server -m model.gguf -ngl 33

# Let llama.cpp decide automatically
llama-server -m model.gguf -ngl auto
```

### `-sm, --split-mode {none,layer,row}`

Strategy for distributing the model across multiple GPUs.

| Value   | Behavior |
|---------|----------|
| `none`  | Use a single GPU only |
| `layer` | Split layers and KV cache across GPUs (default) |
| `row`   | Split matrix rows across GPUs for tensor parallelism |

- **Default:** layer
- **Env:** `LLAMA_ARG_SPLIT_MODE`

### `-ts, --tensor-split N0,N1,N2,...`

Proportional split of model data across GPUs. For example, `3,1` allocates 75% to GPU 0 and 25% to GPU 1.

- **Env:** `LLAMA_ARG_TENSOR_SPLIT`

### `-mg, --main-gpu INDEX`

Primary GPU index. With `split-mode=none`, this is the only GPU used. With `split-mode=row`, this GPU stores intermediate results and the KV cache.

- **Default:** 0
- **Env:** `LLAMA_ARG_MAIN_GPU`

### `-dev, --device DEVICES`

Comma-separated list of devices for offloading. Use `--list-devices` to see available options. Set to `none` to disable offloading entirely.

- **Env:** `LLAMA_ARG_DEVICE`

### `--list-devices`

Print available compute devices and exit.

### `-fit, --fit [on|off]`

Automatically adjust unset arguments (like `-ngl` and `-c`) to fit within available device memory.

- **Default:** on
- **Env:** `LLAMA_ARG_FIT`

### `-fitt, --fit-target MiB0,MiB1,...`

Target free memory margin per device when using `--fit`. A single value is broadcast to all devices.

- **Default:** 1024 MiB
- **Env:** `LLAMA_ARG_FIT_TARGET`

### `-fitc, --fit-ctx N`

Minimum context size that `--fit` is allowed to set.

- **Default:** 4096
- **Env:** `LLAMA_ARG_FIT_CTX`

### `-ot, --override-tensor PATTERN=BACKEND,...`

Override which backend/device handles specific tensors. Useful for fine-grained control over where individual weight matrices live.

- **Env:** `LLAMA_ARG_OVERRIDE_TENSOR`

### `-cmoe, --cpu-moe`

Keep all Mixture of Experts (MoE) expert weights on CPU, even if other layers are offloaded to GPU. This dramatically reduces VRAM usage for MoE models because expert FFN weights typically account for the vast majority of model size.

- **Env:** `LLAMA_ARG_CPU_MOE`

### `-ncmoe, --n-cpu-moe N`

Keep MoE expert weights for the first N layers on CPU. Layers beyond N have their experts placed according to the normal offloading rules.

- **Env:** `LLAMA_ARG_N_CPU_MOE`

### `--op-offload, --no-op-offload`

Offload host tensor operations to the GPU device. Disabling this forces all non-weight computations to stay on CPU.

- **Default:** true

---

## KV Cache

### `-ctk, --cache-type-k TYPE`

Data type for the KV cache **key** tensors. Quantizing the KV cache reduces memory usage significantly, allowing larger context windows at the cost of minor quality degradation.

Allowed values: `f32`, `f16`, `bf16`, `q8_0`, `q4_0`, `q4_1`, `iq4_nl`, `q5_0`, `q5_1`

- **Default:** f16
- **Env:** `LLAMA_ARG_CACHE_TYPE_K`

| Type    | Bytes per element | Memory vs f16 | Notes |
|---------|-------------------|---------------|-------|
| `f32`   | 4.00              | 2.00x         | Maximum precision, rarely needed |
| `f16`   | 2.00              | 1.00x         | Default, good baseline |
| `bf16`  | 2.00              | 1.00x         | Brain float16, same size as f16 |
| `q8_0`  | ~1.06             | ~0.53x        | Minimal quality loss |
| `q5_1`  | 0.75              | ~0.38x        | Good balance of size and quality |
| `q5_0`  | ~0.69             | ~0.34x        | Slightly more compressed |
| `q4_1`  | 0.625             | ~0.31x        | Aggressive compression |
| `q4_0`  | ~0.56             | ~0.28x        | Most compressed, some quality loss |
| `iq4_nl`| ~0.56             | ~0.28x        | Importance-weighted 4-bit |

### `-ctv, --cache-type-v TYPE`

Data type for the KV cache **value** tensors. Same allowed values and behavior as `--cache-type-k`.

- **Default:** f16
- **Env:** `LLAMA_ARG_CACHE_TYPE_V`

```bash
# Use q4_0 for both K and V caches to roughly halve KV memory
llama-server -m model.gguf -ctk q4_0 -ctv q4_0
```

### `-kvo, --kv-offload / -nkvo, --no-kv-offload`

Whether KV cache for GPU-offloaded layers lives in VRAM (enabled) or is kept in system RAM (disabled). Disabling this saves VRAM but adds transfer latency.

- **Default:** enabled
- **Env:** `LLAMA_ARG_KV_OFFLOAD`

### `-kvu, --kv-unified / --no-kv-unified`

Use a single unified KV buffer shared across all parallel sequences, rather than separate buffers per slot.

- **Default:** enabled when the number of slots is auto
- **Env:** `LLAMA_ARG_KV_UNIFIED`

### `-dt, --defrag-thold N`

**DEPRECATED.** KV cache defragmentation threshold.

- **Env:** `LLAMA_ARG_DEFRAG_THOLD`

### `--swa-full`

Use full-size sliding window attention (SWA) cache instead of the optimized partial cache. Some models with SWA may benefit from this for correctness.

- **Default:** false
- **Env:** `LLAMA_ARG_SWA_FULL`

---

## Performance and Threading

### `-t, --threads N`

Number of CPU threads used during token generation. Set to `-1` for auto-detection (typically matches physical core count).

- **Default:** -1
- **Env:** `LLAMA_ARG_THREADS`

### `-tb, --threads-batch N`

Number of threads used during batch/prompt processing (prefill). Prompt processing is more parallelizable, so this can often be set higher than `--threads`.

- **Default:** same as `--threads`

### `-C, --cpu-mask M`

Hexadecimal CPU affinity mask. Pins threads to specific CPU cores.

### `-Cr, --cpu-range lo-hi`

CPU range for thread affinity (e.g., `0-7`). An alternative to `--cpu-mask`.

### `--cpu-strict <0|1>`

Enforce strict CPU placement. When enabled, threads cannot migrate to other cores.

- **Default:** 0

### `--prio N`

Process/thread scheduling priority.

| Value | Priority |
|-------|----------|
| -1    | Low |
| 0     | Normal (default) |
| 1     | Medium |
| 2     | High |
| 3     | Realtime |

### `--poll <0...100>`

Polling level for waiting on work. Higher values reduce latency but increase CPU usage when idle.

- **Default:** 50

### `-Cb, --cpu-mask-batch M`

CPU affinity mask specifically for batch processing threads.

- **Default:** same as `--cpu-mask`

### `-Crb, --cpu-range-batch lo-hi`

CPU range for batch processing thread affinity.

### `--cpu-strict-batch <0|1>`

Strict CPU placement for batch threads.

- **Default:** same as `--cpu-strict`

### `--prio-batch N`

Scheduling priority for batch processing threads.

- **Default:** 0

### `--poll-batch <0|1>`

Polling mode for batch processing threads.

- **Default:** same as `--poll`

### `-fa, --flash-attn [on|off|auto]`

Flash Attention implementation. Flash Attention is a memory-efficient attention algorithm that reduces peak memory usage and speeds up inference.

- **Default:** auto
- **Env:** `LLAMA_ARG_FLASH_ATTN`
- `auto` enables it when the backend supports it.

### `--repack / --no-repack`

Enable weight repacking for optimized memory layout. Repacking rearranges weight tensors for better cache locality during computation.

- **Default:** enabled
- **Env:** `LLAMA_ARG_REPACK`

### `--perf / --no-perf`

Enable internal libllama performance timing counters. Useful for benchmarking.

- **Default:** false
- **Env:** `LLAMA_ARG_PERF`

### `--numa TYPE`

NUMA (Non-Uniform Memory Access) optimization strategy for multi-socket systems.

| Value     | Behavior |
|-----------|----------|
| `distribute` | Spread execution evenly across all NUMA nodes |
| `isolate`    | Only use CPUs on the node where execution started |
| `numactl`    | Use the CPU map provided by the `numactl` utility |

- **Env:** `LLAMA_ARG_NUMA`
- **Tip:** Drop the system page cache before using this if you haven't used it before.

---

## RoPE / Context Scaling

RoPE (Rotary Position Embedding) scaling allows extending the effective context window beyond the model's original training length. These params are typically set automatically from model metadata but can be overridden.

### `--rope-scaling {none,linear,yarn}`

Scaling method for RoPE frequencies.

| Value    | Behavior |
|----------|----------|
| `none`   | No scaling applied |
| `linear` | Linear interpolation (simple, default if not in model metadata) |
| `yarn`   | YaRN (Yet another RoPE extensioN) — more sophisticated scaling with better quality at large extensions |

- **Env:** `LLAMA_ARG_ROPE_SCALING_TYPE`

### `--rope-scale N`

RoPE context scaling factor. Expands the effective context by a factor of N. For example, `--rope-scale 2` doubles the context length.

- **Env:** `LLAMA_ARG_ROPE_SCALE`

### `--rope-freq-base N`

Base frequency for RoPE embeddings. Used by NTK-aware scaling. Higher values compress position embeddings, effectively extending context.

- **Default:** loaded from model
- **Env:** `LLAMA_ARG_ROPE_FREQ_BASE`

### `--rope-freq-scale N`

RoPE frequency scaling factor. Expands context by a factor of `1/N`.

- **Env:** `LLAMA_ARG_ROPE_FREQ_SCALE`

### `--yarn-orig-ctx N`

Original context size the model was trained with. YaRN uses this as the reference point for scaling.

- **Default:** 0 (use model's training context size)
- **Env:** `LLAMA_ARG_YARN_ORIG_CTX`

### `--yarn-ext-factor N`

YaRN extrapolation mix factor. Controls the blend between interpolation and extrapolation. `0.0` means full interpolation.

- **Default:** -1.0 (auto)
- **Env:** `LLAMA_ARG_YARN_EXT_FACTOR`

### `--yarn-attn-factor N`

YaRN attention magnitude scaling factor.

- **Default:** -1.0 (auto)
- **Env:** `LLAMA_ARG_YARN_ATTN_FACTOR`

### `--yarn-beta-slow N`

YaRN high correction dimension (alpha parameter).

- **Default:** -1.0 (auto)
- **Env:** `LLAMA_ARG_YARN_BETA_SLOW`

### `--yarn-beta-fast N`

YaRN low correction dimension (beta parameter).

- **Default:** -1.0 (auto)
- **Env:** `LLAMA_ARG_YARN_BETA_FAST`

---

## Memory Management

### `--mlock`

Force the operating system to keep the entire model locked in physical RAM, preventing it from being swapped to disk or compressed. Ensures consistent performance but requires enough physical memory.

- **Env:** `LLAMA_ARG_MLOCK`

### `--mmap / --no-mmap`

Memory-map the model file. When enabled (default), the OS maps the file into virtual memory and pages it in on demand, which speeds up loading. When disabled, the model is read into memory conventionally — slower load, but may reduce page-outs if `--mlock` is not used.

- **Default:** enabled
- **Env:** `LLAMA_ARG_MMAP`

### `-dio, --direct-io / --no-direct-io`

Use Direct I/O (bypassing the OS page cache) when loading the model. Can be faster for very large models when you don't want to pollute the page cache.

- **Default:** disabled
- **Env:** `LLAMA_ARG_DIO`

### `--no-host`

Bypass host memory buffer. Allows extra device buffers to be used.

- **Env:** `LLAMA_ARG_NO_HOST`

---

## Sampling Parameters

Sampling controls how the next token is selected from the model's probability distribution. Parameters are applied in order as specified by `--samplers`.

### `--samplers SAMPLERS`

Ordered list of samplers to apply, separated by `;`. The order matters: each sampler filters the candidate tokens before passing to the next.

- **Default:** `penalties;dry;top_n_sigma;top_k;typ_p;top_p;min_p;xtc;temperature`

### `--sampler-seq SEQUENCE`

Simplified sampler sequence using single-character codes:

| Code | Sampler |
|------|---------|
| `e`  | penalties |
| `d`  | dry |
| `s`  | top_n_sigma |
| `k`  | top_k |
| `y`  | typ_p |
| `p`  | top_p |
| `m`  | min_p |
| `x`  | xtc |
| `t`  | temperature |

- **Default:** `edskypmxt`

### `-s, --seed SEED`

Random number generator seed. Use `-1` for a random seed.

- **Default:** -1

### `--temp, --temperature N`

Controls randomness. Lower values make output more deterministic (greedy at 0), higher values increase diversity. Technically scales the logits before softmax.

- **Default:** 0.80

### `--top-k N`

Retain only the top K most probable tokens before sampling. Limits the vocabulary to the K most likely candidates.

- **Default:** 40 (0 = disabled)
- **Env:** `LLAMA_ARG_TOP_K`

### `--top-p N`

Nucleus sampling. Retain the smallest set of tokens whose cumulative probability exceeds P. Dynamically adjusts vocabulary size based on the model's confidence.

- **Default:** 0.95 (1.0 = disabled)

### `--min-p N`

Min-p sampling. Discard tokens with probability less than `min_p * max_probability`. Adapts to the model's confidence — keeps fewer tokens when the model is confident, more when uncertain.

- **Default:** 0.05 (0.0 = disabled)

### `--top-nsigma, --top-n-sigma N`

Top-n-sigma sampling. Keeps tokens within N standard deviations of the mean logit.

- **Default:** -1.0 (disabled)

### `--typical, --typical-p N`

Locally typical sampling. Selects tokens that are "typically" surprising given the context — those closest to the expected information content.

- **Default:** 1.0 (disabled)

### `--repeat-last-n N`

Window of recent tokens to scan for repetition penalty. Set to `0` to disable, `-1` to use the full context size.

- **Default:** 64

### `--repeat-penalty N`

Multiplicative penalty applied to tokens that have already appeared in the `repeat-last-n` window. Values > 1.0 penalize repetition, 1.0 = no penalty.

- **Default:** 1.0 (disabled)

### `--presence-penalty N`

Additive penalty applied once when a token has appeared at all in the output. Similar to OpenAI's presence_penalty.

- **Default:** 0.0 (disabled)

### `--frequency-penalty N`

Additive penalty that scales with how many times a token has appeared. Similar to OpenAI's frequency_penalty.

- **Default:** 0.0 (disabled)

### `--dry-multiplier N`

DRY (Don't Repeat Yourself) sampling multiplier. DRY penalizes token sequences that would extend existing repeated patterns.

- **Default:** 0.0 (disabled)

### `--dry-base N`

Base value for DRY penalty calculation.

- **Default:** 1.75

### `--dry-allowed-length N`

Maximum allowed length of repeated sequences before DRY penalty kicks in.

- **Default:** 2

### `--dry-penalty-last-n N`

Lookback window for DRY sampling. `-1` uses the full context.

- **Default:** -1

### `--dry-sequence-breaker STRING`

Characters or tokens that break DRY sequence matching. Multiple can be specified. Default breakers are `\n`, `:`, `"`, `*`. Use `"none"` to disable all breakers.

### `--xtc-probability N`

XTC (eXclude Top Choices) sampling probability. When triggered, removes top candidates to encourage diversity.

- **Default:** 0.0 (disabled)

### `--xtc-threshold N`

XTC threshold. Tokens above this probability may be excluded.

- **Default:** 0.10 (1.0 = disabled)

### `--adaptive-target N`

Adaptive-p sampling. Selects tokens near this probability target.

- **Default:** -1.0 (disabled)
- Valid range: 0.0 to 1.0

### `--adaptive-decay N`

Decay rate for adaptive-p target adaptation. Lower values are more reactive, higher values are more stable.

- **Default:** 0.90
- Valid range: 0.0 to 0.99

### `--dynatemp-range N`

Dynamic temperature range. When set, the temperature varies between `temp - range` and `temp + range` based on token entropy.

- **Default:** 0.0 (disabled)

### `--dynatemp-exp N`

Exponent for dynamic temperature mapping.

- **Default:** 1.0

### `--mirostat N`

Enable Mirostat sampling, which dynamically adjusts the sampling to maintain a target surprise level (entropy). Ignores top-k, top-p, and typical when active.

| Value | Mode |
|-------|------|
| 0     | Disabled (default) |
| 1     | Mirostat 1.0 |
| 2     | Mirostat 2.0 |

### `--mirostat-lr N`

Mirostat learning rate (eta). Controls how quickly the algorithm adapts.

- **Default:** 0.10

### `--mirostat-ent N`

Mirostat target entropy (tau). Lower values produce more focused output, higher values more diverse.

- **Default:** 5.0

### `-l, --logit-bias TOKEN_ID(+/-)BIAS`

Manually adjust the likelihood of specific tokens. Positive bias increases probability, negative decreases it.

```bash
# Make token 15043 (' Hello') more likely
--logit-bias 15043+1

# Make it less likely
--logit-bias 15043-1
```

### `--ignore-eos`

Ignore the end-of-sequence token and continue generating. Equivalent to `--logit-bias EOS-inf`.

### `-bs, --backend-sampling`

Enable experimental backend-side sampling. Moves sampling computation to the GPU.

- **Default:** disabled
- **Env:** `LLAMA_ARG_BACKEND_SAMPLING`

---

## Grammar and Constrained Output

### `--grammar GRAMMAR`

BNF-like grammar string to constrain generation. Forces the model to only produce tokens that conform to the grammar rules.

```bash
--grammar 'root ::= "yes" | "no"'
```

### `--grammar-file FNAME`

Load grammar rules from a file.

### `-j, --json-schema SCHEMA`

JSON Schema to constrain output. The model will only generate valid JSON conforming to the schema. Use `{}` for any valid JSON object.

```bash
--json-schema '{"type": "object", "properties": {"name": {"type": "string"}}}'
```

### `-jf, --json-schema-file FILE`

Load JSON schema from a file.

---

## LoRA and Control Vectors

### `--lora FNAME`

Load a LoRA (Low-Rank Adaptation) adapter file. Multiple adapters can be loaded using comma-separated paths. LoRA adapters modify model behavior without changing the base weights.

```bash
--lora adapter1.gguf,adapter2.gguf
```

### `--lora-scaled FNAME:SCALE,...`

Load LoRA adapters with custom scaling factors. Format: `path:scale`. A scale of `1.0` applies the adapter at full strength, `0.5` at half strength.

```bash
--lora-scaled adapter.gguf:0.5
```

### `--lora-init-without-apply`

Load LoRA adapters at startup but don't apply them. They can be activated later via the `POST /lora-adapters` API endpoint. Useful for dynamically switching adapters at runtime.

### `--control-vector FNAME`

Add a control vector file to steer model behavior along interpretable dimensions (e.g., formality, creativity).

### `--control-vector-scaled FNAME:SCALE,...`

Add a control vector with a custom scaling factor.

### `--control-vector-layer-range START END`

Restrict control vector application to a specific layer range (inclusive).

---

## Speculative Decoding

Speculative decoding uses a smaller "draft" model to predict multiple tokens at once, which the main model then verifies. This can significantly speed up generation.

### `-md, --model-draft FNAME`

Path to the draft model for speculative decoding.

- **Env:** `LLAMA_ARG_MODEL_DRAFT`

### `-hfd, -hfrd, --hf-repo-draft REPO[:QUANT]`

Hugging Face repo for the draft model.

- **Env:** `LLAMA_ARG_HFD_REPO`

### `--draft, --draft-max N`

Maximum number of tokens to draft per speculative step.

- **Default:** 16
- **Env:** `LLAMA_ARG_DRAFT_MAX`

### `--draft-min N`

Minimum number of draft tokens per step.

- **Default:** 0
- **Env:** `LLAMA_ARG_DRAFT_MIN`

### `--draft-p-min P`

Minimum acceptance probability for greedy speculative decoding.

- **Default:** 0.75
- **Env:** `LLAMA_ARG_DRAFT_P_MIN`

### `-cd, --ctx-size-draft N`

Context size for the draft model.

- **Default:** 0 (loaded from model)
- **Env:** `LLAMA_ARG_CTX_SIZE_DRAFT`

### `-ngld, --gpu-layers-draft N`

GPU layers for the draft model. Accepts a number, `auto`, or `all`.

- **Default:** auto
- **Env:** `LLAMA_ARG_N_GPU_LAYERS_DRAFT`

### `-devd, --device-draft DEVICES`

Device list for draft model offloading.

### `-ctkd, --cache-type-k-draft TYPE`

KV cache key type for the draft model.

- **Default:** f16
- **Env:** `LLAMA_ARG_CACHE_TYPE_K_DRAFT`

### `-ctvd, --cache-type-v-draft TYPE`

KV cache value type for the draft model.

- **Default:** f16
- **Env:** `LLAMA_ARG_CACHE_TYPE_V_DRAFT`

### `-td, --threads-draft N`

CPU threads for draft model generation.

- **Default:** same as `--threads`

### `-tbd, --threads-batch-draft N`

CPU threads for draft model batch processing.

- **Default:** same as `--threads-draft`

### `--spec-replace TARGET DRAFT`

String translation table for making incompatible draft/main model tokenizers work together.

### `--spec-type TYPE`

N-gram-based speculative decoding without a draft model.

| Value | Method |
|-------|--------|
| `none` | Disabled (default) |
| `ngram-cache` | N-gram cache |
| `ngram-simple` | Simple n-gram matching |
| `ngram-map-k` | N-gram map with key |
| `ngram-map-k4v` | N-gram map with key (4-value) |
| `ngram-mod` | Modified n-gram |

- **Env:** `LLAMA_ARG_SPEC_TYPE`

### `--spec-ngram-size-n N`

Length of the lookup n-gram for speculative decoding.

- **Default:** 12

### `--spec-ngram-size-m N`

Length of the draft m-gram for speculative decoding.

- **Default:** 48

### `--spec-ngram-min-hits N`

Minimum n-gram match hits before using a draft.

- **Default:** 1

---

## Server Configuration

These parameters are specific to `llama-server`.

### `--host HOST`

IP address or hostname to bind the server to. Use `0.0.0.0` to listen on all interfaces. Paths ending in `.sock` bind to a Unix domain socket.

- **Default:** 127.0.0.1
- **Env:** `LLAMA_ARG_HOST`

### `--port PORT`

TCP port number.

- **Default:** 8080
- **Env:** `LLAMA_ARG_PORT`

### `--path PATH`

Directory path to serve static files from (for custom web UIs).

- **Env:** `LLAMA_ARG_STATIC_PATH`

### `--api-prefix PREFIX`

URL prefix for all API endpoints (e.g., `/v1`). No trailing slash.

- **Env:** `LLAMA_ARG_API_PREFIX`

### `--api-key KEY`

API key for authentication. Multiple keys can be comma-separated.

- **Env:** `LLAMA_API_KEY`

### `--api-key-file FNAME`

File containing API keys (one per line).

### `--ssl-key-file FNAME`

PEM-encoded SSL private key for HTTPS.

- **Env:** `LLAMA_ARG_SSL_KEY_FILE`

### `--ssl-cert-file FNAME`

PEM-encoded SSL certificate for HTTPS.

- **Env:** `LLAMA_ARG_SSL_CERT_FILE`

### `-to, --timeout N`

Server read/write timeout in seconds.

- **Default:** 600
- **Env:** `LLAMA_ARG_TIMEOUT`

### `--threads-http N`

Number of threads for processing HTTP requests. Separate from inference threads.

- **Default:** -1 (auto)
- **Env:** `LLAMA_ARG_THREADS_HTTP`

### `-np, --parallel N`

Number of parallel inference slots. Each slot can handle one concurrent request. More slots allow more simultaneous users but divide the KV cache.

- **Default:** -1 (auto)
- **Env:** `LLAMA_ARG_N_PARALLEL`

### `-cb, --cont-batching / --no-cont-batching`

Continuous batching (dynamic batching). When enabled, the server can interleave token generation across multiple slots in a single batch, improving throughput.

- **Default:** enabled
- **Env:** `LLAMA_ARG_CONT_BATCHING`

### `--cache-prompt / --no-cache-prompt`

Cache and reuse processed prompts. When a new request shares a prefix with a previous one, the cached KV state is reused, skipping re-evaluation.

- **Default:** enabled
- **Env:** `LLAMA_ARG_CACHE_PROMPT`

### `--cache-reuse N`

Minimum chunk size for KV cache reuse via shifting. Requires prompt caching to be enabled.

- **Default:** 0
- **Env:** `LLAMA_ARG_CACHE_REUSE`

### `-cram, --cache-ram N`

Maximum model cache size in MiB. Controls how much memory is used for caching downloaded models.

- **Default:** 8192 (-1 = no limit, 0 = disable)
- **Env:** `LLAMA_ARG_CACHE_RAM`

### `--context-shift / --no-context-shift`

Automatically shift the context window when it fills up during infinite text generation. Drops the oldest tokens and keeps generating.

- **Default:** disabled
- **Env:** `LLAMA_ARG_CONTEXT_SHIFT`

### `-ctxcp, --ctx-checkpoints N`

Maximum number of context checkpoints per slot. Checkpoints enable faster context rewind and recovery.

- **Default:** 32
- **Env:** `LLAMA_ARG_CTX_CHECKPOINTS`

### `-cpent, --checkpoint-every-n-tokens N`

Create a KV cache checkpoint every N tokens during prefill. Set to `-1` to disable.

- **Default:** 8192
- **Env:** `LLAMA_ARG_CHECKPOINT_EVERY_NT`

### `--metrics`

Enable Prometheus-compatible metrics endpoint at `/metrics`.

- **Env:** `LLAMA_ARG_ENDPOINT_METRICS`

### `--props`

Enable the `POST /props` endpoint for changing global server properties at runtime.

- **Env:** `LLAMA_ARG_ENDPOINT_PROPS`

### `--slots / --no-slots`

Expose the `/slots` monitoring endpoint.

- **Default:** enabled
- **Env:** `LLAMA_ARG_ENDPOINT_SLOTS`

### `--slot-save-path PATH`

Directory for saving/loading slot KV cache state to disk.

### `--media-path PATH`

Directory for local media files accessible via `file://` URLs in multimodal requests.

### `-a, --alias STRING`

Model name alias(es) exposed via the API, comma-separated.

- **Env:** `LLAMA_ARG_ALIAS`

### `--tags STRING`

Informational model tags (not used for routing).

- **Env:** `LLAMA_ARG_TAGS`

### `-sps, --slot-prompt-similarity SIMILARITY`

Minimum prompt similarity score required to reuse an existing slot.

- **Default:** 0.10 (0.0 = disabled)

### `--warmup / --no-warmup`

Perform a warmup inference run with an empty prompt at startup. Ensures GPU kernels are compiled and caches are warm before serving real requests.

- **Default:** enabled

### `--sleep-idle-seconds N`

Number of idle seconds before the server enters sleep mode (unloads model from memory). Set to `-1` to disable.

- **Default:** -1

### `--models-dir PATH`

Directory containing models for the router server mode.

- **Env:** `LLAMA_ARG_MODELS_DIR`

### `--models-preset PATH`

INI file with model presets for the router server.

- **Env:** `LLAMA_ARG_MODELS_PRESET`

### `--models-max N`

Maximum number of models loaded simultaneously in router mode.

- **Default:** 4 (0 = unlimited)
- **Env:** `LLAMA_ARG_MODELS_MAX`

### `--models-autoload / --no-models-autoload`

Auto-load models in router mode.

- **Default:** enabled
- **Env:** `LLAMA_ARG_MODELS_AUTOLOAD`

---

## Chat and Templating

### `--jinja / --no-jinja`

Use the Jinja2 template engine for chat formatting. Required for custom templates and advanced tool-use features.

- **Default:** enabled
- **Env:** `LLAMA_ARG_JINJA`

### `--chat-template TEMPLATE`

Set a custom Jinja chat template string, or use a built-in template by name. Overrides the template embedded in the model file.

Built-in templates include: `chatml`, `llama2`, `llama3`, `llama4`, `mistral-v1`, `mistral-v3`, `mistral-v7`, `deepseek`, `deepseek2`, `deepseek3`, `gemma`, `phi3`, `phi4`, `command-r`, `chatglm3`, `chatglm4`, `vicuna`, `zephyr`, and many more.

- **Env:** `LLAMA_ARG_CHAT_TEMPLATE`

### `--chat-template-file PATH`

Load a Jinja chat template from a file.

- **Env:** `LLAMA_ARG_CHAT_TEMPLATE_FILE`

### `--chat-template-kwargs JSON`

Additional parameters passed to the Jinja template parser as a JSON object.

- **Env:** `LLAMA_CHAT_TEMPLATE_KWARGS`

### `--skip-chat-parsing / --no-skip-chat-parsing`

Force raw content parsing even when a Jinja template is set. All model output goes into `message.content` with no parsing of reasoning tags or tool calls.

- **Default:** disabled
- **Env:** `LLAMA_ARG_SKIP_CHAT_PARSING`

### `--prefill-assistant / --no-prefill-assistant`

When the last message in a conversation is from the assistant, prepend it as a generation prefix (similar to Claude's API). Disabling this treats it as a complete message.

- **Default:** enabled
- **Env:** `LLAMA_ARG_PREFILL_ASSISTANT`

### `-sp, --special`

Output special tokens (like BOS, EOS) in the response.

- **Default:** false

### `--spm-infill`

Use Suffix/Prefix/Middle ordering for fill-in-the-middle tasks, instead of the default Prefix/Suffix/Middle.

- **Default:** disabled

---

## Reasoning / Thinking

### `--reasoning-format FORMAT`

Controls how chain-of-thought / thinking content is handled in the output.

| Value | Behavior |
|-------|----------|
| `none` | Leaves thought tags unparsed in `message.content` |
| `deepseek` | Extracts thoughts into `message.reasoning_content` |
| `deepseek-legacy` | Keeps `<think>` tags in `message.content` AND populates `message.reasoning_content` |
| `auto` | Auto-detect from template (default) |

- **Env:** `LLAMA_ARG_THINK`

### `-rea, --reasoning [on|off|auto]`

Enable or disable reasoning/thinking mode in the chat.

- **Default:** auto (detect from template)
- **Env:** `LLAMA_ARG_REASONING`

### `--reasoning-budget N`

Token budget for the thinking/reasoning phase. Limits how many tokens the model can spend "thinking" before producing the final answer.

| Value | Behavior |
|-------|----------|
| -1    | Unrestricted (default) |
| 0     | Immediately end thinking |
| N > 0 | Allow up to N thinking tokens |

- **Env:** `LLAMA_ARG_THINK_BUDGET`

### `--reasoning-budget-message MESSAGE`

Custom message injected before the end-of-thinking tag when the reasoning budget is exhausted. Useful for gently steering the model to wrap up.

- **Env:** `LLAMA_ARG_THINK_BUDGET_MESSAGE`

---

## Multimodal

### `-mm, --mmproj FILE`

Path to a multimodal projector file (vision encoder, etc.). Required for vision-language models. When using `-hf`, this is often downloaded automatically.

- **Env:** `LLAMA_ARG_MMPROJ`

### `-mmu, --mmproj-url URL`

Download the multimodal projector from a URL.

- **Env:** `LLAMA_ARG_MMPROJ_URL`

### `--mmproj-auto / --no-mmproj`

Automatically download/use the multimodal projector when available.

- **Default:** enabled
- **Env:** `LLAMA_ARG_MMPROJ_AUTO`

### `--mmproj-offload / --no-mmproj-offload`

Offload the multimodal projector to GPU.

- **Default:** enabled
- **Env:** `LLAMA_ARG_MMPROJ_OFFLOAD`

### `--image-min-tokens N`

Minimum tokens per image (for variable-resolution vision models).

- **Default:** read from model
- **Env:** `LLAMA_ARG_IMAGE_MIN_TOKENS`

### `--image-max-tokens N`

Maximum tokens per image.

- **Default:** read from model
- **Env:** `LLAMA_ARG_IMAGE_MAX_TOKENS`

---

## Logging

### `-v, --verbose, --log-verbose`

Set verbosity to maximum (log all messages). Useful for debugging.

### `-lv, --verbosity, --log-verbosity N`

Set verbosity threshold. Messages with a higher verbosity level are suppressed.

| Level | Category |
|-------|----------|
| 0     | Generic output |
| 1     | Errors |
| 2     | Warnings |
| 3     | Info (default) |
| 4     | Debug |

- **Env:** `LLAMA_LOG_VERBOSITY`

### `--log-disable`

Disable all logging output.

### `--log-file FNAME`

Write logs to a file instead of stderr.

- **Env:** `LLAMA_LOG_FILE`

### `--log-colors [on|off|auto]`

Colored log output. `auto` enables colors when writing to a terminal.

- **Default:** auto
- **Env:** `LLAMA_LOG_COLORS`

### `--log-prefix`

Add a prefix label to each log message.

- **Env:** `LLAMA_LOG_PREFIX`

### `--log-timestamps`

Add timestamps to each log message.

- **Env:** `LLAMA_LOG_TIMESTAMPS`

---

## Miscellaneous

### `-h, --help, --usage`

Print usage information and exit.

### `--version`

Show version and build info.

### `--license`

Display source code license and dependency information.

### `-cl, --cache-list`

Show list of models in the local cache.

### `--completion-bash`

Print a source-able bash completion script.

### `--verbose-prompt`

Print the full tokenized prompt before generation begins. Helpful for debugging template issues.

- **Default:** false

### `-e, --escape / --no-escape`

Process C-style escape sequences (`\n`, `\r`, `\t`, `\'`, `\"`, `\\`) in prompts.

- **Default:** true

### `-r, --reverse-prompt PROMPT`

Halt generation when this string appears in the output and return control (used in interactive/CLI mode).

### `--pooling {none,mean,cls,last,rank}`

Pooling strategy for embedding models. Uses model default if unspecified.

- **Env:** `LLAMA_ARG_POOLING`

### `--embedding, --embeddings`

Restrict the server to embedding-only mode.

- **Env:** `LLAMA_ARG_EMBEDDINGS`

### `--rerank, --reranking`

Enable the reranking API endpoint.

- **Env:** `LLAMA_ARG_RERANKING`

### `--offline`

Offline mode. Forces use of local cache and prevents all network access.

- **Env:** `LLAMA_OFFLINE`

### `-lcs, --lookup-cache-static FNAME`

Path to a static lookup cache for lookup-based speculative decoding (not updated by generation).

### `-lcd, --lookup-cache-dynamic FNAME`

Path to a dynamic lookup cache for lookup-based speculative decoding (updated during generation).

### `-mv, --model-vocoder FNAME`

Vocoder model for TTS (text-to-speech) audio generation.

### `--tts-use-guide-tokens`

Use guide tokens for improved word recall in TTS.

---

## Parameters Used by This Calculator

This calculator generates a `llama-server` command using these specific parameters:

| Parameter | Calculator Field | Description |
|-----------|-----------------|-------------|
| `-m` | Model file | Path to the GGUF model file |
| `-ngl` | GPU Layers slider | Layers offloaded to GPU |
| `-c` | Max Context (computed) | Context window size in tokens |
| `--cache-type-k` | KV Cache Quantization | Key cache data type |
| `--cache-type-v` | KV Cache Quantization | Value cache data type |

Example generated command:

```bash
llama-server -m model.gguf \
  -ngl 33 \
  -c 32768 \
  --cache-type-k q4_0 \
  --cache-type-v q4_0
```

---

## Environment Variables

All parameters that support environment variables follow this pattern:

- When both a CLI argument and an env var are set, the **CLI argument takes precedence**.
- Boolean env vars accept: `true` / `1` / `on` / `enabled` to enable, and `false` / `0` / `off` / `disabled` to disable.
- Prefixing with `LLAMA_ARG_NO_` disables a boolean option regardless of value (e.g., `LLAMA_ARG_NO_MMAP`).

Docker Compose example:

```yaml
services:
  llamacpp-server:
    image: ghcr.io/ggml-org/llama.cpp:server
    ports:
      - 8080:8080
    volumes:
      - ./models:/models
    environment:
      LLAMA_ARG_MODEL: /models/my_model.gguf
      LLAMA_ARG_CTX_SIZE: 4096
      LLAMA_ARG_N_PARALLEL: 2
      LLAMA_ARG_ENDPOINT_METRICS: 1
      LLAMA_ARG_PORT: 8080
```
