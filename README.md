# Local LLM Context Size Calculator

Estimate the maximum context window you can run locally with `llama.cpp` based on available VRAM, system RAM, model size, layer offloading, KV cache quantization, and MoE offload settings.

Live app: [amitmaity.github.io/local-llm-calculator](https://amitmaity.github.io/local-llm-calculator)

## Features
- Calculates maximum usable context size from VRAM and RAM constraints.
- Supports dense, MoE, hybrid-dense, hybrid-MoE, sliding-dense, and sliding-MoE model layouts.
- Includes presets for Llama, Qwen 2.5, Qwen3, Qwen 3.5, Qwen 3.6, Qwen3-Coder-Next, Mixtral, Gemma 2, Gemma 3, Gemma 4, Muse Glimmer, Ministral 3, Falcon 3, Nanbeige4.1, OLMo 2, LFM2, TwiL-LM, Granite SWASH, SmolLM, TinyLlama, Phi-3, Phi-4, Command-R, and DeepSeek.
- Handles `llama.cpp` style GPU layer offloading with `-ngl`.
- Supports KV cache quant types: `f16`, `f32`, `q8_0`, `q4_0`, `q4_1`, `q5_0`, `q5_1`.
- Includes reverse mode to estimate memory usage for a target context size.
- Generates a suggested `llama-server` command.

## Model Inputs
The calculator lets you configure:
- VRAM and system RAM
- Total model params
- Model type
- Model file size in GB
- Number of layers
- KV heads and head dimension
- GPU layers
- KV cache quantization
- MoE expert count and expert offload
- Hybrid attention layout for Qwen-style models
- Sliding/global attention layout for Gemma-style models

## Notes
- Results are estimates, not exact runtime guarantees.
- Actual memory usage depends on `llama.cpp` version, runtime buffers, batch size, and system configuration.
- For hybrid Qwen-style models, the calculator assumes an evenly spaced full-attention pattern and also includes fixed linear-attention state overhead.
- For Gemma-style sliding/global models, sliding-window KV memory is capped by the configured window while global attention layers scale with full context length.
- File size is the primary weight-memory driver; parameter count is shown as helpful metadata.

## Run Locally
No build step is required.

1. Clone or download this repository.
2. Open `index.html` in your browser.

## Project Structure
```text
index.html   UI structure
style.css    App styling
app.js       Presets, calculations, and UI logic
```

## Deployment
Hosted at: [https://amitmaity.github.io/local-llm-calculator](https://amitmaity.github.io/local-llm-calculator)
