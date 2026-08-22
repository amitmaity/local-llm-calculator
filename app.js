'use strict';

// ============================================================
// Constants
// ============================================================

const KV_QUANT_BYTES = {
  f32:  4.0,
  f16:  2.0,
  q8_0: 1.0625,   // 34 / 32
  q4_0: 0.5625,   // 18 / 32
  q4_1: 0.625,    // 20 / 32
  q5_0: 0.6875,   // 22 / 32
  q5_1: 0.75,     // 24 / 32
};

const GB = 1024 * 1024 * 1024;

// ============================================================
// Model Presets
// ============================================================

const MODEL_PRESETS = [
  {
    group: 'Dense',
    models: [
      { id: 'llama31-8b',    name: 'Llama 3.1 8B',        params: 8,    type: 'dense', layers: 32,  kvHeads: 8,  headDim: 128, defaultSize: 4.9  },
      { id: 'llama31-70b',   name: 'Llama 3.1 70B',       params: 70,   type: 'dense', layers: 80,  kvHeads: 8,  headDim: 128, defaultSize: 40   },
      { id: 'llama31-405b',  name: 'Llama 3.1 405B',      params: 405,  type: 'dense', layers: 126, kvHeads: 8,  headDim: 128, defaultSize: 228  },
      { id: 'llama32-1b',    name: 'Llama 3.2 1B',        params: 1,    type: 'dense', layers: 16,  kvHeads: 8,  headDim: 64,  defaultSize: 0.7  },
      { id: 'llama32-3b',    name: 'Llama 3.2 3B',        params: 3,    type: 'dense', layers: 28,  kvHeads: 8,  headDim: 128, defaultSize: 2.0  },
      { id: 'tinyllama-1.1b', name: 'TinyLlama 1.1B',     params: 1.1,  type: 'dense', layers: 22,  kvHeads: 4,  headDim: 64,  defaultSize: 0.7  },
      { id: 'smollm2-360m',  name: 'SmolLM2 360M',        params: 0.36, type: 'dense', layers: 32,  kvHeads: 5,  headDim: 64,  defaultSize: 0.25 },
      { id: 'smollm2-1.7b',  name: 'SmolLM2 1.7B',        params: 1.7,  type: 'dense', layers: 24,  kvHeads: 32, headDim: 64,  defaultSize: 1.0  },
      { id: 'smollm3-3b',    name: 'SmolLM3 3B',          params: 3,    type: 'dense', layers: 36,  kvHeads: 4,  headDim: 128, defaultSize: 1.8  },
      { id: 'qwen25-0.5b',   name: 'Qwen 2.5 0.5B',       params: 0.5,  type: 'dense', layers: 24,  kvHeads: 2,  headDim: 64,  defaultSize: 0.4  },
      { id: 'qwen25-1.5b',   name: 'Qwen 2.5 1.5B',       params: 1.5,  type: 'dense', layers: 28,  kvHeads: 2,  headDim: 128, defaultSize: 1.0  },
      { id: 'qwen25-3b',     name: 'Qwen 2.5 3B',         params: 3,    type: 'dense', layers: 36,  kvHeads: 2,  headDim: 128, defaultSize: 2.0  },
      { id: 'qwen25-7b',     name: 'Qwen 2.5 7B',         params: 7,    type: 'dense', layers: 28,  kvHeads: 4,  headDim: 128, defaultSize: 4.4  },
      { id: 'qwen3-0.6b',    name: 'Qwen3 0.6B',          params: 0.6,  type: 'dense', layers: 28,  kvHeads: 8,  headDim: 128, defaultSize: 0.5  },
      { id: 'qwen3-1.7b',    name: 'Qwen3 1.7B',          params: 1.7,  type: 'dense', layers: 28,  kvHeads: 8,  headDim: 128, defaultSize: 1.2  },
      { id: 'qwen3-4b',      name: 'Qwen3 4B',            params: 4,    type: 'dense', layers: 36,  kvHeads: 8,  headDim: 128, defaultSize: 2.8  },
      { id: 'qwen3-8b',      name: 'Qwen3 8B',            params: 8,    type: 'dense', layers: 36,  kvHeads: 8,  headDim: 128, defaultSize: 5.0  },
      { id: 'qwen25-14b',    name: 'Qwen 2.5 14B',        params: 14,   type: 'dense', layers: 48,  kvHeads: 8,  headDim: 128, defaultSize: 8.3  },
      { id: 'qwen25-32b',    name: 'Qwen 2.5 32B',        params: 32,   type: 'dense', layers: 64,  kvHeads: 8,  headDim: 128, defaultSize: 18.5 },
      { id: 'qwen25-72b',    name: 'Qwen 2.5 72B',        params: 72,   type: 'dense', layers: 80,  kvHeads: 8,  headDim: 128, defaultSize: 41   },
      { id: 'mistral-7b',    name: 'Mistral 7B v0.3',     params: 7.3,  type: 'dense', layers: 32,  kvHeads: 8,  headDim: 128, defaultSize: 4.4  },
      { id: 'mistral-24b',   name: 'Mistral Small 24B',   params: 24,   type: 'dense', layers: 40,  kvHeads: 8,  headDim: 128, defaultSize: 14   },
      { id: 'gemma2-9b',     name: 'Gemma 2 9B',          params: 9,    type: 'dense', layers: 42,  kvHeads: 4,  headDim: 256, defaultSize: 5.5  },
      { id: 'gemma2-27b',    name: 'Gemma 2 27B',         params: 27,   type: 'dense', layers: 46,  kvHeads: 16, headDim: 128, defaultSize: 15.5 },
      { id: 'phi3-mini',     name: 'Phi-3 Mini 3.8B',     params: 3.8,  type: 'dense', layers: 32,  kvHeads: 8,  headDim: 96,  defaultSize: 2.3  },
      { id: 'phi4-mini',     name: 'Phi-4 Mini 3.8B',     params: 3.8,  type: 'dense', layers: 32,  kvHeads: 8,  headDim: 128, defaultSize: 2.5  },
      { id: 'phi3-medium',   name: 'Phi-3 Medium 14B',    params: 14,   type: 'dense', layers: 40,  kvHeads: 8,  headDim: 128, defaultSize: 8.3  },
      { id: 'commandr-35b',  name: 'Command-R 35B',       params: 35,   type: 'dense', layers: 40,  kvHeads: 8,  headDim: 128, defaultSize: 20   },
      { id: 'falcon3-1b',    name: 'Falcon 3 1B',         params: 1,    type: 'dense', layers: 18,  kvHeads: 4,  headDim: 256, defaultSize: 1.1  },
      { id: 'falcon3-3b',    name: 'Falcon 3 3B',         params: 3,    type: 'dense', layers: 22,  kvHeads: 4,  headDim: 256, defaultSize: 2.0  },
      { id: 'falcon3-7b',    name: 'Falcon 3 7B',         params: 7,    type: 'dense', layers: 28,  kvHeads: 4,  headDim: 256, defaultSize: 4.4  },
      { id: 'falcon3-10b',   name: 'Falcon 3 10B',        params: 10,   type: 'dense', layers: 40,  kvHeads: 4,  headDim: 256, defaultSize: 6.0  },
      { id: 'nanbeige4.1-3b', name: 'Nanbeige4.1 3B',     params: 3,    type: 'dense', layers: 32,  kvHeads: 4,  headDim: 128, defaultSize: 2.3  },
      { id: 'olmo2-1b',      name: 'OLMo 2 1B',           params: 1,    type: 'dense', layers: 16,  kvHeads: 16, headDim: 128, defaultSize: 0.9  },
      { id: 'olmo2-7b',      name: 'OLMo 2 7B',           params: 7,    type: 'dense', layers: 32,  kvHeads: 32, headDim: 128, defaultSize: 4.4  },
      { id: 'lfm2-1.2b',     name: 'LFM2 1.2B',           params: 1.2,  type: 'dense', layers: 16,  kvHeads: 8,  headDim: 64,  defaultSize: 0.73 },
      { id: 'lfm2-2.6b',     name: 'LFM2 2.6B',           params: 2.6,  type: 'dense', layers: 30,  kvHeads: 8,  headDim: 64,  defaultSize: 1.7  },
      { id: 'twil-lm1.7b',   name: 'TwiL-LM 1.7B',        params: 1.7,  type: 'dense', layers: 24,  kvHeads: 32, headDim: 64,  defaultSize: 1.0  },
      { id: 'twil-lm3b',     name: 'TwiL-LM 3B',          params: 3,    type: 'dense', layers: 36,  kvHeads: 4,  headDim: 128, defaultSize: 2.0  },
    ],
  },
  {
    group: 'MoE',
    models: [
      { id: 'mixtral-8x7b',  name: 'Mixtral 8x7B',       params: 46.7, type: 'moe', layers: 32, kvHeads: 8, headDim: 128, defaultSize: 26, totalExperts: 8,   activeExperts: 2, attnFraction: 0.045 },
      { id: 'mixtral-8x22b', name: 'Mixtral 8x22B',      params: 141,  type: 'moe', layers: 56, kvHeads: 8, headDim: 128, defaultSize: 80, totalExperts: 8,   activeExperts: 2, attnFraction: 0.04  },
      { id: 'deepseek-v3',   name: 'DeepSeek-V3 671B',   params: 671,  type: 'moe', layers: 61, kvHeads: 8, headDim: 128, defaultSize: 377, totalExperts: 256, activeExperts: 8, attnFraction: 0.03  },
      { id: 'deepseek-r1',   name: 'DeepSeek-R1 671B',   params: 671,  type: 'moe', layers: 61, kvHeads: 8, headDim: 128, defaultSize: 377, totalExperts: 256, activeExperts: 8, attnFraction: 0.03  },
      { id: 'qwen25-moe',    name: 'Qwen 2.5-MoE A2.7B', params: 14.3, type: 'moe', layers: 24, kvHeads: 4, headDim: 128, defaultSize: 8.5, totalExperts: 60,  activeExperts: 4, attnFraction: 0.06  },
    ],
  },
  {
    group: 'Hybrid Dense (Qwen 3.5/3.6/3.8)',
    models: [
      { id: 'qwen35-0.8b', name: 'Qwen3.5 0.8B',  params: 0.8, type: 'hybrid-dense', layers: 24, kvHeads: 2, headDim: 256, defaultSize: 0.6,  fullAttnInterval: 4, linearKeyDim: 128, linearValueHeads: 16, linearValueDim: 128 },
      { id: 'qwen35-2b',   name: 'Qwen3.5 2B',    params: 2,   type: 'hybrid-dense', layers: 24, kvHeads: 2, headDim: 256, defaultSize: 1.5,  fullAttnInterval: 4, linearKeyDim: 128, linearValueHeads: 16, linearValueDim: 128 },
      { id: 'qwen35-4b',   name: 'Qwen3.5 4B',    params: 4,   type: 'hybrid-dense', layers: 32, kvHeads: 4, headDim: 256, defaultSize: 2.8,  fullAttnInterval: 4, linearKeyDim: 128, linearValueHeads: 32, linearValueDim: 128 },
      { id: 'qwen35-9b',   name: 'Qwen3.5 9B',    params: 9,   type: 'hybrid-dense', layers: 32, kvHeads: 4, headDim: 256, defaultSize: 5.5,  fullAttnInterval: 4, linearKeyDim: 128, linearValueHeads: 32, linearValueDim: 128 },
      { id: 'qwen35-27b',  name: 'Qwen3.5 27B',   params: 27,  type: 'hybrid-dense', layers: 64, kvHeads: 4, headDim: 256, defaultSize: 16,   fullAttnInterval: 4, linearKeyDim: 128, linearValueHeads: 48, linearValueDim: 128 },
      { id: 'qwen36-27b',  name: 'Qwen3.6 27B',   params: 27,  type: 'hybrid-dense', layers: 64, kvHeads: 4, headDim: 256, defaultSize: 16,   fullAttnInterval: 4, linearKeyDim: 128, linearValueHeads: 48, linearValueDim: 128 },
      { id: 'qwen38-27b',  name: 'Qwen3.8 27B',   params: 27,  type: 'hybrid-dense', layers: 64, kvHeads: 4, headDim: 256, defaultSize: 16,   fullAttnInterval: 4, linearKeyDim: 128, linearValueHeads: 48, linearValueDim: 128 },
    ],
  },
  {
    group: 'Hybrid MoE',
    models: [
      { id: 'qwen35-35b-a3b',   name: 'Qwen3.5 35B-A3B',          params: 35,  type: 'hybrid-moe', layers: 40, kvHeads: 2, headDim: 256, defaultSize: 20,  fullAttnInterval: 4, linearKeyDim: 128, linearValueHeads: 32, linearValueDim: 128, totalExperts: 256, activeExperts: 9,  attnFraction: 0.04 },
      { id: 'qwen36-35b-a3b',   name: 'Qwen3.6 35B-A3B',          params: 35,  type: 'hybrid-moe', layers: 40, kvHeads: 2, headDim: 256, defaultSize: 20,  fullAttnInterval: 4, linearKeyDim: 128, linearValueHeads: 32, linearValueDim: 128, totalExperts: 256, activeExperts: 9,  attnFraction: 0.04 },
      { id: 'qwen35-122b-a10b', name: 'Qwen3.5 122B-A10B',        params: 122, type: 'hybrid-moe', layers: 48, kvHeads: 2, headDim: 256, defaultSize: 70,  fullAttnInterval: 4, linearKeyDim: 128, linearValueHeads: 64, linearValueDim: 128, totalExperts: 256, activeExperts: 9,  attnFraction: 0.04 },
      { id: 'qwen35-397b-a17b', name: 'Qwen3.5 397B-A17B',        params: 397, type: 'hybrid-moe', layers: 60, kvHeads: 2, headDim: 256, defaultSize: 225, fullAttnInterval: 4, linearKeyDim: 128, linearValueHeads: 64, linearValueDim: 128, totalExperts: 512, activeExperts: 11, attnFraction: 0.03 },
      { id: 'qwen3-coder-next', name: 'Qwen3-Coder-Next 80B-A3B', params: 80,  type: 'hybrid-moe', layers: 48, kvHeads: 2, headDim: 256, defaultSize: 46,  fullAttnInterval: 4, linearKeyDim: 128, linearValueHeads: 32, linearValueDim: 128, totalExperts: 512, activeExperts: 10, attnFraction: 0.04 },
    ],
  },
  {
    group: 'Gemma 3',
    models: [
      { id: 'gemma3-270m', name: 'Gemma 3 270M', params: 0.27, type: 'sliding-dense', layers: 18, kvHeads: 1, headDim: 256, defaultSize: 0.2,  slidingWindow: 512,  globalAttnInterval: 6, slidingKvHeads: 1, slidingHeadDim: 256, globalKvHeads: 1, globalHeadDim: 256, kvSharedLayers: 0 },
      { id: 'gemma3-1b',   name: 'Gemma 3 1B',   params: 1,    type: 'sliding-dense', layers: 26, kvHeads: 1, headDim: 256, defaultSize: 0.8,  slidingWindow: 512,  globalAttnInterval: 6, slidingKvHeads: 1, slidingHeadDim: 256, globalKvHeads: 1, globalHeadDim: 256, kvSharedLayers: 0 },
      { id: 'gemma3-4b',   name: 'Gemma 3 4B',   params: 4,    type: 'sliding-dense', layers: 34, kvHeads: 4, headDim: 256, defaultSize: 2.5,  slidingWindow: 1024, globalAttnInterval: 6, slidingKvHeads: 4, slidingHeadDim: 256, globalKvHeads: 4, globalHeadDim: 256, kvSharedLayers: 0 },
      { id: 'gemma3-12b',  name: 'Gemma 3 12B',  params: 12,   type: 'sliding-dense', layers: 48, kvHeads: 8, headDim: 256, defaultSize: 7.5,  slidingWindow: 1024, globalAttnInterval: 6, slidingKvHeads: 8, slidingHeadDim: 256, globalKvHeads: 8, globalHeadDim: 256, kvSharedLayers: 0 },
      { id: 'gemma3-27b',  name: 'Gemma 3 27B',  params: 27,   type: 'sliding-dense', layers: 62, kvHeads: 16, headDim: 128, defaultSize: 16,   slidingWindow: 1024, globalAttnInterval: 6, slidingKvHeads: 16, slidingHeadDim: 128, globalKvHeads: 16, globalHeadDim: 128, kvSharedLayers: 0 },
    ],
  },
  {
    group: 'Ministral 3',
    models: [
      { id: 'ministral3-3b',  name: 'Ministral 3 3B',  params: 3,  type: 'dense', layers: 26, kvHeads: 8, headDim: 128, defaultSize: 2.0 },
      { id: 'ministral3-8b',  name: 'Ministral 3 8B',  params: 8,  type: 'dense', layers: 34, kvHeads: 8, headDim: 128, defaultSize: 6.5 },
      { id: 'ministral3-14b', name: 'Ministral 3 14B', params: 14, type: 'dense', layers: 40, kvHeads: 8, headDim: 128, defaultSize: 9.5 },
    ],
  },
  {
    group: 'Granite',
    models: [
      { id: 'granite-swash-3b', name: 'Granite SWASH 3B-a600M', params: 3.02, type: 'sliding-moe', layers: 28, kvHeads: 4, headDim: 64, defaultSize: 1.8, slidingWindow: 128, globalAttnInterval: 4, slidingKvHeads: 4, slidingHeadDim: 64, globalKvHeads: 4, globalHeadDim: 64, kvSharedLayers: 0, totalExperts: 48, activeExperts: 4, attnFraction: 0.05 },
    ],
  },
  {
    group: 'Muse Glimmer',
    models: [
      { id: 'muse-glimmer-30b', name: 'Muse Glimmer 30B', params: 29.6, type: 'sliding-dense', layers: 52, kvHeads: 2, headDim: 128, defaultSize: 16.8, slidingWindow: 2048, globalAttnInterval: 4, slidingKvHeads: 2, slidingHeadDim: 128, globalKvHeads: 2, globalHeadDim: 128, kvSharedLayers: 0 },
    ],
  },
  {
    group: 'Gemma 4',
    models: [
      { id: 'gemma4-e2b',      name: 'Gemma 4 E2B',          params: 5.1,  type: 'sliding-dense', layers: 35, kvHeads: 1, headDim: 512, defaultSize: 3.2,  slidingWindow: 512,  globalAttnInterval: 5, slidingKvHeads: 1,  slidingHeadDim: 256, globalKvHeads: 1, globalHeadDim: 512, kvSharedLayers: 20 },
      { id: 'gemma4-e4b',      name: 'Gemma 4 E4B',          params: 8,    type: 'sliding-dense', layers: 42, kvHeads: 2, headDim: 512, defaultSize: 5.0,  slidingWindow: 512,  globalAttnInterval: 6, slidingKvHeads: 2,  slidingHeadDim: 256, globalKvHeads: 2, globalHeadDim: 512, kvSharedLayers: 18 },
      { id: 'gemma4-12b',      name: 'Gemma 4 12B Unified',  params: 11.95, type: 'sliding-dense', layers: 48, kvHeads: 8, headDim: 256, defaultSize: 7.5,  slidingWindow: 1024, globalAttnInterval: 6, slidingKvHeads: 8,  slidingHeadDim: 256, globalKvHeads: 1, globalHeadDim: 512, kvSharedLayers: 0  },
      { id: 'gemma4-31b',      name: 'Gemma 4 31B',          params: 30.7, type: 'sliding-dense', layers: 60, kvHeads: 4, headDim: 512, defaultSize: 17.4, slidingWindow: 1024, globalAttnInterval: 6, slidingKvHeads: 16, slidingHeadDim: 256, globalKvHeads: 4, globalHeadDim: 512, kvSharedLayers: 0  },
      { id: 'gemma4-26b-a4b',  name: 'Gemma 4 26B-A4B',      params: 26,   type: 'sliding-moe',   layers: 30, kvHeads: 2, headDim: 512, defaultSize: 15.6, slidingWindow: 1024, globalAttnInterval: 6, slidingKvHeads: 8,  slidingHeadDim: 256, globalKvHeads: 2, globalHeadDim: 512, kvSharedLayers: 0, totalExperts: 128, activeExperts: 8, attnFraction: 0.05 },
    ],
  },
];

const PRECISION_SCALE = {
  fp16: 1,
  fp8: 0.5,
  q8: 0.53,
  q4: 0.28,
};

const GEN_PRESETS = [
  {
    group: 'Stable Diffusion',
    modality: 'image',
    models: [
      { id: 'sd15', name: 'SD 1.5', params: 0.86, ditFp16GB: 1.72, textEncoderFp16GB: 0.25, vaeFp16GB: 0.16, hiddenDim: 320, layers: 12, spatialCompression: 8, temporalCompression: 1, patchSize: 1, refWidth: 512, refHeight: 512, refFrames: 1, refActivationGB: 1.8, defaultWidth: 512, defaultHeight: 512, defaultTeOnGpu: true },
      { id: 'sdxl', name: 'SDXL', params: 2.6, ditFp16GB: 5.2, textEncoderFp16GB: 1.65, vaeFp16GB: 0.16, hiddenDim: 640, layers: 18, spatialCompression: 8, temporalCompression: 1, patchSize: 1, refWidth: 1024, refHeight: 1024, refFrames: 1, refActivationGB: 2.8, defaultWidth: 1024, defaultHeight: 1024, defaultTeOnGpu: true },
      { id: 'sd35-medium', name: 'SD 3.5 Medium', params: 2.5, ditFp16GB: 5.0, textEncoderFp16GB: 11.3, vaeFp16GB: 0.16, hiddenDim: 1536, layers: 24, spatialCompression: 8, temporalCompression: 1, patchSize: 2, refWidth: 1024, refHeight: 1024, refFrames: 1, refActivationGB: 3.2, defaultWidth: 1024, defaultHeight: 1024, defaultTeOnGpu: false },
      { id: 'sd35-large', name: 'SD 3.5 Large', params: 8, ditFp16GB: 16.0, textEncoderFp16GB: 11.3, vaeFp16GB: 0.16, hiddenDim: 2432, layers: 38, spatialCompression: 8, temporalCompression: 1, patchSize: 2, refWidth: 1024, refHeight: 1024, refFrames: 1, refActivationGB: 4.0, defaultWidth: 1024, defaultHeight: 1024, defaultTeOnGpu: false },
    ],
  },
  {
    group: 'FLUX',
    modality: 'image',
    models: [
      { id: 'flux-schnell', name: 'FLUX.1 Schnell', params: 12, ditFp16GB: 23.8, textEncoderFp16GB: 9.75, vaeFp16GB: 0.32, hiddenDim: 3072, layers: 57, spatialCompression: 8, temporalCompression: 1, patchSize: 2, refWidth: 1024, refHeight: 1024, refFrames: 1, refActivationGB: 1.2, defaultWidth: 1024, defaultHeight: 1024, defaultTeOnGpu: false },
      { id: 'flux-dev', name: 'FLUX.1 Dev', params: 12, ditFp16GB: 23.8, textEncoderFp16GB: 9.75, vaeFp16GB: 0.32, hiddenDim: 3072, layers: 57, spatialCompression: 8, temporalCompression: 1, patchSize: 2, refWidth: 1024, refHeight: 1024, refFrames: 1, refActivationGB: 1.4, defaultWidth: 1024, defaultHeight: 1024, defaultTeOnGpu: false },
    ],
  },
  {
    group: 'Qwen',
    modality: 'image',
    models: [
      { id: 'qwen-image', name: 'Qwen-Image', params: 20, ditFp16GB: 40.0, textEncoderFp16GB: 14.0, vaeFp16GB: 0.35, hiddenDim: 3584, layers: 60, spatialCompression: 16, temporalCompression: 1, patchSize: 1, refWidth: 1024, refHeight: 1024, refFrames: 1, refActivationGB: 3.5, defaultWidth: 1024, defaultHeight: 1024, defaultTeOnGpu: false },
    ],
  },
  {
    group: 'Wan',
    modality: 'video',
    models: [
      { id: 'wan21-1.3b', name: 'Wan 2.1 1.3B', params: 1.3, ditFp16GB: 2.6, textEncoderFp16GB: 9.5, vaeFp16GB: 0.8, hiddenDim: 1536, layers: 30, spatialCompression: 16, temporalCompression: 4, patchSize: 1, refWidth: 832, refHeight: 480, refFrames: 81, refActivationGB: 4.5, defaultWidth: 832, defaultHeight: 480, defaultDuration: 5, defaultFps: 16, defaultTeOnGpu: false },
      { id: 'wan21-14b', name: 'Wan 2.1 14B', params: 14, ditFp16GB: 28.0, textEncoderFp16GB: 9.5, vaeFp16GB: 0.8, hiddenDim: 5120, layers: 40, spatialCompression: 16, temporalCompression: 4, patchSize: 1, refWidth: 1280, refHeight: 720, refFrames: 81, refActivationGB: 26, defaultWidth: 1280, defaultHeight: 720, defaultDuration: 5, defaultFps: 16, defaultTeOnGpu: false },
      { id: 'wan22-ti2v-5b', name: 'Wan 2.2 TI2V-5B', params: 5, ditFp16GB: 10.0, textEncoderFp16GB: 9.5, vaeFp16GB: 0.6, hiddenDim: 3072, layers: 30, spatialCompression: 32, temporalCompression: 4, patchSize: 1, refWidth: 1280, refHeight: 720, refFrames: 121, refActivationGB: 10, defaultWidth: 1280, defaultHeight: 720, defaultDuration: 5, defaultFps: 24, defaultTeOnGpu: false },
      { id: 'wan22-a14b', name: 'Wan 2.2 A14B', params: 27, ditFp16GB: 54.0, textEncoderFp16GB: 9.5, vaeFp16GB: 0.8, hiddenDim: 5120, layers: 40, spatialCompression: 16, temporalCompression: 4, patchSize: 1, refWidth: 1280, refHeight: 720, refFrames: 81, refActivationGB: 18, defaultWidth: 1280, defaultHeight: 720, defaultDuration: 5, defaultFps: 16, defaultTeOnGpu: false },
    ],
  },
  {
    group: 'HunyuanVideo',
    modality: 'video',
    models: [
      { id: 'hunyuan-13b', name: 'HunyuanVideo 13B', params: 13, ditFp16GB: 26.0, textEncoderFp16GB: 9.5, vaeFp16GB: 1.2, hiddenDim: 3072, layers: 60, spatialCompression: 16, temporalCompression: 4, patchSize: 1, refWidth: 1280, refHeight: 720, refFrames: 129, refActivationGB: 40, defaultWidth: 1280, defaultHeight: 720, defaultDuration: 5, defaultFps: 24, defaultTeOnGpu: false },
      { id: 'hunyuan-1.5', name: 'HunyuanVideo 1.5', params: 8.3, ditFp16GB: 16.6, textEncoderFp16GB: 9.5, vaeFp16GB: 0.8, hiddenDim: 2048, layers: 48, spatialCompression: 16, temporalCompression: 4, patchSize: 1, refWidth: 1280, refHeight: 720, refFrames: 121, refActivationGB: 5.5, defaultWidth: 1280, defaultHeight: 720, defaultDuration: 5, defaultFps: 24, defaultTeOnGpu: false },
    ],
  },
  {
    group: 'Other Video',
    modality: 'video',
    models: [
      { id: 'ltx-2b', name: 'LTX-Video 2B', params: 2, ditFp16GB: 4.0, textEncoderFp16GB: 9.5, vaeFp16GB: 0.5, hiddenDim: 2048, layers: 28, spatialCompression: 32, temporalCompression: 8, patchSize: 1, refWidth: 768, refHeight: 512, refFrames: 121, refActivationGB: 8, defaultWidth: 768, defaultHeight: 512, defaultDuration: 5, defaultFps: 24, defaultTeOnGpu: false },
      { id: 'cogvideox-5b', name: 'CogVideoX 5B', params: 5, ditFp16GB: 10.6, textEncoderFp16GB: 9.5, vaeFp16GB: 0.8, hiddenDim: 3072, layers: 42, spatialCompression: 16, temporalCompression: 4, patchSize: 1, refWidth: 720, refHeight: 480, refFrames: 49, refActivationGB: 14, defaultWidth: 720, defaultHeight: 480, defaultDuration: 6, defaultFps: 8, defaultTeOnGpu: false },
    ],
  },
];

const MODE_COPY = {
  llm: {
    title: 'LLM Context Size Calculator',
    subtitle: 'Estimate the maximum context window for running quantized models locally with llama.cpp',
    heroTitle: 'Maximum Context Size',
    heroUnit: 'tokens',
    footer: 'Calculations are estimates. Actual memory usage varies with llama.cpp version, batch size, and system configuration.',
    defaultPreset: 'llama31-8b',
  },
  image: {
    title: 'Image Generation VRAM Calculator',
    subtitle: 'Estimate peak VRAM for local image models such as FLUX, SDXL, and Qwen-Image',
    heroTitle: 'Peak VRAM',
    heroUnit: 'GB',
    footer: 'Peak-VRAM estimates assume the DiT/UNet is fully on GPU. ComfyUI block-swapping and extra nodes can use more or less memory than this.',
    defaultPreset: 'flux-schnell',
  },
  video: {
    title: 'Video Generation VRAM Calculator',
    subtitle: 'Estimate peak VRAM for local video models such as Wan, HunyuanVideo, and LTX',
    heroTitle: 'Peak VRAM',
    heroUnit: 'GB',
    footer: 'Peak-VRAM estimates assume the DiT is fully on GPU. Frame count, resolution, and VAE tiling dominate video memory more than step count.',
    defaultPreset: 'wan22-ti2v-5b',
  },
};

let currentMode = 'llm';

// ============================================================
// DOM helpers
// ============================================================

const $ = (id) => document.getElementById(id);

const el = {
  vram:           $('vram'),
  ram:            $('ram'),
  preset:         $('model-preset'),
  modelType:      $('model-type'),
  totalParams:    $('total-params'),
  paramsNote:     $('params-note'),
  numLayers:      $('num-layers'),
  kvHeads:        $('kv-heads'),
  headDim:        $('head-dim'),
  modelSize:      $('model-size'),
  hybridSection:  $('hybrid-section'),
  fullAttnInterval: $('full-attn-interval'),
  fullAttnLayers: $('full-attn-layers'),
  linearKeyDim:   $('linear-key-dim'),
  linearValueHeads: $('linear-value-heads'),
  linearValueDim: $('linear-value-dim'),
  hybridNote:     $('hybrid-note'),
  slidingSection: $('sliding-section'),
  slidingWindow:  $('sliding-window'),
  globalAttnInterval: $('global-attn-interval'),
  globalAttnLayers: $('global-attn-layers'),
  slidingKvHeads: $('sliding-kv-heads'),
  slidingHeadDim: $('sliding-head-dim'),
  globalKvHeads:  $('global-kv-heads'),
  globalHeadDim:  $('global-head-dim'),
  kvSharedLayers: $('kv-shared-layers'),
  slidingNote:    $('sliding-note'),
  moeSection:     $('moe-section'),
  totalExperts:   $('total-experts'),
  activeExperts:  $('active-experts'),
  moeNote:        $('moe-note'),
  expertsGpu:     $('experts-gpu'),
  expertsGpuLbl:  $('experts-gpu-label'),
  attnFraction:   $('attn-fraction'),
  kvQuant:        $('kv-quant'),
  gpuLayers:      $('gpu-layers'),
  gpuLayersLbl:   $('gpu-layers-label'),
  vramOverhead:   $('vram-overhead'),
  ramOverhead:    $('ram-overhead'),
  desiredCtx:     $('desired-context'),
  maxContext:      $('max-context'),
  contextNote:    $('context-note'),
  reverseSection: $('reverse-section'),
  reverseResult:  $('reverse-result'),
  llamaCmd:       $('llama-command'),
  copyBtn:        $('copy-command'),
  pageTitle:      $('page-title'),
  pageSubtitle:   $('page-subtitle'),
  heroTitle:      $('hero-title'),
  heroUnit:       $('hero-unit'),
  footerNote:     $('footer-note'),
  ditSize:        $('dit-size'),
  teSize:         $('te-size'),
  vaeSize:        $('vae-size'),
  spatialCompression: $('spatial-compression'),
  patchSize:      $('patch-size'),
  temporalCompression: $('temporal-compression'),
  activationBase: $('activation-base'),
  genArchNote:    $('gen-arch-note'),
  refWidth:       $('ref-width'),
  refHeight:      $('ref-height'),
  refFrames:      $('ref-frames'),
  genPrecision:   $('gen-precision'),
  genWidth:       $('gen-width'),
  genHeight:      $('gen-height'),
  genDuration:    $('gen-duration'),
  genFps:         $('gen-fps'),
  framesNote:     $('frames-note'),
  genBatch:       $('gen-batch'),
  teOnGpu:        $('te-on-gpu'),
  vaeTiling:      $('vae-tiling'),
  reverseTitle:   $('reverse-title'),
  genFitNote:     $('gen-fit-note'),
  vramMidLabel:   $('vram-mid-label'),
  ramMidLabel:    $('ram-mid-label'),
  vramAuxRow:     $('vram-aux-row'),
  ramAuxRow:      $('ram-aux-row'),
};

// ============================================================
// Preset Management
// ============================================================

function activeCatalog() {
  if (currentMode === 'llm') return MODEL_PRESETS;
  return GEN_PRESETS.filter((g) => g.modality === currentMode);
}

function populatePresets() {
  const sel = el.preset;
  sel.innerHTML = '';
  const custom = document.createElement('option');
  custom.value = 'custom';
  custom.textContent = '-- Custom --';
  sel.appendChild(custom);

  for (const group of activeCatalog()) {
    const og = document.createElement('optgroup');
    og.label = group.group;
    for (const m of group.models) {
      const opt = document.createElement('option');
      opt.value = m.id;
      opt.textContent = m.name;
      og.appendChild(opt);
    }
    sel.appendChild(og);
  }
}

function findPreset(id) {
  for (const g of activeCatalog())
    for (const m of g.models)
      if (m.id === id) return m;
  return null;
}

function isHybridType(type) {
  return type === 'hybrid-dense' || type === 'hybrid-moe';
}

function isMoeType(type) {
  return type === 'moe' || type === 'hybrid-moe' || type === 'sliding-moe';
}

function isSlidingType(type) {
  return type === 'sliding-dense' || type === 'sliding-moe';
}

function applyPreset(id) {
  const p = findPreset(id);
  if (!p) return;
  if (currentMode === 'llm') applyLlmPreset(p);
  else applyGenPreset(p);
}

function applyLlmPreset(p) {
  el.totalParams.value = p.params;
  el.modelType.value   = p.type;
  el.numLayers.value   = p.layers;
  el.kvHeads.value     = p.kvHeads;
  el.headDim.value     = p.headDim;
  el.modelSize.value   = p.defaultSize;

  const isHybrid = isHybridType(p.type);
  const isMoe    = isMoeType(p.type);
  const isSliding = isSlidingType(p.type);

  if (isHybrid) {
    el.fullAttnInterval.value = p.fullAttnInterval;
    el.linearKeyDim.value = p.linearKeyDim;
    el.linearValueHeads.value = p.linearValueHeads;
    el.linearValueDim.value = p.linearValueDim;
  }

  if (isSliding) {
    el.slidingWindow.value = p.slidingWindow;
    el.globalAttnInterval.value = p.globalAttnInterval;
    el.slidingKvHeads.value = p.slidingKvHeads;
    el.slidingHeadDim.value = p.slidingHeadDim;
    el.globalKvHeads.value = p.globalKvHeads;
    el.globalHeadDim.value = p.globalHeadDim;
    el.kvSharedLayers.value = p.kvSharedLayers || 0;
  }

  if (isMoe) {
    el.totalExperts.value  = p.totalExperts;
    el.activeExperts.value = p.activeExperts;
    el.attnFraction.value  = p.attnFraction;
    el.expertsGpu.max      = p.totalExperts;
    el.expertsGpu.value    = p.totalExperts;
  }

  updateVisibility();
  updateDerivedNotes();
  el.gpuLayers.value = p.layers + 1;
  updateGpuSlider();
  updateExpertsSlider();
  calculate();
}

function applyGenPreset(p) {
  el.totalParams.value = p.params;
  el.ditSize.value = p.ditFp16GB;
  el.teSize.value = p.textEncoderFp16GB;
  el.vaeSize.value = p.vaeFp16GB;
  el.spatialCompression.value = p.spatialCompression;
  el.patchSize.value = p.patchSize;
  el.temporalCompression.value = p.temporalCompression || 1;
  el.activationBase.value = p.refActivationGB;
  el.refWidth.value = p.refWidth;
  el.refHeight.value = p.refHeight;
  el.refFrames.value = p.refFrames;
  el.genWidth.value = p.defaultWidth;
  el.genHeight.value = p.defaultHeight;
  el.teOnGpu.checked = !!p.defaultTeOnGpu;

  if (currentMode === 'video') {
    el.genDuration.value = p.defaultDuration || 5;
    el.genFps.value = p.defaultFps || 24;
  }

  syncResPresetButtons();
  updateDerivedNotes();
  calculate();
}

function setMode(mode) {
  if (mode !== 'llm' && mode !== 'image' && mode !== 'video') return;
  currentMode = mode;
  document.body.classList.remove('mode-llm', 'mode-image', 'mode-video');
  document.body.classList.add('mode-' + mode);

  const copy = MODE_COPY[mode];
  el.pageTitle.textContent = copy.title;
  el.pageSubtitle.textContent = copy.subtitle;
  el.heroTitle.textContent = copy.heroTitle;
  el.heroUnit.textContent = copy.heroUnit;
  el.footerNote.textContent = copy.footer;
  document.title = copy.title;

  document.querySelectorAll('.mode-switch button').forEach((btn) => {
    const active = btn.dataset.mode === mode;
    btn.classList.toggle('active', active);
    btn.setAttribute('aria-selected', active ? 'true' : 'false');
  });

  const midLabel = mode === 'llm' ? 'KV Cache' : 'Activations / Latents';
  el.vramMidLabel.textContent = midLabel;
  el.ramMidLabel.textContent = midLabel;
  el.vramAuxRow.hidden = mode === 'llm';
  el.ramAuxRow.hidden = mode === 'llm';

  populatePresets();
  el.preset.value = copy.defaultPreset;
  applyPreset(copy.defaultPreset);
}

// ============================================================
// UI Visibility & Slider Updates
// ============================================================

function updateVisibility() {
  const t = el.modelType.value;
  const isHybrid = isHybridType(t);
  const isMoe    = isMoeType(t);
  const isSliding = isSlidingType(t);

  el.hybridSection.hidden = !isHybrid;
  el.slidingSection.hidden = !isSliding;
  el.moeSection.hidden    = !isMoe;

  if (isHybrid) {
    updateHybridDerivedFields();
  }
  if (isSliding) {
    updateSlidingDerivedFields();
  }

  updateDerivedNotes();
}

function updateHybridDerivedFields() {
  const layers = int(el.numLayers) || 1;
  const interval = Math.min(layers, Math.max(1, int(el.fullAttnInterval) || 1));
  const full = Math.max(1, Math.floor(layers / interval));
  const linear = Math.max(0, layers - full);

  el.fullAttnInterval.value = interval;
  el.fullAttnLayers.value = full;
  el.hybridNote.textContent = `Assumes an evenly spaced layout: every ${interval}th layer is full attention (${full} KV-cache layers, ${linear} linear-state layers).`;
}

function updateSlidingDerivedFields() {
  const layers = int(el.numLayers) || 1;
  const interval = Math.min(layers, Math.max(1, int(el.globalAttnInterval) || 1));
  const global = Math.max(1, Math.floor(layers / interval));
  const sliding = Math.max(0, layers - global);
  const shared = Math.min(layers - 1, Math.max(0, int(el.kvSharedLayers) || 0));
  const unique = Math.max(1, layers - shared);
  const window = Math.max(1, int(el.slidingWindow) || 1);

  el.globalAttnInterval.value = interval;
  el.globalAttnLayers.value = global;
  el.kvSharedLayers.value = shared;
  el.slidingNote.textContent = `Uses ${sliding} sliding-window layers capped at ${window.toLocaleString()} tokens and ${global} global layers. Counts ${unique} unique KV layers after sharing.`;
}

function updateDerivedNotes() {
  const params = num(el.totalParams);

  if (currentMode !== 'llm') {
    const dit = num(el.ditSize);
    const te = num(el.teSize);
    const vae = num(el.vaeSize);
    const scale = PRECISION_SCALE[el.genPrecision.value] || 1;
    const bits = scale * 16;
    if (params > 0 && dit > 0) {
      el.paramsNote.textContent = `DiT/UNet weights at ${el.genPrecision.value} are about ${fmtGB(dit * scale)} (${bits.toFixed(1)}-bit scale). Parameter count is metadata only.`;
    } else {
      el.paramsNote.textContent = '';
    }
    updateGenArchNote();
    return;
  }

  const size = num(el.modelSize);
  if (params > 0 && size > 0) {
    const bitsPerWeight = (size * GB * 8) / (params * 1e9);
    el.paramsNote.textContent = `Current file size implies about ${bitsPerWeight.toFixed(2)} bits/weight. Context estimates use file size, not parameter count.`;
  } else {
    el.paramsNote.textContent = '';
  }

  const isMoe = isMoeType(el.modelType.value);
  if (isMoe) {
    const active = int(el.activeExperts) || 0;
    const total = int(el.totalExperts) || 0;
    el.moeNote.textContent = `${active} of ${total} experts activate per token. Static memory here is driven mainly by file size and expert placement across GPU/CPU.`;
  } else {
    el.moeNote.textContent = '';
  }
}

function genFrameCount() {
  if (currentMode !== 'video') return 1;
  const fps = Math.max(1, num(el.genFps) || 24);
  const duration = Math.max(0.5, num(el.genDuration) || 1);
  return Math.max(1, Math.round(duration * fps) + 1);
}

function genLatentTokens(width, height, frames, spatial, temporal, patch) {
  const t = Math.max(1, frames / Math.max(1, temporal));
  const h = Math.max(1, height / Math.max(1, spatial));
  const w = Math.max(1, width / Math.max(1, spatial));
  const p = Math.max(1, patch);
  return (t * h * w) / (p * p);
}

function updateGenArchNote() {
  const width = Math.max(64, int(el.genWidth) || 64);
  const height = Math.max(64, int(el.genHeight) || 64);
  const frames = genFrameCount();
  const spatial = Math.max(1, int(el.spatialCompression) || 8);
  const temporal = currentMode === 'video' ? Math.max(1, int(el.temporalCompression) || 4) : 1;
  const patch = Math.max(1, int(el.patchSize) || 1);
  const tokens = genLatentTokens(width, height, frames, spatial, temporal, patch);
  const refTokens = genLatentTokens(
    Math.max(1, int(el.refWidth) || width),
    Math.max(1, int(el.refHeight) || height),
    Math.max(1, int(el.refFrames) || frames),
    spatial,
    temporal,
    patch,
  );
  const ratio = tokens / Math.max(1, refTokens);
  el.genArchNote.textContent = `${Math.round(tokens).toLocaleString()} latent tokens · ${ratio.toFixed(2)}× the preset activation baseline.`;
  if (el.framesNote) {
    el.framesNote.textContent = currentMode === 'video'
      ? `${frames.toLocaleString()} frames at ${Math.max(1, num(el.genFps) || 24)} fps (includes a start frame).`
      : '';
  }
}

function updateGpuSlider() {
  const n = int(el.numLayers) || 1;
  el.gpuLayers.max = n + 1;
  if (int(el.gpuLayers) > n + 1) el.gpuLayers.value = n + 1;
  updateGpuLabel();
}

function updateGpuLabel() {
  const v = int(el.gpuLayers);
  const n = int(el.numLayers) || 1;
  const onGpu = Math.min(v, n);
  const onCpu = n - onGpu;
  const outputGpu = v > n;

  let text = `${onGpu} / ${n} layers on GPU`;
  if (outputGpu) text += ' + output';
  if (onCpu > 0) text += ` · ${onCpu} on CPU`;
  el.gpuLayersLbl.textContent = text;
}

function updateExpertsSlider() {
  const total = int(el.totalExperts) || 1;
  el.expertsGpu.max = total;
  if (int(el.expertsGpu) > total) el.expertsGpu.value = total;
  updateExpertsLabel();
}

function updateExpertsLabel() {
  el.expertsGpuLbl.textContent = `${int(el.expertsGpu)} / ${int(el.totalExperts) || 1}`;
}

// ============================================================
// Calculation Engine
// ============================================================

function countAttentionLayers(layerCount, interval) {
  const global = Math.max(0, Math.floor(layerCount / interval));
  return {
    global,
    sliding: Math.max(0, layerCount - global),
  };
}

function applyKvSharingToMix(mix, totalMix, sharedLayers) {
  const sharedSliding = Math.min(sharedLayers, totalMix.sliding);
  const slidingRatio = totalMix.sliding > 0
    ? Math.max(0, totalMix.sliding - sharedSliding) / totalMix.sliding
    : 1;
  const sharedGlobal = Math.max(0, sharedLayers - sharedSliding);
  const globalRatio = totalMix.global > 0
    ? Math.max(0, totalMix.global - sharedGlobal) / totalMix.global
    : 1;

  return {
    sliding: Math.min(mix.sliding, Math.round(mix.sliding * slidingRatio)),
    global: Math.min(mix.global, Math.round(mix.global * globalRatio)),
  };
}

function kvProfile(globalLayers, slidingLayers, globalBytesPerLayer, slidingBytesPerLayer, slidingWindow) {
  return {
    globalBytesPerToken: globalLayers * globalBytesPerLayer,
    slidingBytesPerToken: slidingLayers * slidingBytesPerLayer,
    slidingWindow,
  };
}

function maxContextForProfile(memoryGB, profile) {
  const budget = memoryGB * GB;
  const globalBpt = profile.globalBytesPerToken;
  const slidingBpt = profile.slidingBytesPerToken;
  const totalBpt = globalBpt + slidingBpt;

  if (totalBpt <= 0) return Infinity;
  if (budget <= 0) return 0;
  if (slidingBpt <= 0) return Math.floor(budget / globalBpt);

  const window = Math.max(1, profile.slidingWindow);
  const withinWindow = Math.floor(budget / totalBpt);
  if (withinWindow <= window) return withinWindow;

  const cappedSlidingBytes = slidingBpt * window;
  const remaining = budget - cappedSlidingBytes;
  if (remaining <= 0) return Math.floor(budget / totalBpt);
  if (globalBpt <= 0) return Infinity;
  return Math.floor(remaining / globalBpt);
}

function kvUsageForContextGB(profile, ctx) {
  if (!Number.isFinite(ctx)) return Infinity;
  const slidingCtx = Math.min(ctx, Math.max(1, profile.slidingWindow));
  const bytes = (profile.globalBytesPerToken * ctx) + (profile.slidingBytesPerToken * slidingCtx);
  return bytes / GB;
}

function readGenInputs() {
  const width = Math.max(64, int(el.genWidth) || 64);
  const height = Math.max(64, int(el.genHeight) || 64);
  const frames = genFrameCount();
  return {
    vram: num(el.vram),
    ram: num(el.ram),
    ditFp16: num(el.ditSize),
    teFp16: num(el.teSize),
    vaeFp16: num(el.vaeSize),
    precision: el.genPrecision.value,
    teOnGpu: el.teOnGpu.checked,
    vaeTiling: el.vaeTiling.checked,
    width,
    height,
    frames,
    batch: Math.max(1, int(el.genBatch) || 1),
    spatial: Math.max(1, int(el.spatialCompression) || 8),
    temporal: currentMode === 'video' ? Math.max(1, int(el.temporalCompression) || 4) : 1,
    patch: Math.max(1, int(el.patchSize) || 1),
    refWidth: Math.max(1, int(el.refWidth) || width),
    refHeight: Math.max(1, int(el.refHeight) || height),
    refFrames: Math.max(1, int(el.refFrames) || frames),
    activationBase: Math.max(0, num(el.activationBase)),
    vramOH: (num(el.vramOverhead) || 0) / 1024,
    ramOH: (num(el.ramOverhead) || 0) / 1024,
  };
}

function estimateGenMemory(opts) {
  const scale = PRECISION_SCALE[opts.precision] || 1;
  const tokens = genLatentTokens(opts.width, opts.height, opts.frames, opts.spatial, opts.temporal, opts.patch);
  const refTokens = Math.max(1, genLatentTokens(opts.refWidth, opts.refHeight, opts.refFrames, opts.spatial, opts.temporal, opts.patch));
  const actScale = opts.vaeTiling ? 0.75 : 1;
  const gpuAct = opts.activationBase * (tokens / refTokens) * opts.batch * actScale;
  const gpuModel = opts.ditFp16 * scale;
  const te = opts.teFp16 * scale;
  const gpuAux = (opts.teOnGpu ? te : 0) + opts.vaeFp16;
  const ramAux = opts.teOnGpu ? 0 : te;

  return {
    gpuModel,
    gpuAct,
    gpuAux,
    gpuOH: opts.vramOH,
    ramModel: 0,
    ramAct: 0,
    ramAux,
    ramOH: opts.ramOH,
    totalVram: gpuModel + gpuAct + gpuAux + opts.vramOH,
    totalRam: ramAux + opts.ramOH,
    tokens,
  };
}

function genFits(mem, vram, ram) {
  return mem.totalVram <= vram && mem.totalRam <= ram;
}

function maxSquareResolution(baseOpts) {
  let best = 0;
  for (let size = 256; size <= 2048; size += 64) {
    const mem = estimateGenMemory({ ...baseOpts, width: size, height: size });
    if (genFits(mem, baseOpts.vram, baseOpts.ram)) best = size;
    else break;
  }
  return best;
}

function maxVideoDuration(baseOpts) {
  const fps = Math.max(1, num(el.genFps) || 24);
  let best = 0;
  for (let dur = 0.5; dur <= 20; dur += 0.5) {
    const frames = Math.max(1, Math.round(dur * fps) + 1);
    const mem = estimateGenMemory({ ...baseOpts, frames });
    if (genFits(mem, baseOpts.vram, baseOpts.ram)) best = dur;
    else break;
  }
  return best;
}

function calculate() {
  if (currentMode !== 'llm') {
    calculateGen();
    return;
  }
  calculateLlm();
}

function calculateGen() {
  updateGenArchNote();
  const opts = readGenInputs();
  const mem = estimateGenMemory(opts);
  const fitsVram = mem.totalVram <= opts.vram;
  const fitsRam = mem.totalRam <= opts.ram;
  const fits = fitsVram && fitsRam;

  let bottleneck = '';
  if (mem.gpuModel + mem.gpuAux + mem.gpuOH > opts.vram) {
    bottleneck = 'Static GPU memory exceeds available VRAM';
  } else if (!fitsVram) {
    bottleneck = 'Limited by VRAM';
  } else if (!fitsRam) {
    bottleneck = 'Limited by RAM (offloaded text encoder)';
  } else {
    bottleneck = 'Fits in available memory';
  }

  const maxRes = currentMode === 'image' ? maxSquareResolution(opts) : 0;
  const maxDur = currentMode === 'video' ? maxVideoDuration(opts) : 0;
  let headroom = '';
  if (currentMode === 'image') {
    headroom = maxRes > 0
      ? `Max square resolution about ${maxRes}×${maxRes}`
      : 'Not enough memory for a 256×256 image with these settings';
  } else {
    headroom = maxDur > 0
      ? `Max duration about ${maxDur.toFixed(1)}s at ${Math.max(1, num(el.genFps) || 24)} fps`
      : 'Not enough memory for a 0.5s clip with these settings';
  }

  renderResults({
    mode: currentMode,
    maxCtx: mem.totalVram,
    bottleneck: `${bottleneck}. ${headroom}.`,
    fits,
    gpuModelMem: mem.gpuModel,
    cpuModelMem: mem.ramModel,
    kvVramUsed: mem.gpuAct,
    kvRamUsed: mem.ramAct,
    auxVram: mem.gpuAux,
    auxRam: mem.ramAux,
    vramOH: mem.gpuOH,
    ramOH: mem.ramOH,
    totalVram: mem.totalVram,
    totalRam: mem.totalRam,
    freeVram: opts.vram - mem.totalVram,
    freeRam: opts.ram - mem.totalRam,
    vram: opts.vram,
    ram: opts.ram,
    headroom,
    width: opts.width,
    height: opts.height,
    frames: opts.frames,
    precision: opts.precision,
  });

  renderGenReverse(opts, mem, maxRes, maxDur);
  if (el.genFitNote) {
    const te = opts.teOnGpu ? 'text encoder on GPU' : 'text encoder offloaded to RAM';
    const vae = opts.vaeTiling ? 'VAE tiling on' : 'VAE tiling off';
    el.genFitNote.textContent = `Assumes the DiT/UNet is fully on GPU at ${opts.precision}, with ${te} and ${vae}. Attention is treated as flash/sage-style (linear in latent tokens). ComfyUI block-swapping can use less VRAM than this estimate.`;
  }
}

function renderGenReverse(opts, mem, maxRes, maxDur) {
  el.reverseSection.hidden = false;
  el.reverseTitle.textContent = currentMode === 'image' ? 'Resolution Headroom' : 'Duration Headroom';
  const fits = genFits(mem, opts.vram, opts.ram);
  const target = currentMode === 'image'
    ? `${opts.width}×${opts.height}`
    : `${opts.frames.toLocaleString()} frames (${(opts.frames / Math.max(1, num(el.genFps) || 24)).toFixed(1)}s)`;
  const extra = currentMode === 'image'
    ? (maxRes > 0 ? `Largest square that fits: <strong>${maxRes}×${maxRes}</strong>` : 'No square resolution in the 256–2048 range fits.')
    : (maxDur > 0 ? `Longest clip that fits: <strong>${maxDur.toFixed(1)}s</strong>` : 'No clip of 0.5s or longer fits.');

  el.reverseResult.innerHTML =
    `<p>For <strong>${target}</strong>, batch ${opts.batch}:</p>` +
    `<p>VRAM needed: <strong>${fmtGB(mem.totalVram)}</strong> / ${fmtGB(opts.vram)} ` +
    `<span class="${fits && mem.totalVram <= opts.vram ? 'fits' : 'no-fit'}">${mem.totalVram <= opts.vram ? 'OK' : 'OVER'}</span></p>` +
    `<p>RAM needed: <strong>${fmtGB(mem.totalRam)}</strong> / ${fmtGB(opts.ram)} ` +
    `<span class="${mem.totalRam <= opts.ram ? 'fits' : 'no-fit'}">${mem.totalRam <= opts.ram ? 'OK' : 'OVER'}</span></p>` +
    `<p>${extra}</p>` +
    `<p><strong class="${fits ? 'fits' : 'no-fit'}">${fits ? 'This generation setting fits!' : 'Does not fit in available memory.'}</strong></p>`;
}

function calculateLlm() {
  const vram       = num(el.vram);
  const ram        = num(el.ram);
  const modelType  = el.modelType.value;
  const numLayers  = int(el.numLayers) || 1;
  const kvHeads    = int(el.kvHeads)   || 1;
  const headDim    = int(el.headDim)   || 128;
  const fileSize   = num(el.modelSize);
  const kvQuant    = el.kvQuant.value;
  const gpuInput   = int(el.gpuLayers);
  const vramOH     = (num(el.vramOverhead) || 0) / 1024;
  const ramOH      = (num(el.ramOverhead)  || 0) / 1024;

  const isMoe    = isMoeType(modelType);
  const isHybrid = isHybridType(modelType);
  const isSliding = isSlidingType(modelType);
  const kvBytes  = KV_QUANT_BYTES[kvQuant] || 2;

  // Bytes of KV cache per token per full-attention layer
  const kvPerTokenPerLayer = 2 * kvHeads * headDim * kvBytes;

  // Hybrid architectures assume evenly spaced full-attention layers.
  const fullAttnInterval = isHybrid ? Math.min(numLayers, Math.max(1, int(el.fullAttnInterval) || 1)) : 1;
  const fullAttnLayers = isHybrid ? Math.max(1, Math.floor(numLayers / fullAttnInterval)) : numLayers;
  const linearKeyDim = isHybrid ? Math.max(1, int(el.linearKeyDim) || 128) : 0;
  const linearValueHeads = isHybrid ? Math.max(1, int(el.linearValueHeads) || 32) : 0;
  const linearValueDim = isHybrid ? Math.max(1, int(el.linearValueDim) || 128) : 0;

  // GPU / CPU layer split (model layers, not counting output)
  const gpuModelLayers = Math.min(gpuInput, numLayers);
  const cpuModelLayers = numLayers - gpuModelLayers;
  const outputOnGpu    = gpuInput > numLayers;

  // Model weight distribution (GB)
  let gpuModelMem, cpuModelMem;
  const unitWeight = fileSize / (numLayers + 1);

  if (isMoe) {
    const totalExp     = int(el.totalExperts) || 1;
    const expOnGpu     = Math.min(totalExp, Math.max(0, int(el.expertsGpu)));
    const attnFrac     = num(el.attnFraction) || 0.05;
    const expFrac      = 1 - attnFrac;
    const perLayer     = fileSize * numLayers / (numLayers + 1) / numLayers;
    const gpuPerLayer  = perLayer * (attnFrac + expFrac * (expOnGpu / totalExp));

    gpuModelMem = gpuPerLayer * gpuModelLayers + (outputOnGpu ? unitWeight : 0);
    cpuModelMem = Math.max(0, fileSize - gpuModelMem);
  } else {
    gpuModelMem = unitWeight * gpuModelLayers + (outputOnGpu ? unitWeight : 0);
    cpuModelMem = Math.max(0, fileSize - gpuModelMem);
  }

  // KV-cache layers on GPU vs CPU
  let kvGpuLayers, kvCpuLayers, gpuKvProfile, cpuKvProfile;
  if (isSliding) {
    const globalAttnInterval = Math.min(numLayers, Math.max(1, int(el.globalAttnInterval) || 1));
    const slidingWindow = Math.max(1, int(el.slidingWindow) || 1);
    const sharedLayers = Math.min(numLayers - 1, Math.max(0, int(el.kvSharedLayers) || 0));
    const totalMix = countAttentionLayers(numLayers, globalAttnInterval);
    const gpuMixRaw = countAttentionLayers(gpuModelLayers, globalAttnInterval);
    const cpuMixRaw = {
      sliding: Math.max(0, totalMix.sliding - gpuMixRaw.sliding),
      global: Math.max(0, totalMix.global - gpuMixRaw.global),
    };
    const gpuMix = applyKvSharingToMix(gpuMixRaw, totalMix, sharedLayers);
    const cpuMix = applyKvSharingToMix(cpuMixRaw, totalMix, sharedLayers);
    const slidingBytesPerLayer = 2 * (int(el.slidingKvHeads) || 1) * (int(el.slidingHeadDim) || 256) * kvBytes;
    const globalBytesPerLayer = 2 * (int(el.globalKvHeads) || 1) * (int(el.globalHeadDim) || 512) * kvBytes;

    kvGpuLayers = gpuMix.sliding + gpuMix.global;
    kvCpuLayers = cpuMix.sliding + cpuMix.global;
    gpuKvProfile = kvProfile(gpuMix.global, gpuMix.sliding, globalBytesPerLayer, slidingBytesPerLayer, slidingWindow);
    cpuKvProfile = kvProfile(cpuMix.global, cpuMix.sliding, globalBytesPerLayer, slidingBytesPerLayer, slidingWindow);
  } else if (isHybrid && fullAttnInterval > 1) {
    kvGpuLayers = Math.floor(gpuModelLayers / fullAttnInterval);
    kvCpuLayers = fullAttnLayers - kvGpuLayers;
    gpuKvProfile = kvProfile(kvGpuLayers, 0, kvPerTokenPerLayer, 0, 1);
    cpuKvProfile = kvProfile(kvCpuLayers, 0, kvPerTokenPerLayer, 0, 1);
  } else {
    kvGpuLayers = gpuModelLayers;
    kvCpuLayers = cpuModelLayers;
    gpuKvProfile = kvProfile(kvGpuLayers, 0, kvPerTokenPerLayer, 0, 1);
    cpuKvProfile = kvProfile(kvCpuLayers, 0, kvPerTokenPerLayer, 0, 1);
  }

  // Hybrid linear-attention layers keep a fixed-size recurrent state per layer.
  const linearGpuLayers = isHybrid ? Math.max(0, gpuModelLayers - kvGpuLayers) : 0;
  const linearCpuLayers = isHybrid ? Math.max(0, cpuModelLayers - kvCpuLayers) : 0;
  const linearStateBytesPerLayer = isHybrid
    ? linearValueHeads * linearValueDim * linearKeyDim * kvBytes
    : 0;
  const linearStateGpuGB = (linearStateBytesPerLayer * linearGpuLayers) / GB;
  const linearStateCpuGB = (linearStateBytesPerLayer * linearCpuLayers) / GB;

  // Only count VRAM overhead when GPU is actually used; hybrid linear state is fixed overhead.
  const effectiveVramOH = ((gpuModelLayers > 0 || outputOnGpu) ? vramOH : 0) + linearStateGpuGB;
  const effectiveRamOH = ramOH + linearStateCpuGB;

  // Available memory for KV cache (GB)
  const availVram = Math.max(0, vram - gpuModelMem - effectiveVramOH);
  const availRam  = Math.max(0, ram  - cpuModelMem - effectiveRamOH);

  // Max context from each constraint
  const maxVram = maxContextForProfile(availVram, gpuKvProfile);
  const maxRam = maxContextForProfile(availRam, cpuKvProfile);

  let maxCtx = Math.min(maxVram, maxRam);
  if (!Number.isFinite(maxCtx)) maxCtx = 0;
  maxCtx = Math.max(0, Math.floor(maxCtx / 256) * 256);

  // Bottleneck
  let bottleneck = '';
  if (gpuModelMem + effectiveVramOH > vram && gpuModelLayers > 0) {
    bottleneck = 'Static GPU memory exceeds available VRAM';
    maxCtx = 0;
  } else if (cpuModelMem + effectiveRamOH > ram && cpuModelLayers > 0) {
    bottleneck = 'Static CPU memory exceeds available RAM';
    maxCtx = 0;
  } else if (maxCtx <= 0) {
    bottleneck = 'Not enough memory for any context';
  } else if (maxVram <= maxRam) {
    bottleneck = 'Limited by VRAM';
  } else {
    bottleneck = 'Limited by RAM';
  }
  if (kvGpuLayers === 0 && kvCpuLayers === 0) {
    bottleneck = 'No KV-cache layers assigned';
    maxCtx = 0;
  }

  // Actual KV usage at max context (GB)
  const kvVramUsed = kvUsageForContextGB(gpuKvProfile, maxCtx);
  const kvRamUsed  = kvUsageForContextGB(cpuKvProfile, maxCtx);

  const results = {
    maxCtx,
    bottleneck,
    gpuModelMem,
    cpuModelMem,
    kvVramUsed,
    kvRamUsed,
    vramOH: effectiveVramOH,
    ramOH: effectiveRamOH,
    totalVram: gpuModelMem + kvVramUsed + effectiveVramOH,
    totalRam:  cpuModelMem + kvRamUsed  + effectiveRamOH,
    freeVram:  vram - (gpuModelMem + kvVramUsed + effectiveVramOH),
    freeRam:   ram  - (cpuModelMem + kvRamUsed  + effectiveRamOH),
    vram,
    ram,
    gpuInput,
    kvQuant,
    kvGpuLayers,
    kvCpuLayers,
    fullAttnLayers,
    fullAttnInterval,
    numLayers,
  };

  renderResults(results);
  renderReverse(gpuKvProfile, cpuKvProfile, gpuModelMem, cpuModelMem, effectiveVramOH, effectiveRamOH, vram, ram);
  renderCommand(results);
}

// ============================================================
// Reverse Mode
// ============================================================

function renderReverse(gpuKvProfile, cpuKvProfile, gpuMem, cpuMem, vramOH, ramOH, vram, ram) {
  const ctx = parseInt(el.desiredCtx.value) || 0;
  if (ctx <= 0) { el.reverseSection.hidden = true; return; }

  el.reverseSection.hidden = false;
  el.reverseTitle.textContent = 'Reverse Mode';
  const kvVram = kvUsageForContextGB(gpuKvProfile, ctx);
  const kvRam  = kvUsageForContextGB(cpuKvProfile, ctx);
  const needVram = gpuMem + kvVram + vramOH;
  const needRam  = cpuMem + kvRam  + ramOH;
  const fitsVram = needVram <= vram || (gpuKvProfile.globalBytesPerToken + gpuKvProfile.slidingBytesPerToken) === 0;
  const fitsRam  = needRam  <= ram  || (cpuKvProfile.globalBytesPerToken + cpuKvProfile.slidingBytesPerToken) === 0;
  const fits = fitsVram && fitsRam;

  el.reverseResult.innerHTML =
    `<p>For <strong>${ctx.toLocaleString()}</strong> tokens of context:</p>` +
    `<p>VRAM needed: <strong>${fmtGB(needVram)}</strong> / ${fmtGB(vram)} ` +
    `<span class="${fitsVram ? 'fits' : 'no-fit'}">${fitsVram ? 'OK' : 'OVER'}</span></p>` +
    `<p>RAM needed: <strong>${fmtGB(needRam)}</strong> / ${fmtGB(ram)} ` +
    `<span class="${fitsRam ? 'fits' : 'no-fit'}">${fitsRam ? 'OK' : 'OVER'}</span></p>` +
    `<p><strong class="${fits ? 'fits' : 'no-fit'}">${fits ? 'This context size fits!' : 'Does not fit in available memory.'}</strong></p>`;
}

// ============================================================
// Render Results
// ============================================================

function renderResults(r) {
  const isGen = r.mode === 'image' || r.mode === 'video';

  if (isGen) {
    el.maxContext.textContent = r.maxCtx >= 10 ? r.maxCtx.toFixed(1) : r.maxCtx.toFixed(2);
    el.maxContext.classList.toggle('over', !r.fits);
  } else if (r.maxCtx > 0) {
    el.maxContext.textContent = r.maxCtx.toLocaleString();
    el.maxContext.classList.remove('over');
  } else {
    el.maxContext.textContent = '0';
    el.maxContext.classList.add('over');
  }

  if (r.bottleneck) {
    el.contextNote.textContent = r.bottleneck;
    el.contextNote.className = (isGen ? r.fits : r.maxCtx > 0) ? 'note' : 'note warn';
  } else {
    el.contextNote.textContent = '';
  }

  const auxVram = r.auxVram || 0;
  const auxRam = r.auxRam || 0;

  renderMemBar('vram', r.gpuModelMem, r.kvVramUsed, r.vramOH, r.vram, auxVram);
  $('vram-model-val').textContent    = fmtGB(r.gpuModelMem);
  $('vram-kv-val').textContent       = fmtGB(r.kvVramUsed);
  $('vram-aux-val').textContent      = fmtGB(auxVram);
  $('vram-overhead-val').textContent  = fmtGB(r.vramOH);
  $('vram-total-val').textContent     = fmtGB(r.totalVram);
  setFreeCell('vram-free-val', r.freeVram);

  renderMemBar('ram', r.cpuModelMem, r.kvRamUsed, r.ramOH, r.ram, auxRam);
  $('ram-model-val').textContent    = fmtGB(r.cpuModelMem);
  $('ram-kv-val').textContent       = fmtGB(r.kvRamUsed);
  $('ram-aux-val').textContent      = fmtGB(auxRam);
  $('ram-overhead-val').textContent  = fmtGB(r.ramOH);
  $('ram-total-val').textContent     = fmtGB(r.totalRam);
  setFreeCell('ram-free-val', r.freeRam);
}

function renderMemBar(prefix, model, kv, overhead, total, aux = 0) {
  const cap = Math.max(total, model + kv + aux + overhead, 0.001);
  $(prefix + '-seg-model').style.width    = pct(model / cap);
  $(prefix + '-seg-kv').style.width       = pct(kv / cap);
  $(prefix + '-seg-aux').style.width      = pct(aux / cap);
  $(prefix + '-seg-overhead').style.width  = pct(overhead / cap);
  $(prefix + '-bar-label').textContent = `${fmtGB(model + kv + aux + overhead)} / ${fmtGB(total)}`;
}

function setFreeCell(id, val) {
  const td = $(id);
  td.textContent = fmtGB(val);
  td.classList.toggle('negative', val < 0);
}

// ============================================================
// llama.cpp Command
// ============================================================

function renderCommand(r) {
  if (r.maxCtx <= 0) {
    el.llamaCmd.textContent = '# Model does not fit with the current settings';
    return;
  }
  const parts = ['llama-server', '-m model.gguf'];
  parts.push(`-ngl ${r.gpuInput}`);
  parts.push(`-c ${r.maxCtx}`);
  if (r.kvQuant !== 'f16') {
    parts.push(`--cache-type-k ${r.kvQuant}`);
    parts.push(`--cache-type-v ${r.kvQuant}`);
  }
  el.llamaCmd.textContent = parts.join(' \\\n  ');
}

// ============================================================
// Utilities
// ============================================================

function int(e) { return parseInt(e.value) || 0; }
function num(e) { return parseFloat(e.value) || 0; }
function pct(v) { return Math.max(0, Math.min(100, v * 100)).toFixed(2) + '%'; }

function fmtGB(gb) {
  if (Math.abs(gb) < 0.01) return '0 GB';
  if (Math.abs(gb) < 1) return (gb * 1024).toFixed(0) + ' MB';
  return gb.toFixed(2) + ' GB';
}

// ============================================================
// Event Binding
// ============================================================

function bindEvents() {
  document.querySelectorAll('.mode-switch button').forEach((btn) => {
    btn.addEventListener('click', () => setMode(btn.dataset.mode));
  });

  // Preset selector
  el.preset.addEventListener('change', () => {
    if (el.preset.value !== 'custom') applyPreset(el.preset.value);
  });

  // Model type changes visibility
  el.modelType.addEventListener('change', () => {
    updateVisibility();
    updateGpuSlider();
    updateExpertsSlider();
    calculate();
  });

  // Layer count changes slider range
  el.numLayers.addEventListener('input', () => {
    updateHybridDerivedFields();
    updateSlidingDerivedFields();
    updateDerivedNotes();
    updateGpuSlider();
    calculate();
  });

  // Hybrid layout updates
  el.fullAttnInterval.addEventListener('input', () => {
    updateHybridDerivedFields();
    updateVisibility();
    calculate();
  });
  [el.linearKeyDim, el.linearValueHeads, el.linearValueDim].forEach((field) => {
    field.addEventListener('input', calculate);
  });

  // Sliding/global attention layout updates
  [el.slidingWindow, el.globalAttnInterval, el.kvSharedLayers].forEach((field) => {
    field.addEventListener('input', () => {
      updateSlidingDerivedFields();
      calculate();
    });
  });
  [el.slidingKvHeads, el.slidingHeadDim, el.globalKvHeads, el.globalHeadDim].forEach((field) => {
    field.addEventListener('input', calculate);
  });

  // Expert sliders
  el.totalExperts.addEventListener('input', () => {
    updateExpertsSlider();
    updateDerivedNotes();
    calculate();
  });
  el.activeExperts.addEventListener('input', updateDerivedNotes);
  el.expertsGpu.addEventListener('input', () => {
    updateExpertsLabel();
    calculate();
  });

  // GPU layers slider
  el.gpuLayers.addEventListener('input', () => {
    updateGpuLabel();
    calculate();
  });

  // VRAM quick buttons
  $('vram-presets').addEventListener('click', (e) => {
    if (e.target.tagName !== 'BUTTON') return;
    el.vram.value = e.target.dataset.value;
    $('vram-presets').querySelectorAll('button').forEach(b => b.classList.remove('active'));
    e.target.classList.add('active');
    calculate();
  });

  // Collapsible advanced section
  document.querySelectorAll('.collapsible-toggle').forEach(toggle => {
    toggle.addEventListener('click', () => {
      const card = toggle.closest('.collapsible');
      const body = card.querySelector('.collapsible-body');
      card.classList.toggle('open');
      body.hidden = !body.hidden;
    });
  });

  // Copy command
  el.copyBtn.addEventListener('click', () => {
    navigator.clipboard.writeText(el.llamaCmd.textContent).then(() => {
      el.copyBtn.textContent = 'Copied!';
      el.copyBtn.classList.add('copied');
      setTimeout(() => {
        el.copyBtn.textContent = 'Copy';
        el.copyBtn.classList.remove('copied');
      }, 1500);
    });
  });

  // Generic recalculate on any remaining input change
  const recalcIds = [
    'vram', 'ram', 'kv-heads', 'head-dim',
    'model-size', 'kv-quant', 'attn-fraction',
    'vram-overhead', 'ram-overhead', 'desired-context',
  ];
  for (const id of recalcIds) {
    $(id).addEventListener('input', calculate);
  }
  el.totalParams.addEventListener('input', () => {
    if (currentMode !== 'llm') el.preset.value = 'custom';
    updateDerivedNotes();
  });
  el.modelSize.addEventListener('input', updateDerivedNotes);
  el.kvQuant.addEventListener('change', calculate);
  el.genPrecision.addEventListener('change', () => {
    updateDerivedNotes();
    calculate();
  });
  el.teOnGpu.addEventListener('change', calculate);
  el.vaeTiling.addEventListener('change', calculate);

  [
    'dit-size', 'te-size', 'vae-size',
    'spatial-compression', 'patch-size', 'temporal-compression',
    'activation-base',
  ].forEach((id) => {
    $(id).addEventListener('input', () => {
      el.preset.value = 'custom';
      updateDerivedNotes();
      calculate();
    });
  });

  ['gen-width', 'gen-height', 'gen-duration', 'gen-fps', 'gen-batch'].forEach((id) => {
    $(id).addEventListener('input', () => {
      syncResPresetButtons();
      updateDerivedNotes();
      calculate();
    });
  });

  $('image-res-presets').addEventListener('click', (e) => {
    if (e.target.tagName !== 'BUTTON') return;
    const size = parseInt(e.target.dataset.size, 10);
    el.genWidth.value = size;
    el.genHeight.value = size;
    syncResPresetButtons();
    calculate();
  });

  $('video-res-presets').addEventListener('click', (e) => {
    if (e.target.tagName !== 'BUTTON') return;
    el.genWidth.value = e.target.dataset.width;
    el.genHeight.value = e.target.dataset.height;
    syncResPresetButtons();
    calculate();
  });
}

function syncResPresetButtons() {
  const width = int(el.genWidth);
  const height = int(el.genHeight);
  $('image-res-presets').querySelectorAll('button').forEach((b) => {
    const size = parseInt(b.dataset.size, 10);
    b.classList.toggle('active', width === size && height === size);
  });
  $('video-res-presets').querySelectorAll('button').forEach((b) => {
    b.classList.toggle('active', width === parseInt(b.dataset.width, 10) && height === parseInt(b.dataset.height, 10));
  });
}

// ============================================================
// Init
// ============================================================

document.addEventListener('DOMContentLoaded', () => {
  populatePresets();
  bindEvents();
  el.preset.value = 'llama31-8b';
  applyPreset('llama31-8b');
});
