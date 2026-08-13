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
    group: 'Hybrid Dense (Qwen 3.5/3.6)',
    models: [
      { id: 'qwen35-0.8b', name: 'Qwen3.5 0.8B',  params: 0.8, type: 'hybrid-dense', layers: 24, kvHeads: 2, headDim: 256, defaultSize: 0.6,  fullAttnInterval: 4, linearKeyDim: 128, linearValueHeads: 16, linearValueDim: 128 },
      { id: 'qwen35-2b',   name: 'Qwen3.5 2B',    params: 2,   type: 'hybrid-dense', layers: 24, kvHeads: 2, headDim: 256, defaultSize: 1.5,  fullAttnInterval: 4, linearKeyDim: 128, linearValueHeads: 16, linearValueDim: 128 },
      { id: 'qwen35-4b',   name: 'Qwen3.5 4B',    params: 4,   type: 'hybrid-dense', layers: 32, kvHeads: 4, headDim: 256, defaultSize: 2.8,  fullAttnInterval: 4, linearKeyDim: 128, linearValueHeads: 32, linearValueDim: 128 },
      { id: 'qwen35-9b',   name: 'Qwen3.5 9B',    params: 9,   type: 'hybrid-dense', layers: 32, kvHeads: 4, headDim: 256, defaultSize: 5.5,  fullAttnInterval: 4, linearKeyDim: 128, linearValueHeads: 32, linearValueDim: 128 },
      { id: 'qwen35-27b',  name: 'Qwen3.5 27B',   params: 27,  type: 'hybrid-dense', layers: 64, kvHeads: 4, headDim: 256, defaultSize: 16,   fullAttnInterval: 4, linearKeyDim: 128, linearValueHeads: 48, linearValueDim: 128 },
      { id: 'qwen36-27b',  name: 'Qwen3.6 27B',   params: 27,  type: 'hybrid-dense', layers: 64, kvHeads: 4, headDim: 256, defaultSize: 16,   fullAttnInterval: 4, linearKeyDim: 128, linearValueHeads: 48, linearValueDim: 128 },
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
};

// ============================================================
// Preset Management
// ============================================================

function populatePresets() {
  const sel = el.preset;
  const custom = document.createElement('option');
  custom.value = 'custom';
  custom.textContent = '-- Custom --';
  sel.appendChild(custom);

  for (const group of MODEL_PRESETS) {
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
  for (const g of MODEL_PRESETS)
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

function calculate() {
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
  // Big number
  if (r.maxCtx > 0) {
    el.maxContext.textContent = r.maxCtx.toLocaleString();
    el.maxContext.classList.remove('over');
  } else {
    el.maxContext.textContent = '0';
    el.maxContext.classList.add('over');
  }

  // Note
  if (r.bottleneck) {
    el.contextNote.textContent = r.bottleneck;
    el.contextNote.className = r.maxCtx > 0 ? 'note' : 'note warn';
  } else {
    el.contextNote.textContent = '';
  }

  // VRAM bar & table
  renderMemBar('vram', r.gpuModelMem, r.kvVramUsed, r.vramOH, r.vram);
  $('vram-model-val').textContent    = fmtGB(r.gpuModelMem);
  $('vram-kv-val').textContent       = fmtGB(r.kvVramUsed);
  $('vram-overhead-val').textContent  = fmtGB(r.vramOH);
  $('vram-total-val').textContent     = fmtGB(r.totalVram);
  setFreeCell('vram-free-val', r.freeVram);

  // RAM bar & table
  renderMemBar('ram', r.cpuModelMem, r.kvRamUsed, r.ramOH, r.ram);
  $('ram-model-val').textContent    = fmtGB(r.cpuModelMem);
  $('ram-kv-val').textContent       = fmtGB(r.kvRamUsed);
  $('ram-overhead-val').textContent  = fmtGB(r.ramOH);
  $('ram-total-val').textContent     = fmtGB(r.totalRam);
  setFreeCell('ram-free-val', r.freeRam);
}

function renderMemBar(prefix, model, kv, overhead, total) {
  const cap = Math.max(total, model + kv + overhead, 0.001);
  $(prefix + '-seg-model').style.width    = pct(model / cap);
  $(prefix + '-seg-kv').style.width       = pct(kv / cap);
  $(prefix + '-seg-overhead').style.width  = pct(overhead / cap);
  $(prefix + '-bar-label').textContent = `${fmtGB(model + kv + overhead)} / ${fmtGB(total)}`;
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
  el.totalParams.addEventListener('input', updateDerivedNotes);
  el.modelSize.addEventListener('input', updateDerivedNotes);
  // 'change' for selects that don't fire input
  el.kvQuant.addEventListener('change', calculate);
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
