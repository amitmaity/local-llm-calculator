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
      { id: 'qwen25-7b',     name: 'Qwen 2.5 7B',         params: 7,    type: 'dense', layers: 28,  kvHeads: 4,  headDim: 128, defaultSize: 4.4  },
      { id: 'qwen25-14b',    name: 'Qwen 2.5 14B',        params: 14,   type: 'dense', layers: 48,  kvHeads: 8,  headDim: 128, defaultSize: 8.3  },
      { id: 'qwen25-32b',    name: 'Qwen 2.5 32B',        params: 32,   type: 'dense', layers: 64,  kvHeads: 8,  headDim: 128, defaultSize: 18.5 },
      { id: 'qwen25-72b',    name: 'Qwen 2.5 72B',        params: 72,   type: 'dense', layers: 80,  kvHeads: 8,  headDim: 128, defaultSize: 41   },
      { id: 'mistral-7b',    name: 'Mistral 7B v0.3',     params: 7.3,  type: 'dense', layers: 32,  kvHeads: 8,  headDim: 128, defaultSize: 4.4  },
      { id: 'mistral-24b',   name: 'Mistral Small 24B',   params: 24,   type: 'dense', layers: 40,  kvHeads: 8,  headDim: 128, defaultSize: 14   },
      { id: 'gemma2-9b',     name: 'Gemma 2 9B',          params: 9,    type: 'dense', layers: 42,  kvHeads: 4,  headDim: 256, defaultSize: 5.5  },
      { id: 'gemma2-27b',    name: 'Gemma 2 27B',         params: 27,   type: 'dense', layers: 46,  kvHeads: 16, headDim: 128, defaultSize: 15.5 },
      { id: 'phi3-mini',     name: 'Phi-3 Mini 3.8B',     params: 3.8,  type: 'dense', layers: 32,  kvHeads: 8,  headDim: 96,  defaultSize: 2.3  },
      { id: 'phi3-medium',   name: 'Phi-3 Medium 14B',    params: 14,   type: 'dense', layers: 40,  kvHeads: 8,  headDim: 128, defaultSize: 8.3  },
      { id: 'commandr-35b',  name: 'Command-R 35B',       params: 35,   type: 'dense', layers: 40,  kvHeads: 8,  headDim: 128, defaultSize: 20   },
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
    group: 'Hybrid Dense (Qwen 3.5)',
    models: [
      { id: 'qwen35-0.8b', name: 'Qwen3.5 0.8B',  params: 0.8, type: 'hybrid-dense', layers: 24, kvHeads: 2, headDim: 256, defaultSize: 0.6,  fullAttnInterval: 4, linearKeyDim: 128, linearValueHeads: 16, linearValueDim: 128 },
      { id: 'qwen35-2b',   name: 'Qwen3.5 2B',    params: 2,   type: 'hybrid-dense', layers: 24, kvHeads: 2, headDim: 256, defaultSize: 1.5,  fullAttnInterval: 4, linearKeyDim: 128, linearValueHeads: 16, linearValueDim: 128 },
      { id: 'qwen35-4b',   name: 'Qwen3.5 4B',    params: 4,   type: 'hybrid-dense', layers: 32, kvHeads: 4, headDim: 256, defaultSize: 2.8,  fullAttnInterval: 4, linearKeyDim: 128, linearValueHeads: 32, linearValueDim: 128 },
      { id: 'qwen35-9b',   name: 'Qwen3.5 9B',    params: 9,   type: 'hybrid-dense', layers: 32, kvHeads: 4, headDim: 256, defaultSize: 5.5,  fullAttnInterval: 4, linearKeyDim: 128, linearValueHeads: 32, linearValueDim: 128 },
      { id: 'qwen35-27b',  name: 'Qwen3.5 27B',   params: 27,  type: 'hybrid-dense', layers: 64, kvHeads: 4, headDim: 256, defaultSize: 16,   fullAttnInterval: 4, linearKeyDim: 128, linearValueHeads: 48, linearValueDim: 128 },
    ],
  },
  {
    group: 'Hybrid MoE',
    models: [
      { id: 'qwen35-35b-a3b',   name: 'Qwen3.5 35B-A3B',          params: 35,  type: 'hybrid-moe', layers: 40, kvHeads: 2, headDim: 256, defaultSize: 20,  fullAttnInterval: 4, linearKeyDim: 128, linearValueHeads: 32, linearValueDim: 128, totalExperts: 256, activeExperts: 9,  attnFraction: 0.04 },
      { id: 'qwen35-122b-a10b', name: 'Qwen3.5 122B-A10B',        params: 122, type: 'hybrid-moe', layers: 48, kvHeads: 2, headDim: 256, defaultSize: 70,  fullAttnInterval: 4, linearKeyDim: 128, linearValueHeads: 64, linearValueDim: 128, totalExperts: 256, activeExperts: 9,  attnFraction: 0.04 },
      { id: 'qwen35-397b-a17b', name: 'Qwen3.5 397B-A17B',        params: 397, type: 'hybrid-moe', layers: 60, kvHeads: 2, headDim: 256, defaultSize: 225, fullAttnInterval: 4, linearKeyDim: 128, linearValueHeads: 64, linearValueDim: 128, totalExperts: 512, activeExperts: 11, attnFraction: 0.03 },
      { id: 'qwen3-coder-next', name: 'Qwen3-Coder-Next 80B-A3B', params: 80,  type: 'hybrid-moe', layers: 48, kvHeads: 2, headDim: 256, defaultSize: 46,  fullAttnInterval: 4, linearKeyDim: 128, linearValueHeads: 32, linearValueDim: 128, totalExperts: 512, activeExperts: 10, attnFraction: 0.04 },
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

function applyPreset(id) {
  const p = findPreset(id);
  if (!p) return;

  el.totalParams.value = p.params;
  el.modelType.value   = p.type;
  el.numLayers.value   = p.layers;
  el.kvHeads.value     = p.kvHeads;
  el.headDim.value     = p.headDim;
  el.modelSize.value   = p.defaultSize;

  const isHybrid = p.type === 'hybrid-dense' || p.type === 'hybrid-moe';
  const isMoe    = p.type === 'moe'          || p.type === 'hybrid-moe';

  if (isHybrid) {
    el.fullAttnInterval.value = p.fullAttnInterval;
    el.linearKeyDim.value = p.linearKeyDim;
    el.linearValueHeads.value = p.linearValueHeads;
    el.linearValueDim.value = p.linearValueDim;
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
  const isHybrid = t === 'hybrid-dense' || t === 'hybrid-moe';
  const isMoe    = t === 'moe'          || t === 'hybrid-moe';

  el.hybridSection.hidden = !isHybrid;
  el.moeSection.hidden    = !isMoe;

  if (isHybrid) {
    updateHybridDerivedFields();
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

function updateDerivedNotes() {
  const params = num(el.totalParams);
  const size = num(el.modelSize);
  if (params > 0 && size > 0) {
    const bitsPerWeight = (size * GB * 8) / (params * 1e9);
    el.paramsNote.textContent = `Current file size implies about ${bitsPerWeight.toFixed(2)} bits/weight. Context estimates use file size, not parameter count.`;
  } else {
    el.paramsNote.textContent = '';
  }

  const isMoe = el.modelType.value === 'moe' || el.modelType.value === 'hybrid-moe';
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

  const isMoe    = modelType === 'moe'          || modelType === 'hybrid-moe';
  const isHybrid = modelType === 'hybrid-dense'  || modelType === 'hybrid-moe';
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
    const expOnGpu     = int(el.expertsGpu)   || totalExp;
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
  let kvGpuLayers, kvCpuLayers;
  if (isHybrid && fullAttnInterval > 1) {
    kvGpuLayers = Math.floor(gpuModelLayers / fullAttnInterval);
    kvCpuLayers = fullAttnLayers - kvGpuLayers;
  } else {
    kvGpuLayers = gpuModelLayers;
    kvCpuLayers = cpuModelLayers;
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

  // KV bytes per token on GPU / CPU side
  const kvBytesPerTokenGpu = kvPerTokenPerLayer * kvGpuLayers;
  const kvBytesPerTokenCpu = kvPerTokenPerLayer * kvCpuLayers;

  // Max context from each constraint
  let maxVram = Infinity, maxRam = Infinity;
  if (kvGpuLayers > 0 && kvBytesPerTokenGpu > 0)
    maxVram = Math.floor((availVram * GB) / kvBytesPerTokenGpu);
  if (kvCpuLayers > 0 && kvBytesPerTokenCpu > 0)
    maxRam = Math.floor((availRam * GB) / kvBytesPerTokenCpu);

  let maxCtx = Math.min(maxVram, maxRam);
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
  const kvVramUsed = (kvBytesPerTokenGpu * maxCtx) / GB;
  const kvRamUsed  = (kvBytesPerTokenCpu * maxCtx) / GB;

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
  renderReverse(kvBytesPerTokenGpu, kvBytesPerTokenCpu, gpuModelMem, cpuModelMem, effectiveVramOH, effectiveRamOH, vram, ram);
  renderCommand(results);
}

// ============================================================
// Reverse Mode
// ============================================================

function renderReverse(kvBpTGpu, kvBpTCpu, gpuMem, cpuMem, vramOH, ramOH, vram, ram) {
  const ctx = parseInt(el.desiredCtx.value) || 0;
  if (ctx <= 0) { el.reverseSection.hidden = true; return; }

  el.reverseSection.hidden = false;
  const kvVram = (kvBpTGpu * ctx) / GB;
  const kvRam  = (kvBpTCpu * ctx) / GB;
  const needVram = gpuMem + kvVram + vramOH;
  const needRam  = cpuMem + kvRam  + ramOH;
  const fitsVram = needVram <= vram || kvBpTGpu === 0;
  const fitsRam  = needRam  <= ram  || kvBpTCpu === 0;
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
