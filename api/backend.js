
/* ══════════════════════════════════════════════════════════
   /api/backend — one Vercel serverless function, two jobs

   The frontend sends { type: 'chat', messages: [...] } or
   { type: 'search', query: '...' } to this same endpoint.
   This is the only file that touches your real API keys.
   They come from environment variables set in Vercel, never
   from this file's source code.

   Required environment variables (Vercel → Settings →
   Environment Variables):
     GROQ_KEY       (free)
     OPENROUTER_KEY (free)
     NVIDIA_KEY     (free)
     GOOGLE_KEY     (free; also powers image analysis and live web search)
     CF_ACCOUNT_ID + CF_API_TOKEN  (Cloudflare Workers AI: the Flash picture engine)
     HF_TOKEN       (Hugging Face: the Studio picture engine)
     ANTHROPIC_KEY  (optional, PAID: Anthropic has no free API tier)
     TAVILY_KEY
     APP_SECRET     (must match the APP_SECRET in index.html)
   ══════════════════════════════════════════════════════════ */

module.exports = async function handler(req, res) {
  if (req.method !== 'POST') {
    res.status(405).json({ error: 'method_not_allowed' });
    return;
  }

  const providedSecret = req.headers['x-chaturvedi-key'];
  if (process.env.APP_SECRET && providedSecret !== process.env.APP_SECRET) {
    res.status(401).json({ error: 'unauthorized' });
    return;
  }

  const body = req.body || {};

  if (body.type === 'search') {
    await handleSearch(body, res);
  } else if (body.type === 'vision') {
    await handleVision(body, res);
  } else if (body.type === 'image') {
    await handleImage(body, res);
  } else {
    await handleChat(body, res);
  }
};

/* ── CHAT ─────────────────────────────────────────────────── */
/* Self-updating providers.
   Instead of hard-coding model names (which is what broke Gemini),
   each provider's own "list models" endpoint is asked what exists
   right now, and the best match is picked by preference. The result
   is cached for a few hours. If a model ever returns "not found",
   the list is refreshed immediately and the next candidate is used.
   Hard-coded names below are only a last-resort fallback. */

const MODEL_CACHE_MS = 6 * 60 * 60 * 1000;
const modelCache = {}; // providerId -> { at, models: [...] }

const PROVIDERS = [
  {
    id: 'groq', env: 'GROQ_KEY', kind: 'openai',
    url: 'https://api.groq.com/openai/v1/chat/completions',
    modelsUrl: 'https://api.groq.com/openai/v1/models',
    fallback: ['llama-3.3-70b-versatile'],
    prefs: [/llama-3\.3-70b/, /gpt-oss-120b/, /llama.*70b/, /qwen/, /llama/],
    exclude: /whisper|guard|tts|embed|orpheus|playai|compound|safeguard/i
  },
  {
    id: 'openrouter', env: 'OPENROUTER_KEY', kind: 'openai',
    url: 'https://openrouter.ai/api/v1/chat/completions',
    modelsUrl: 'https://openrouter.ai/api/v1/models',
    onlyFree: true,
    fallback: ['meta-llama/llama-3.3-70b-instruct:free'],
    prefs: [/llama-3\.3-70b-instruct/, /gpt-oss-120b/, /qwen.*(235b|72b|32b)/, /deepseek.*(v3|chat)/, /llama.*instruct/, /gemma.*(27b|it)/, /./],
    exclude: /vision|image|audio|embed|guard|moderation|-vl-|\bvl\b/i
  },
  {
    id: 'nvidia', env: 'NVIDIA_KEY', kind: 'openai',
    url: 'https://integrate.api.nvidia.com/v1/chat/completions',
    modelsUrl: 'https://integrate.api.nvidia.com/v1/models',
    fallback: ['meta/llama-3.3-70b-instruct'],
    prefs: [/llama-3\.3-70b-instruct/, /llama-3\.1-70b-instruct/, /nemotron.*(70b|super)/, /llama.*instruct/, /mistral.*instruct/],
    exclude: /embed|rerank|guard|vision|vlm|clip|parse|reward|safety|riva|vila|neva|paligemma|fuyu|cosmos|deplot|kosmos|sdxl|stable|flux|nv-/i
  },
  {
    id: 'google', env: 'GOOGLE_KEY', kind: 'gemini',
    modelsUrl: 'https://generativelanguage.googleapis.com/v1beta/models',
    fallback: ['gemini-flash-latest'],
    prefs: [/^gemini-flash-latest$/, /^gemini-[\d.]+-flash$/, /^gemini-[\d.]+-flash-lite$/, /flash/],
    exclude: /image|live|tts|audio|embed|thinking|vision|robotics|computer|imagen|veo|gemma|transcribe|omni|native|exp|banana/i
  },
  {
    /* Optional and PAID (Anthropic has no free API tier).
       Only used if ANTHROPIC_KEY is set, and always tried last. */
    id: 'anthropic', env: 'ANTHROPIC_KEY', kind: 'anthropic',
    url: 'https://api.anthropic.com/v1/messages',
    modelsUrl: 'https://api.anthropic.com/v1/models',
    fallback: ['claude-haiku-5-5'],
    prefs: [/haiku/, /sonnet/, /opus/],
    exclude: /^$/
  }
];

/* ── SWAROOPS (modes) ─────────────────────────────────────
   Spark  = fastest: small, quick models, short answers.
   Stride = balanced everyday default.
   Summit = deepest: strongest models, step-by-step reasoning,
            long answers, longer time allowed.
   They change the real model choice, provider order, answer
   length budget, temperature and instructions: not just a label. */
const MODES = {
  spark: {
    tier: 'fast', maxTokens: 700, temp: 0.5, timeoutMs: 12000,
    order: ['groq', 'google', 'openrouter', 'nvidia', 'anthropic'],
    hint: 'Mode: Spark (fast). Lead with the answer and keep it brief, under about 120 words, unless the task genuinely needs more (code, steps, files). No preamble and no recap.'
  },
  stride: {
    tier: 'balanced', maxTokens: 1500, temp: 0.7, timeoutMs: 20000,
    order: ['groq', 'openrouter', 'nvidia', 'google', 'anthropic'],
    hint: ''
  },
  summit: {
    tier: 'deep', maxTokens: 3500, temp: 0.4, timeoutMs: 40000,
    order: ['anthropic', 'google', 'groq', 'nvidia', 'openrouter'],
    hint: 'Mode: Summit (deep). Work through the problem carefully before you answer: consider edge cases and alternative explanations, check numbers and logic, and give a thorough, well-structured answer. Say clearly what you are unsure about.'
  }
};

const TIERS = {
  groq: {
    fast: { prefs: [/llama-3\.1-8b/, /\b8b\b/, /gemma/, /llama/], fallback: ['llama-3.1-8b-instant'] },
    deep: { prefs: [/gpt-oss-120b/, /llama-3\.3-70b/, /llama-4/, /qwen.*32b/, /llama.*70b/], fallback: ['openai/gpt-oss-120b', 'llama-3.3-70b-versatile'] }
  },
  openrouter: {
    fast: { prefs: [/llama-3\.[12]-(1b|3b|8b)/, /gemma.*(1b|3n|4b|9b)/, /mistral.*(7b|nemo|small)/, /./], fallback: ['meta-llama/llama-3.2-3b-instruct:free'] },
    deep: { prefs: [/gpt-oss-120b/, /qwen.*(235b|480b)/, /deepseek.*(r1|v3)/, /llama-3\.1-405b/, /llama-3\.3-70b/, /llama.*70b/], fallback: ['openai/gpt-oss-120b:free', 'meta-llama/llama-3.3-70b-instruct:free'] }
  },
  nvidia: {
    fast: { prefs: [/llama-3\.1-8b-instruct/, /llama.*(8b|3b)-instruct/, /mistral.*(7b|nemo)/, /nemotron.*nano/], fallback: ['meta/llama-3.1-8b-instruct'] },
    deep: { prefs: [/llama-3\.1-405b-instruct/, /nemotron.*(ultra|super)/, /deepseek.*(r1|v3)/, /qwen.*(235b|397b|72b)/, /llama-3\.3-70b-instruct/], fallback: ['meta/llama-3.1-405b-instruct', 'meta/llama-3.3-70b-instruct'] }
  },
  google: {
    fast: { prefs: [/^gemini-flash-lite-latest$/, /^gemini-[\d.]+-flash-lite$/, /^gemini-flash-latest$/, /flash/], fallback: ['gemini-flash-lite-latest', 'gemini-flash-latest'] },
    deep: { prefs: [/^gemini-pro-latest$/, /^gemini-[\d.]+-pro$/, /pro/], fallback: ['gemini-pro-latest', 'gemini-flash-latest'] }
  },
  anthropic: {
    fast: { prefs: [/haiku/], fallback: ['claude-haiku-5-5'] },
    deep: { prefs: [/opus/, /sonnet/], fallback: ['claude-opus-5-5', 'claude-sonnet-5-5'] }
  }
};

function providersForMode(modeKey) {
  const M = MODES[modeKey] || MODES.stride;
  const out = [];
  M.order.forEach(function (id) {
    const base = PROVIDERS.filter(function (p) { return p.id === id; })[0];
    if (!base) return;
    const t = (M.tier === 'balanced' ? {} : ((TIERS[id] || {})[M.tier] || {}));
    out.push(Object.assign({}, base, {
      prefs: t.prefs || base.prefs,
      fallback: t.fallback || base.fallback,
      cacheKey: id + ':' + (MODES[modeKey] ? modeKey : 'stride'),
      maxTokens: M.maxTokens,
      temp: M.temp,
      timeoutMs: M.timeoutMs
    }));
  });
  return out;
}

async function tryProviders(list, caller) {
  const reasons = [];
  let anyKey = false;

  for (let i = 0; i < list.length; i++) {
    const p = list[i];
    const key = process.env[p.env];
    if (!key) continue;
    anyKey = true;

    const tried = {};
    let stop = false;

    for (let round = 0; round < 2 && !stop; round++) {
      const all = await getCandidates(p, key, round === 1);
      const cands = all.filter(function (m) { return !tried[m]; });
      if (!cands.length) break;

      let sawModelError = false;
      for (let j = 0; j < cands.length; j++) {
        const model = cands[j];
        tried[model] = true;
        try {
          const reply = await withTimeout(caller(p, key, model), p.timeoutMs || 20000);
          if (reply && String(reply).trim()) {
            rememberWorking(p.cacheKey || p.id, model);
            return { reply: reply, provider: p.id, model: model };
          }
          reasons.push(p.id + '_empty_reply');
        } catch (e) {
          reasons.push(String((e && e.message) || e));
          const s = e && e.status;
          if (s === 401 || s === 403) { stop = true; break; }          // bad key: skip provider
          if (s === 404 || s === 400 || s === 410) { sawModelError = true; continue; } // model gone: next candidate
          stop = true; break;                                           // rate limit / outage: next provider
        }
      }
      if (!sawModelError) break; // only refresh the list when a model was "not found"
    }
  }

  return {
    error: 'all_providers_failed',
    detail: anyKey ? (reasons.join('; ') || 'unknown') : 'no_api_keys_set_in_vercel'
  };
}

async function handleChat(body, res) {
  const messages = body.messages;
  if (!Array.isArray(messages) || messages.length === 0) {
    res.status(400).json({ error: 'bad_request', detail: 'messages array required' });
    return;
  }
  const modeKey = MODES[body.mode] ? body.mode : 'stride';
  // Fast, small models write poor documents, so Spark steps up to Stride when creating files.
  const effKey = (body.purpose === 'create' && modeKey === 'spark') ? 'stride' : modeKey;
  const M = MODES[effKey];
  const providers = providersForMode(effKey);

  const mt = Number(body.maxTokens);
  if (isFinite(mt) && mt >= 200) {
    providers.forEach(function (p) { p.maxTokens = Math.min(4000, Math.round(mt)); });
  }

  // The mode's instruction is skipped for file creation, where the reply must be pure JSON.
  const msgs = messages.slice();
  if (M.hint && body.purpose !== 'create' && body.purpose !== 'utility') {
    const i = msgs.findIndex(function (m) { return m.role === 'system'; });
    if (i >= 0) msgs[i] = { role: 'system', content: String(msgs[i].content) + '\n\n' + M.hint };
    else msgs.unshift({ role: 'system', content: M.hint });
  }

  const t0 = Date.now();
  const out = await tryProviders(providers, function (p, key, model) {
    return callProvider(p, key, model, msgs);
  });
  if (out.reply) {
    res.status(200).json({ reply: out.reply, meta: { mode: modeKey, provider: out.provider, model: out.model, ms: Date.now() - t0 } });
    return;
  }
  res.status(502).json(out);
}

/* ── PICTURES: two engines, genuinely different ───────────
   Flash  = Cloudflare Workers AI (FLUX schnell class): a draft in seconds.
   Studio = Hugging Face (FLUX.1 dev / SD 3.5 class): slower, finer detail.
   Each falls back to the other if it is not set up, out of credit or down.
   Model lists are discovered live, like the chat models, so names can change. */
const IMG_MODES = {
  flash:  { order: ['cloudflare', 'huggingface'], tier: 'flash',  timeoutMs: 26000 },
  studio: { order: ['huggingface', 'cloudflare'], tier: 'studio', timeoutMs: 45000 }
};
const IMG_PROVIDERS = {
  cloudflare: {
    exclude: /img2img|inpaint|flux-2|embed|upscal/i,
    tiers: {
      flash:  { prefs: [/flux-1-schnell/, /lightning/, /dreamshaper.*lcm/, /./], fallback: ['@cf/black-forest-labs/flux-1-schnell'] },
      studio: { prefs: [/stable-diffusion-xl-base/, /phoenix/, /lucid/, /flux-1-schnell/], fallback: ['@cf/stabilityai/stable-diffusion-xl-base-1.0', '@cf/black-forest-labs/flux-1-schnell'] }
    }
  },
  huggingface: {
    exclude: /lora|controlnet|inpaint|refiner|ip-adapter|pixart|edit/i,
    tiers: {
      flash:  { prefs: [/FLUX\.1-schnell/i, /turbo|lightning/i, /stable-diffusion-xl-base/i], fallback: ['black-forest-labs/FLUX.1-schnell'] },
      studio: { prefs: [/FLUX\.1-dev/i, /stable-diffusion-3\.5-large/i, /FLUX\.1-Krea/i, /stable-diffusion-3-medium/i, /stable-diffusion-xl-base/i, /FLUX\.1-schnell/i], fallback: ['black-forest-labs/FLUX.1-dev', 'stabilityai/stable-diffusion-3.5-large', 'black-forest-labs/FLUX.1-schnell'] }
    }
  }
};
const imgCache = {};

function imgCreds(id) {
  if (id === 'cloudflare') return process.env.CF_ACCOUNT_ID && process.env.CF_API_TOKEN ? { acc: process.env.CF_ACCOUNT_ID, tok: process.env.CF_API_TOKEN } : null;
  if (id === 'huggingface') return process.env.HF_TOKEN ? { tok: process.env.HF_TOKEN } : null;
  return null;
}

async function imgDiscover(id, cred) {
  if (id === 'cloudflare') {
    const r = await withTimeout(fetch('https://api.cloudflare.com/client/v4/accounts/' + cred.acc + '/ai/models/search?per_page=100', { headers: { 'Authorization': 'Bearer ' + cred.tok } }), 5000);
    if (!r.ok) throw httpError('cloudflare_models', r.status);
    const data = await r.json();
    return (data.result || []).filter(function (m) {
      const t = m.task && m.task.name ? String(m.task.name) : '';
      return /text.?to.?image/i.test(t) || (!t && /flux|stable-diffusion|dreamshaper|lightning|phoenix|lucid/i.test(m.name || ''));
    }).map(function (m) { return m.name; });
  }
  const r = await withTimeout(fetch('https://huggingface.co/api/models?pipeline_tag=text-to-image&inference_provider=hf-inference&sort=likes&direction=-1&limit=40', { headers: { 'Authorization': 'Bearer ' + cred.tok } }), 5000);
  if (!r.ok) throw httpError('huggingface_models', r.status);
  return (await r.json()).map(function (m) { return m.id; }).filter(Boolean);
}

async function imgCandidates(id, cred, tier, force) {
  const key = id + ':' + tier;
  const c = imgCache[key];
  if (!force && c && Date.now() - c.at < MODEL_CACHE_MS) return c.models;
  const P = IMG_PROVIDERS[id], T = P.tiers[tier];
  let found = [];
  try { found = rankModels(await imgDiscover(id, cred), { prefs: T.prefs, exclude: P.exclude }); } catch (e) { found = []; }
  let models = found.slice(0, 3);
  if (!models.length && c) models = c.models;
  T.fallback.forEach(function (m) { if (models.indexOf(m) === -1) models.push(m); });
  imgCache[key] = { at: Date.now(), models: models };
  return models;
}

async function imgRun(id, cred, model, prompt, w, h) {
  if (id === 'cloudflare') {
    const body = /flux-1-schnell/.test(model) ? { prompt: prompt, steps: 4 }
      : { prompt: prompt, width: w, height: h, num_steps: /lightning/.test(model) ? 4 : 20 };
    const r = await fetch('https://api.cloudflare.com/client/v4/accounts/' + cred.acc + '/ai/run/' + model, {
      method: 'POST', headers: { 'Authorization': 'Bearer ' + cred.tok, 'Content-Type': 'application/json' }, body: JSON.stringify(body)
    });
    if (!r.ok) throw httpError('cloudflare', r.status);
    const ct = r.headers.get('content-type') || '';
    if (ct.indexOf('image/') === 0) return { mime: ct.split(';')[0], b64: Buffer.from(await r.arrayBuffer()).toString('base64') };
    const j = await r.json();
    const b64 = j && j.result && j.result.image;
    if (!b64) throw new Error('cloudflare_no_image');
    return { mime: 'image/jpeg', b64: b64 };
  }
  const side = function (v) { return Math.min(896, v); };
  const params = { width: side(w), height: side(h) };
  if (/schnell/i.test(model)) params.num_inference_steps = 4;
  const r = await fetch('https://router.huggingface.co/hf-inference/models/' + model, {
    method: 'POST',
    headers: { 'Authorization': 'Bearer ' + cred.tok, 'Content-Type': 'application/json', 'Accept': 'image/jpeg', 'x-wait-for-model': 'true' },
    body: JSON.stringify({ inputs: prompt, parameters: params })
  });
  if (!r.ok) throw httpError('huggingface', r.status);
  const ct = r.headers.get('content-type') || 'image/jpeg';
  if (ct.indexOf('image/') !== 0) throw new Error('huggingface_not_image');
  return { mime: ct.split(';')[0], b64: Buffer.from(await r.arrayBuffer()).toString('base64') };
}

async function handleImage(body, res) {
  const prompt = typeof body.prompt === 'string' ? body.prompt.trim().slice(0, 600) : '';
  if (prompt.length < 3) { res.status(400).json({ error: 'bad_request', detail: 'prompt required' }); return; }
  const modeKey = IMG_MODES[body.mode] ? body.mode : 'flash';
  const M = IMG_MODES[modeKey];
  const clampDim = function (v) { v = Number(v); if (!isFinite(v)) v = 1024; return Math.max(256, Math.min(1024, Math.round(v / 16) * 16)); };
  const w = clampDim(body.width), h = clampDim(body.height);

  const t0 = Date.now();
  const reasons = [];
  let anyKey = false;

  for (let i = 0; i < M.order.length; i++) {
    const id = M.order[i];
    const cred = imgCreds(id);
    if (!cred) continue;
    anyKey = true;

    const tried = {};
    let stop = false;
    for (let round = 0; round < 2 && !stop; round++) {
      const cands = (await imgCandidates(id, cred, M.tier, round === 1)).filter(function (m) { return !tried[m]; }).slice(0, 3);
      if (!cands.length) break;
      let sawModelError = false;
      for (let j = 0; j < cands.length; j++) {
        if (Date.now() - t0 > 52000) { stop = true; break; }
        const model = cands[j];
        tried[model] = true;
        try {
          const out = await withTimeout(imgRun(id, cred, model, prompt, w, h), M.timeoutMs);
          if (out.b64.length > 4300000) throw new Error(id + '_image_too_large');
          res.status(200).json({ image: 'data:' + out.mime + ';base64,' + out.b64, meta: { mode: modeKey, provider: id, model: model, ms: Date.now() - t0, fellBack: i > 0 } });
          return;
        } catch (e) {
          reasons.push(String((e && e.message) || e));
          const s = e && e.status;
          if (s === 401 || s === 402 || s === 403) { stop = true; break; }               // bad key or no credit: next engine
          if (s === 400 || s === 404 || s === 410 || s === 422) { sawModelError = true; continue; } // model not usable: next model
          stop = true; break;                                                           // busy / slow / outage: next engine
        }
      }
      if (!sawModelError) break;
    }
  }
  res.status(502).json({ error: 'all_image_engines_failed', detail: anyKey ? (reasons.join('; ') || 'unknown') : 'no_image_keys_set (add CF_ACCOUNT_ID + CF_API_TOKEN and/or HF_TOKEN)' });
}

/* ── VISION (image analysis) ──────────────────────────────── */
/* Same self-updating idea, but only providers that can see images,
   and with their own model preferences (cached separately). */
const VISION_SETTINGS = {
  google: {},
  anthropic: {},
  groq: {
    prefs: [/llama-4/, /scout|maverick/, /vision/],
    fallback: ['meta-llama/llama-4-scout-17b-16e-instruct']
  },
  openrouter: {
    prefs: [/gemma-3.*it/, /llama-4/, /qwen.*vl/, /vision|-vl-|\bvl\b/, /mistral.*(small|3)/],
    exclude: /embed|audio|guard|moderation/i,
    fallback: ['google/gemma-3-27b-it:free']
  }
};

function visionProviders() {
  const out = [];
  ['google', 'groq', 'openrouter', 'anthropic'].forEach(function (id) {
    const base = PROVIDERS.filter(function (p) { return p.id === id; })[0];
    const s = VISION_SETTINGS[id];
    out.push(Object.assign({}, base, {
      prefs: s.prefs || base.prefs,
      exclude: s.exclude || base.exclude,
      fallback: s.fallback || base.fallback,
      cacheKey: id + ':vision'
    }));
  });
  return out;
}

async function handleVision(body, res) {
  const prompt = typeof body.prompt === 'string' ? body.prompt : '';
  const system = typeof body.system === 'string' ? body.system : '';
  const images = Array.isArray(body.images) ? body.images.slice(0, 3) : [];

  const parsed = [];
  images.forEach(function (d) {
    const m = /^data:(image\/(?:png|jpeg|jpg|webp|gif));base64,([A-Za-z0-9+\/=]+)$/.exec(String(d));
    if (m) parsed.push({ mime: m[1] === 'image/jpg' ? 'image/jpeg' : m[1], data: m[2], url: String(d) });
  });
  if (!prompt || !parsed.length) {
    res.status(400).json({ error: 'bad_request', detail: 'prompt and at least one image required' });
    return;
  }

  const out = await tryProviders(visionProviders(), function (p, key, model) {
    return callVision(p, key, model, system, prompt, parsed);
  });
  if (out.reply) { res.status(200).json({ reply: out.reply }); return; }
  res.status(502).json(out);
}

async function callVision(p, key, model, system, prompt, imgs) {
  if (p.kind === 'gemini') {
    const parts = [{ text: (system ? system + '\n\n' : '') + prompt }];
    imgs.forEach(function (im) { parts.push({ inline_data: { mime_type: im.mime, data: im.data } }); });
    const url = 'https://generativelanguage.googleapis.com/v1beta/models/' + model + ':generateContent?key=' + encodeURIComponent(key);
    const r = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ contents: [{ role: 'user', parts: parts }], generationConfig: { temperature: 0.5, maxOutputTokens: 1500 } })
    });
    if (!r.ok) throw httpError('google', r.status);
    const data = await r.json();
    return data.candidates && data.candidates[0] && data.candidates[0].content && data.candidates[0].content.parts[0].text;
  }

  if (p.kind === 'anthropic') {
    const content = imgs.map(function (im) {
      return { type: 'image', source: { type: 'base64', media_type: im.mime, data: im.data } };
    });
    content.push({ type: 'text', text: prompt });
    const r = await fetch(p.url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'x-api-key': key, 'anthropic-version': '2023-06-01' },
      body: JSON.stringify({ model: model, max_tokens: 1500, system: system || undefined, messages: [{ role: 'user', content: content }] })
    });
    if (!r.ok) throw httpError(p.id, r.status);
    const data = await r.json();
    return data.content && data.content[0] && data.content[0].text;
  }

  const userContent = [{ type: 'text', text: prompt }];
  imgs.forEach(function (im) { userContent.push({ type: 'image_url', image_url: { url: im.url } }); });
  const msgs = [];
  if (system) msgs.push({ role: 'system', content: system });
  msgs.push({ role: 'user', content: userContent });
  const headers = { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key };
  if (p.id === 'openrouter') { headers['HTTP-Referer'] = 'https://chaturvedi.app'; headers['X-Title'] = 'Trishul'; }
  const r = await fetch(p.url, { method: 'POST', headers: headers, body: JSON.stringify({ model: model, messages: msgs, max_tokens: 1500 }) });
  if (!r.ok) throw httpError(p.id, r.status);
  const data = await r.json();
  return data.choices && data.choices[0] && data.choices[0].message && data.choices[0].message.content;
}

function withTimeout(promise, ms) {
  return Promise.race([
    promise,
    new Promise(function (_, reject) {
      setTimeout(function () { reject(new Error('timeout')); }, ms);
    })
  ]);
}

function httpError(id, status) {
  const e = new Error(id + '_http_' + status);
  e.status = status;
  return e;
}

/* ── model discovery ──────────────────────────────────────── */
function rememberWorking(id, model) {
  const c = modelCache[id];
  if (!c) { modelCache[id] = { at: Date.now(), models: [model] }; return; }
  c.models = [model].concat(c.models.filter(function (m) { return m !== model; }));
}

async function getCandidates(p, key, forceRefresh) {
  const ck = p.cacheKey || p.id;
  const c = modelCache[ck];
  if (!forceRefresh && c && Date.now() - c.at < MODEL_CACHE_MS) return c.models;

  let found = [];
  try {
    found = rankModels(await listModels(p, key), p);
  } catch (e) {
    found = []; // discovery failed: fall back to the last known list or the hard-coded names
  }

  let models = found.slice(0, 3);
  if (!models.length && c) models = c.models;
  p.fallback.forEach(function (m) { if (models.indexOf(m) === -1) models.push(m); });

  modelCache[ck] = { at: Date.now(), models: models };
  return models;
}

async function listModels(p, key) {
  let url = p.modelsUrl;
  let headers = {};
  if (p.kind === 'openai') headers = { 'Authorization': 'Bearer ' + key };
  if (p.kind === 'gemini') url += '?pageSize=200&key=' + encodeURIComponent(key);
  if (p.kind === 'anthropic') headers = { 'x-api-key': key, 'anthropic-version': '2023-06-01' };

  const r = await withTimeout(fetch(url, { headers: headers }), 4000);
  if (!r.ok) throw httpError(p.id + '_models', r.status);
  const data = await r.json();

  if (p.kind === 'gemini') {
    return (data.models || [])
      .filter(function (m) { return (m.supportedGenerationMethods || []).indexOf('generateContent') !== -1; })
      .map(function (m) { return String(m.name || '').replace(/^models\//, ''); });
  }

  let list = data.data || [];
  if (p.onlyFree) {
    list = list.filter(function (m) { return /:free$/.test(m.id || ''); });
  }
  return list.map(function (m) { return m.id; }).filter(Boolean);
}

function rankModels(ids, p) {
  const ok = ids.filter(function (id) { return !p.exclude.test(id); });
  const out = [];
  p.prefs.forEach(function (re) {
    ok.filter(function (id) { return re.test(id) && out.indexOf(id) === -1; })
      .sort().reverse() // newest-looking version first
      .forEach(function (id) { out.push(id); });
  });
  return out;
}

/* ── calling each kind of provider ────────────────────────── */
async function callProvider(p, key, model, messages) {
  if (p.kind === 'gemini') return callGemini(messages, key, model, p);
  if (p.kind === 'anthropic') return callAnthropic(p, messages, key, model);
  return callOpenAICompatible(p, key, model, messages);
}

async function callOpenAICompatible(p, key, model, messages) {
  const reqBody = { model: model, messages: messages, max_tokens: p.maxTokens || 1500 };
  if (model.indexOf(':free') === -1) reqBody.temperature = p.temp == null ? 0.7 : p.temp;

  const headers = { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key };
  if (p.id === 'openrouter') {
    headers['HTTP-Referer'] = 'https://chaturvedi.app';
    headers['X-Title'] = 'Trishul';
  }

  const r = await fetch(p.url, { method: 'POST', headers: headers, body: JSON.stringify(reqBody) });
  if (!r.ok) throw httpError(p.id, r.status);
  const data = await r.json();
  return data.choices && data.choices[0] && data.choices[0].message && data.choices[0].message.content;
}

async function callAnthropic(p, messages, key, model) {
  const system = messages.filter(function (m) { return m.role === 'system'; })
    .map(function (m) { return m.content; }).join('\n\n');

  const turns = [];
  messages.filter(function (m) { return m.role !== 'system'; }).forEach(function (m) {
    const role = m.role === 'assistant' ? 'assistant' : 'user';
    const text = m.content || '';
    if (turns.length && turns[turns.length - 1].role === role) {
      turns[turns.length - 1].content += '\n' + text;
    } else {
      turns.push({ role: role, content: text });
    }
  });
  if (!turns.length || turns[0].role !== 'user') turns.unshift({ role: 'user', content: '(start)' });

  const r = await fetch(p.url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'x-api-key': key, 'anthropic-version': '2023-06-01' },
    body: JSON.stringify({ model: model, max_tokens: p.maxTokens || 1500, system: system || undefined, messages: turns })
  });
  if (!r.ok) throw httpError(p.id, r.status);
  const data = await r.json();
  return data.content && data.content[0] && data.content[0].text;
}

async function callGemini(messages, key, model, p) {
  const sys = messages.find(function (m) { return m.role === 'system'; });
  const rest = messages.filter(function (m) { return m.role !== 'system'; });

  const contents = [];
  if (sys) {
    contents.push({ role: 'user', parts: [{ text: 'Instructions: ' + sys.content }] });
    contents.push({ role: 'model', parts: [{ text: 'Understood.' }] });
  }
  rest.forEach(function (m) {
    contents.push({ role: m.role === 'assistant' ? 'model' : 'user', parts: [{ text: m.content || '' }] });
  });

  const merged = [];
  contents.forEach(function (c) {
    if (merged.length && merged[merged.length - 1].role === c.role) {
      merged[merged.length - 1].parts[0].text += '\n' + c.parts[0].text;
    } else {
      merged.push(c);
    }
  });

  const url = 'https://generativelanguage.googleapis.com/v1beta/models/' + model + ':generateContent?key=' + encodeURIComponent(key);
  const r = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ contents: merged, generationConfig: { temperature: p && p.temp != null ? p.temp : 0.7, maxOutputTokens: (p && p.maxTokens) || 1500 } })
  });
  if (!r.ok) throw httpError('google', r.status);
  const data = await r.json();
  return data.candidates && data.candidates[0] && data.candidates[0].content && data.candidates[0].content.parts[0].text;
}

/* ── SEARCH ───────────────────────────────────────────────── */
async function handleSearch(body, res) {
  const query = body.query;
  if (!query || typeof query !== 'string') {
    res.status(400).json({ error: 'bad_request', detail: 'query string required' });
    return;
  }

  let found = { text: '', sources: [] };
  if (process.env.TAVILY_KEY) {
    try { found = await withTimeout(tavilyContext(query), 7000); } catch (e) { found = { text: '', sources: [] }; }
  }
  if (!found.text && process.env.GOOGLE_KEY) {
    try { found = await withTimeout(groundedContext(query), 9000); } catch (e) { found = { text: '', sources: [] }; }
  }
  res.status(200).json({ context: found.text, sources: found.sources });
}

async function tavilyContext(query) {
  const r = await fetch('https://api.tavily.com/search', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + process.env.TAVILY_KEY },
    body: JSON.stringify({ api_key: process.env.TAVILY_KEY, query: query, max_results: 5, include_answer: true })
  });
  if (!r.ok) return { text: '', sources: [] };
  const data = await r.json();
  const results = data.results || [];
  if (!results.length && !data.answer) return { text: '', sources: [] };

  const lines = results.slice(0, 5).map(function (x) {
    return '- ' + x.title + ' \u2014 ' + x.url + '\n  ' + (x.content || '').slice(0, 220);
  }).join('\n');

  return {
    text: '\n\nCurrent information from the web (more recent than your training; trust it over your memory when they conflict):\n'
      + (data.answer ? 'Summary: ' + data.answer + '\n' : '') + lines,
    sources: results.slice(0, 5).map(function (x) { return { title: String(x.title || '').slice(0, 80), url: x.url }; }).filter(function (x) { return x.url; })
  };
}

/* Free fallback: Gemini with Google Search grounding (uses GOOGLE_KEY). */
async function groundedContext(query) {
  const base = PROVIDERS.filter(function (p) { return p.id === 'google'; })[0];
  const p = Object.assign({}, base, { cacheKey: 'google:search' });
  const today = new Date().toDateString();

  let srcs = [];
  const out = await tryProviders([p], async function (pp, key, model) {
    const url = 'https://generativelanguage.googleapis.com/v1beta/models/' + model + ':generateContent?key=' + encodeURIComponent(key);
    const r = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        contents: [{ role: 'user', parts: [{ text: 'Today is ' + today + '. Using Google Search, give the most current, specific facts (names, teams, dates, numbers) about the following. Facts only, at most 120 words.\n\n' + query }] }],
        tools: [{ google_search: {} }],
        generationConfig: { temperature: 0.2, maxOutputTokens: 500 }
      })
    });
    if (!r.ok) throw httpError('google_search', r.status);
    const data = await r.json();
    const cand = data.candidates && data.candidates[0];
    const chunks = (cand && cand.groundingMetadata && cand.groundingMetadata.groundingChunks) || [];
    srcs = chunks.map(function (c) { return c.web ? { title: String(c.web.title || '').slice(0, 80), url: c.web.uri } : null; }).filter(function (x) { return x && x.url; }).slice(0, 5);
    const parts = cand && cand.content && cand.content.parts;
    return parts ? parts.map(function (x) { return x.text || ''; }).join('').trim() : '';
  });

  if (!out.reply) return { text: '', sources: [] };
  return { text: '\n\nCurrent information from the web (Google Search; more recent than your training; trust it over your memory when they conflict):\n' + out.reply, sources: srcs };
}
