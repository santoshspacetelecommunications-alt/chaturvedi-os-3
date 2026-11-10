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
     GOOGLE_KEY     (free)
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
          const reply = await withTimeout(caller(p, key, model), 20000);
          if (reply && String(reply).trim()) {
            rememberWorking(p.cacheKey || p.id, model);
            return { reply: reply };
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
  const out = await tryProviders(PROVIDERS, function (p, key, model) {
    return callProvider(p, key, model, messages);
  });
  if (out.reply) { res.status(200).json({ reply: out.reply }); return; }
  res.status(502).json(out);
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
  if (p.kind === 'gemini') return callGemini(messages, key, model);
  if (p.kind === 'anthropic') return callAnthropic(p, messages, key, model);
  return callOpenAICompatible(p, key, model, messages);
}

async function callOpenAICompatible(p, key, model, messages) {
  const reqBody = { model: model, messages: messages, max_tokens: 1500 };
  if (model.indexOf(':free') === -1) reqBody.temperature = 0.7;

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
    body: JSON.stringify({ model: model, max_tokens: 1500, system: system || undefined, messages: turns })
  });
  if (!r.ok) throw httpError(p.id, r.status);
  const data = await r.json();
  return data.content && data.content[0] && data.content[0].text;
}

async function callGemini(messages, key, model) {
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
    body: JSON.stringify({ contents: merged, generationConfig: { temperature: 0.7, maxOutputTokens: 1500 } })
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

  if (!process.env.TAVILY_KEY) {
    res.status(200).json({ context: '' });
    return;
  }

  try {
    const r = await fetch('https://api.tavily.com/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ api_key: process.env.TAVILY_KEY, query: query, max_results: 4, include_answer: true })
    });

    if (!r.ok) { res.status(200).json({ context: '' }); return; }

    const data = await r.json();
    if (!data.results || !data.results.length) { res.status(200).json({ context: '' }); return; }

    const lines = data.results.slice(0, 4).map(function (x) {
      return '- ' + x.title + ' \u2014 ' + x.url + '\n  ' + (x.content || '').slice(0, 180);
    }).join('\n');

    res.status(200).json({ context: '\n\nCurrent information from the web (more recent than your training, use it):\n' + lines });
  } catch (e) {
    res.status(200).json({ context: '' });
  }
}
