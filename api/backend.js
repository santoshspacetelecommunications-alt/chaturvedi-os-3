/* ══════════════════════════════════════════════════════════
   /api/backend — one Vercel serverless function, two jobs

   The frontend sends { type: 'chat', messages: [...] } or
   { type: 'search', query: '...' } to this same endpoint.
   This is the only file that touches your real API keys.
   They come from environment variables set in Vercel, never
   from this file's source code.

   Required environment variables (Vercel → Settings →
   Environment Variables):
     GROQ_KEY
     OPENROUTER_KEY
     NVIDIA_KEY
     GOOGLE_KEY     (optional)
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
  } else {
    await handleChat(body, res);
  }
};

/* ── CHAT ─────────────────────────────────────────────────── */
async function handleChat(body, res) {
  const messages = body.messages;
  if (!Array.isArray(messages) || messages.length === 0) {
    res.status(400).json({ error: 'bad_request', detail: 'messages array required' });
    return;
  }

  const providers = [
    { id: 'groq', key: process.env.GROQ_KEY, model: 'llama-3.3-70b-versatile', url: 'https://api.groq.com/openai/v1/chat/completions' },
    { id: 'openrouter', key: process.env.OPENROUTER_KEY, model: 'meta-llama/llama-3.3-70b-instruct:free', url: 'https://openrouter.ai/api/v1/chat/completions' },
    { id: 'nvidia', key: process.env.NVIDIA_KEY, model: 'meta/llama-3.3-70b-instruct', url: 'https://integrate.api.nvidia.com/v1/chat/completions' }
  ];

  let lastError = null;

  for (let i = 0; i < providers.length; i++) {
    const p = providers[i];
    if (!p.key) continue;
    try {
      const reply = await withTimeout(callOpenAICompatible(p, messages), 20000);
      if (reply && reply.trim()) {
        res.status(200).json({ reply: reply });
        return;
      }
    } catch (e) {
      lastError = e;
    }
  }

  if (process.env.GOOGLE_KEY) {
    try {
      const reply = await withTimeout(callGemini(messages, process.env.GOOGLE_KEY), 20000);
      if (reply && reply.trim()) {
        res.status(200).json({ reply: reply });
        return;
      }
    } catch (e) {
      lastError = e;
    }
  }

  res.status(502).json({
    error: 'all_providers_failed',
    detail: String((lastError && lastError.message) || lastError || 'unknown')
  });
}

function withTimeout(promise, ms) {
  return Promise.race([
    promise,
    new Promise(function (_, reject) {
      setTimeout(function () { reject(new Error('timeout')); }, ms);
    })
  ]);
}

async function callOpenAICompatible(p, messages) {
  const reqBody = { model: p.model, messages: messages, max_tokens: 1500 };
  if (p.model.indexOf(':free') === -1) reqBody.temperature = 0.7;

  const headers = { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + p.key };
  if (p.id === 'openrouter') {
    headers['HTTP-Referer'] = 'https://chaturvedi.app';
    headers['X-Title'] = 'Chaturvedi';
  }

  const r = await fetch(p.url, { method: 'POST', headers: headers, body: JSON.stringify(reqBody) });
  if (!r.ok) throw new Error(p.id + '_http_' + r.status);
  const data = await r.json();
  return data.choices && data.choices[0] && data.choices[0].message && data.choices[0].message.content;
}

async function callGemini(messages, key) {
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

  const url = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=' + key;
  const r = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ contents: merged, generationConfig: { temperature: 0.7, maxOutputTokens: 1500 } })
  });
  if (!r.ok) throw new Error('google_http_' + r.status);
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
