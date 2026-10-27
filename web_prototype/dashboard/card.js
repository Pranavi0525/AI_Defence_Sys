/* web_prototype/dashboard/card.js -- Credit Card Fraud Detection module + top-level module switch.
 *
 * Self-contained: renders into #cardModule, talks only to /api/card/*, and never calls any payment endpoint. It shares nothing with
 * the payment code in index.html except the API-base text box (#apiBase). Every number shown comes from the API response.
 * The model behind it is trained on SYNTHETIC simulator data; the UI says so wherever a score is shown.
 */
(function () {
  'use strict';

  var MODULE_KEY = 'ai_defense_module';
  var COLS = ['card', 'ts', 'amount', 'mcc', 'channel', 'lat', 'lon', 'merchant', 'declined'];
  var REQUIRED = COLS.slice(0, 8);
  var SHOW_ROWS = 200;

  // ---------------------------------------------------------------- helpers
  function $(s, r) { return (r || document).querySelector(s); }
  function esc(v) { return String(v == null ? '' : v).replace(/[&<>"']/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]; }); }
  function pct(x, d) { return x == null ? '—' : (x * 100).toFixed(d == null ? 1 : d) + '%'; }
  function fx(x, d) { return x == null ? '—' : Number(x).toFixed(d); }
  function sci(x) { return x == null ? '—' : (x !== 0 && Math.abs(x) < 0.001 ? Number(x).toExponential(2) : Number(x).toFixed(4)); }
  function apiBase() { var el = document.getElementById('apiBase'); return ((el && el.value) || '').trim().replace(/\/+$/, ''); }

  function ccFetch(path, opts) {
    var base = apiBase();
    if (!base) return Promise.reject({ kind: 'no_base' });
    var ctl = new AbortController(); var timer = setTimeout(function () { ctl.abort(); }, 60000);
    return fetch(base + path, Object.assign({}, opts || {}, { signal: ctl.signal })).then(function (res) {
      clearTimeout(timer);
      return res.json().catch(function () { return null; }).then(function (body) {
        if (!res.ok) throw { kind: 'http', status: res.status, body: body };
        return body;
      });
    }, function (e) { clearTimeout(timer); throw { kind: e && e.name === 'AbortError' ? 'timeout' : 'network' }; });
  }

  function errorHtml(e) {
    if (!e || e.kind === 'network') return 'Could not reach the backend. Check the API URL at the top of the page.';
    if (e.kind === 'no_base') return 'Set the backend API URL in the box at the top of the page.';
    if (e.kind === 'timeout') return 'The backend took too long to respond (it may be waking up). Try again.';
    var b = e.body || {};
    if (e.status === 503) return 'The credit-card model is not available: ' + esc(b.detail || 'artifacts not loaded') + ' The digital-payment module is unaffected.';
    if (e.status === 404) return 'This backend does not have the credit-card API (/api/card). Update or redeploy the backend.';
    if (e.status === 422 && b.error === 'invalid_card_input') {
      var items = (b.detail || []).map(function (d) {
        var where = d.row == null ? 'Input' : 'Data row ' + (d.row + 1);
        return '<li>' + where + (d.field ? ' · <b>' + esc(d.field) + '</b>' : '') + ': ' + esc(d.problem) + '</li>';
      }).join('');
      var more = b.total_problems > (b.detail || []).length ? '<li>… and ' + (b.total_problems - b.detail.length) + ' more</li>' : '';
      return 'The input was rejected (nothing was scored, nothing was guessed):<ul>' + items + more + '</ul>';
    }
    if (e.status === 422 && b.error === 'validation_error') {
      return 'The request was rejected:<ul>' + (b.detail || []).map(function (d) { return '<li>' + esc((d.loc || []).slice(1).join('.')) + ': ' + esc(d.msg) + '</li>'; }).join('') + '</ul>';
    }
    return 'Unexpected error (HTTP ' + esc(e.status) + ').';
  }

  // CSV (RFC-4180-ish) -> array of row arrays
  function parseCsvRows(text) {
    var rows = [], row = [], cell = '', q = false, i = 0, c;
    text = String(text || '').replace(/^﻿/, '');
    for (; i < text.length; i++) {
      c = text[i];
      if (q) {
        if (c === '"') { if (text[i + 1] === '"') { cell += '"'; i++; } else q = false; } else cell += c;
      } else if (c === '"') q = true;
      else if (c === ',') { row.push(cell); cell = ''; }
      else if (c === '\n' || c === '\r') { if (c === '\r' && text[i + 1] === '\n') i++; row.push(cell); rows.push(row); row = []; cell = ''; }
      else cell += c;
    }
    if (cell !== '' || row.length) { row.push(cell); rows.push(row); }
    return rows.filter(function (r) { return !(r.length === 1 && r[0].trim() === ''); });
  }
  // -> array of objects, converted the same way the server converts CSV cells
  function csvToObjects(text) {
    var rows = parseCsvRows(text);
    if (!rows.length) return [];
    var head = rows[0].map(function (h) { return h.trim().toLowerCase(); });
    return rows.slice(1).map(function (r) {
      var o = {};
      head.forEach(function (h, i) {
        var v = (r[i] == null ? '' : r[i]).trim();
        if (v === '') o[h] = null;
        else if (['ts', 'amount', 'lat', 'lon', 'declined'].indexOf(h) >= 0 && !isNaN(Number(v))) o[h] = Number(v);
        else if ((h === 'card' || h === 'mcc') && /^-?\d+$/.test(v)) o[h] = parseInt(v, 10);
        else o[h] = v;
      });
      return o;
    });
  }
  function csvEscape(v) { v = v == null ? '' : String(v); return /[",\n\r]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v; }
  function toCsv(header, rows) { return [header.join(',')].concat(rows.map(function (r) { return header.map(function (h) { return csvEscape(r[h]); }).join(','); })).join('\r\n') + '\r\n'; }
  function download(name, text) {
    var a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([text], { type: 'text/csv;charset=utf-8' })); a.download = name;
    document.body.appendChild(a); a.click(); setTimeout(function () { URL.revokeObjectURL(a.href); a.remove(); }, 500);
  }

  // ---------------------------------------------------------------- example INPUTS (hand-written; results come from the model)
  var DAY = 86400, T0 = 20 * DAY + 10 * 3600;
  function exLegitHistory(card) {
    var out = [], m = ['M5411_3', 'M5812_2', 'M5411_3', 'M5411_1', 'M5812_2', 'M5411_3', 'M5541_4'], mcc = [5411, 5812, 5411, 5411, 5812, 5411, 5541], amt = [640, 380, 910, 455, 520, 700, 1500];
    for (var i = 0; i < 7; i++) out.push({ card: card, ts: T0 + i * 86400 + i * 1237, amount: amt[i], mcc: mcc[i], channel: 'POS', lat: 13.0827, lon: 80.2707, merchant: m[i], declined: 0 });
    return out;
  }
  function exLegit() { var h = exLegitHistory('C-1001'); return { history: h, tx: { card: 'C-1001', ts: T0 + 7 * 86400 + 900, amount: 560, mcc: 5411, channel: 'POS', lat: 13.0827, lon: 80.2707, merchant: 'M5411_3', declined: 0 } }; }
  function exBurstTail(card, t) {
    var out = [], decl = [1, 1, 0, 1, 0, 1], i;
    for (i = 0; i < 6; i++) { t += 40; out.push({ card: card, ts: t, amount: [2.5, 4, 9.99, 3, 14, 6][i], mcc: [5411, 5812, 5311, 5651, 4111, 5814][i], channel: 'ECOM', lat: 19.076, lon: 72.8777, merchant: 'M0_t' + (i % 3), declined: decl[i] }); }
    return { rows: out, t: t };
  }
  function exBurst() {
    var h = exLegitHistory('C-2002'), t = T0 + 7 * DAY, b = exBurstTail('C-2002', t);
    return { history: h.concat(b.rows), tx: { card: 'C-2002', ts: b.t + 95, amount: 6400, mcc: 5947, channel: 'ECOM', lat: 19.076, lon: 72.8777, merchant: 'M0_t1', declined: 0 } };
  }
  function exBatch() {
    var a = exLegit(), b = exBurst();
    return a.history.concat([a.tx], b.history, [b.tx]);
  }

  // ---------------------------------------------------------------- module markup
  var HTML = '' +
    '<div class="cc-hero">' +
    '  <h1>Credit Card Fraud Detection</h1>' +
    '  <p>Scores card authorisations with a separate model, feature set and threshold from the digital-payment module. Velocity and novelty features look at the card\'s <b>earlier</b> transactions, so supply a card\'s history for a meaningful result.</p>' +
    '  <div class="banner-msg warn"><div><b>Synthetic prototype.</b> The model was trained and evaluated on simulator data only. <b>model score is a ranking score, not a real-world fraud probability</b>, and this tool must not be used for real authorisation decisions.</div></div>' +
    '  <div class="cc-status" id="ccStatus" role="status" aria-live="polite"><span class="dot"></span><span>checking card model…</span></div>' +
    '</div>' +
    '<div class="ws-tabs" role="tablist" aria-label="Credit card views">' +
    '  <button class="ws-tab" id="ccTabSingle" role="tab" aria-selected="true" aria-controls="ccPanelSingle">Single transaction</button>' +
    '  <button class="ws-tab" id="ccTabBatch" role="tab" aria-selected="false" aria-controls="ccPanelBatch">Batch (CSV)</button>' +
    '  <button class="ws-tab" id="ccTabModel" role="tab" aria-selected="false" aria-controls="ccPanelModel">Model &amp; evaluation</button>' +
    '</div>' +

    '<div id="ccPanelSingle" class="ws-panel" role="tabpanel" aria-labelledby="ccTabSingle">' +
    ' <form id="ccForm" novalidate>' +
    '  <div class="cc-panel"><h3>Transaction to score</h3><p class="cc-sub">Fields are the ones the model\'s feature pipeline reads. <span class="mono">ts</span> is event time in seconds (hour-of-day uses ts mod 86400, UTC).</p>' +
    '   <div class="form-grid">' +
    '    <div class="field"><label for="cc_card">card</label><input type="text" id="cc_card" placeholder="C-1001"></div>' +
    '    <div class="field"><label for="cc_ts">ts (seconds)</label><input type="number" id="cc_ts" step="any" min="0"></div>' +
    '    <div class="field"><label for="cc_amount">amount</label><input type="number" id="cc_amount" step="any" min="0"></div>' +
    '    <div class="field"><label for="cc_mcc">mcc</label><input type="text" id="cc_mcc" placeholder="5411"></div>' +
    '    <div class="field"><label for="cc_channel">channel</label><select id="cc_channel"><option>POS</option><option>ECOM</option><option>ATM</option></select></div>' +
    '    <div class="field"><label for="cc_lat">lat</label><input type="number" id="cc_lat" step="any" min="-90" max="90"></div>' +
    '    <div class="field"><label for="cc_lon">lon</label><input type="number" id="cc_lon" step="any" min="-180" max="180"></div>' +
    '    <div class="field"><label for="cc_merchant">merchant</label><input type="text" id="cc_merchant" placeholder="M5411_3"></div>' +
    '    <div class="field"><label>&nbsp;</label><label class="checkbox-row"><input type="checkbox" id="cc_declined"> declined (this authorisation)</label><span class="hint">Only PRIOR declines are used as features.</span></div>' +
    '   </div>' +
    '  </div>' +
    '  <div class="cc-panel"><h3>Card history (optional, recommended)</h3><p class="cc-sub">Earlier transactions of the same card, as CSV with header <span class="mono">' + COLS.join(',') + '</span> (<span class="mono">declined</span> optional). With no history the transaction is scored as the card\'s first.</p>' +
    '   <textarea class="cc-area" id="cc_history" spellcheck="false" aria-label="Card history CSV"></textarea>' +
    '   <div class="cc-row"><button type="submit" class="run-btn" id="ccScoreBtn">Score transaction</button>' +
    '    <button type="button" class="cc-chip" id="ccExLegit">Load example: ordinary card</button>' +
    '    <button type="button" class="cc-chip" id="ccExBurst">Load example: card-testing burst</button>' +
    '    <button type="button" class="cc-chip" id="ccClear">Clear</button></div>' +
    '  </div>' +
    ' </form>' +
    ' <div id="ccSingleOut" aria-live="polite"></div>' +
    '</div>' +

    '<div id="ccPanelBatch" class="ws-panel" role="tabpanel" aria-labelledby="ccTabBatch" hidden>' +
    ' <div class="cc-panel"><h3>Batch prediction</h3><p class="cc-sub">Upload or paste a CSV with header <span class="mono">' + COLS.join(',') + '</span> (<span class="mono">declined</span> optional). Every card\'s rows are scored in time order, each using only that card\'s earlier rows. The whole file is rejected, with row numbers, if any row is invalid — nothing is dropped or guessed.</p>' +
    '  <input type="file" id="ccFile" class="cc-file" accept=".csv,text/csv" aria-label="CSV file">' +
    '  <div style="margin-top:12px"><textarea class="cc-area" id="ccBatchText" spellcheck="false" aria-label="Batch CSV" style="min-height:160px"></textarea></div>' +
    '  <div class="cc-row"><button type="button" class="run-btn" id="ccBatchBtn">Score batch</button>' +
    '   <button type="button" class="cc-chip" id="ccBatchSample">Load sample batch</button>' +
    '   <button type="button" class="cc-chip" id="ccBatchTemplate">Download CSV template</button></div>' +
    ' </div>' +
    ' <div id="ccBatchOut" aria-live="polite"></div>' +
    '</div>' +

    '<div id="ccPanelModel" class="ws-panel" role="tabpanel" aria-labelledby="ccTabModel" hidden><div id="ccModelOut"><div class="empty-note">Loading model card…</div></div></div>';

  // ---------------------------------------------------------------- status + model card
  var modelCard = null;
  function refreshStatus() {
    var el = $('#ccStatus'); if (!el) return Promise.resolve();
    return ccFetch('/api/card/status').then(function (s) {
      if (s.ready) el.innerHTML = '<span class="dot ok"></span><span>card model ready · ' + esc(s.model_version) + ' · threshold ' + esc(sci(s.threshold)) + ' · synthetic-trained</span>';
      else el.innerHTML = '<span class="dot bad"></span><span>card model unavailable: ' + esc(s.error || 'not loaded') + '</span>';
    }, function (e) {
      el.innerHTML = '<span class="dot bad"></span><span>' + (e && e.status === 404 ? 'backend has no /api/card routes' : e && e.kind === 'no_base' ? 'set the API URL above' : 'backend unreachable') + '</span>';
    });
  }

  function renderModelCard(c) {
    var a = c.evaluation.at_threshold, cm = function (v, cls) { return '<div class="' + cls + '">' + esc(v) + '</div>'; };
    var sweep = Object.keys(c.evaluation.threshold_sweep_by_validation_fpr).sort(function (x, y) { return Number(y) - Number(x); }).map(function (k) {
      var s = c.evaluation.threshold_sweep_by_validation_fpr[k];
      return '<tr><td class="mono">' + esc(pct(Number(k), 1)) + '</td><td class="mono">' + esc(sci(s.threshold)) + '</td><td class="mono">' + esc(pct(s.recall)) + '</td><td class="mono">' + esc(pct(s.precision)) + '</td><td class="mono">' + esc(fx(s.f1, 3)) + '</td><td class="mono">' + esc(s.fp) + '</td><td class="mono">' + esc(s.fn) + '</td></tr>';
    }).join('');
    $('#ccModelOut').innerHTML = '' +
      '<div class="banner-msg warn"><div><b>' + esc(c.evaluation.label) + '.</b> ' + esc(c.score_meaning) + '</div></div>' +
      '<div class="cc-panel"><h3>Operating point</h3><p class="cc-sub">Model ' + esc(c.model_version) + ' · seed ' + esc(c.seed) + ' · trained ' + esc(c.trained_utc) + ' · data: ' + esc(c.data_source) + '</p>' +
      '<div class="cc-kv"><div><div class="k">Threshold (model score)</div><div class="v">' + esc(sci(c.threshold)) + '</div></div>' +
      '<div><div class="k">Target FPR on validation legit</div><div class="v">' + esc(pct(c.operating_point.target_false_positive_rate_on_validation_legit)) + '</div></div>' +
      '<div><div class="k">Features</div><div class="v">' + esc(c.features.length) + '</div></div></div></div>' +
      '<div class="cc-panel"><h3>Held-out synthetic test block at this threshold</h3><p class="cc-sub">' + esc(c.evaluation.test_rows) + ' rows, ' + esc(c.evaluation.test_fraud_rows) + ' fraud rows in ' + esc(c.evaluation.test_fraud_events) + ' simulated fraud events (event recall ' + esc(pct(c.evaluation.event_recall)) + ').</p>' +
      '<div class="cc-kv" style="margin:0 0 14px"><div><div class="k">Precision</div><div class="v">' + esc(pct(a.precision)) + '</div></div><div><div class="k">Recall</div><div class="v">' + esc(pct(a.recall)) + '</div></div>' +
      '<div><div class="k">F1</div><div class="v">' + esc(fx(a.f1, 3)) + '</div></div><div><div class="k">False-positive rate</div><div class="v">' + esc(pct(a.false_positive_rate, 2)) + '</div></div>' +
      '<div><div class="k">Precision at ASSUMED 0.6% prevalence</div><div class="v">' + esc(pct(a.precision_at_assumed_0p6pct)) + '</div></div></div>' +
      '<div class="cc-matrix"><div class="h"></div><div class="h">predicted fraud</div><div class="h">predicted legit</div>' +
      '<div class="h">actual fraud</div>' + cm('TP ' + a.tp, 'good') + cm('FN ' + a.fn, 'bad') + '<div class="h">actual legit</div>' + cm('FP ' + a.fp, 'bad') + cm('TN ' + a.tn, 'good') + '</div></div>' +
      '<div class="cc-panel"><h3>Threshold-dependent performance</h3><p class="cc-sub">Same held-out synthetic block; threshold set at each false-positive rate on validation legitimate traffic.</p>' +
      '<div class="cc-table-wrap"><table class="data"><thead><tr><th>Target FPR</th><th>Threshold</th><th>Recall</th><th>Precision</th><th>F1</th><th>False positives</th><th>False negatives</th></tr></thead><tbody>' + sweep + '</tbody></table></div></div>' +
      '<div class="cc-panel"><h3>Limitations</h3><ul class="cc-limits">' + c.limitations.map(function (l) { return '<li>' + esc(l) + '</li>'; }).join('') + '</ul></div>' +
      '<details class="tech-details"><summary>Input schema &amp; the ' + esc(c.features.length) + ' model features</summary><pre>' + esc(JSON.stringify({ input_schema: c.input_schema, features: c.features }, null, 2)) + '</pre></details>';
  }
  function loadModelCard() {
    return ccFetch('/api/card/model-card').then(function (c) { modelCard = c; renderModelCard(c); }, function (e) {
      $('#ccModelOut').innerHTML = '<div class="banner-msg error"><div>' + errorHtml(e) + '</div></div>';
    });
  }

  // ---------------------------------------------------------------- single prediction
  function fillForm(t) {
    $('#cc_card').value = t.card; $('#cc_ts').value = t.ts; $('#cc_amount').value = t.amount; $('#cc_mcc').value = t.mcc; $('#cc_channel').value = t.channel;
    $('#cc_lat').value = t.lat; $('#cc_lon').value = t.lon; $('#cc_merchant').value = t.merchant; $('#cc_declined').checked = !!t.declined;
  }
  function readForm() {
    var num = function (id) { var v = $(id).value.trim(); return v === '' ? null : Number(v); };
    var idv = function (id) { var v = $(id).value.trim(); return v === '' ? null : (/^-?\d+$/.test(v) ? parseInt(v, 10) : v); };
    return { card: idv('#cc_card'), ts: num('#cc_ts'), amount: num('#cc_amount'), mcc: idv('#cc_mcc'), channel: $('#cc_channel').value, lat: num('#cc_lat'), lon: num('#cc_lon'),
             merchant: $('#cc_merchant').value.trim(), declined: $('#cc_declined').checked ? 1 : 0 };
  }

  function renderSingle(r) {
    var fraud = r.classification === 'FRAUD';
    var why = fraud
      ? 'model score ' + sci(r.model_score) + ' is at or above the threshold ' + sci(r.threshold) + '; it ranks above ' + pct(r.legit_percentile, 2) + ' of legitimate validation transactions.'
      : 'model score ' + sci(r.model_score) + ' is below the threshold ' + sci(r.threshold) + '; it ranks above ' + pct(r.legit_percentile, 2) + ' of legitimate validation transactions.';
    var warns = (r.warnings || []).length ? '<ul class="cc-warn-list">' + r.warnings.map(function (w) { return '<li>' + esc(w) + '</li>'; }).join('') + '</ul>' : '';
    $('#ccSingleOut').innerHTML = '<div class="cc-panel"><div class="cc-verdict ' + (fraud ? 'fraud' : 'legit') + '"><div class="cc-label">' + esc(r.classification) + '</div><div class="cc-why">' + esc(why) + '</div></div>' +
      '<div class="cc-kv"><div><div class="k">Model score</div><div class="v">' + esc(sci(r.model_score)) + '</div></div>' +
      '<div><div class="k">Percentile vs legitimate (validation)</div><div class="v">' + esc(pct(r.legit_percentile, 2)) + '</div></div>' +
      '<div><div class="k">Threshold</div><div class="v">' + esc(sci(r.threshold)) + '</div></div>' +
      '<div><div class="k">Prior transactions used</div><div class="v">' + esc(r.prior_transactions_on_card) + '</div></div>' +
      '<div><div class="k">Latency</div><div class="v">' + esc(fx(r.inference_latency_ms, 1)) + ' ms</div></div></div>' + warns +
      '<div class="cc-note">' + esc(r.disclaimer) + ' · ' + esc(r.model_version) + '</div>' +
      '<details class="tech-details" style="margin-top:12px"><summary>Raw API response</summary><pre>' + esc(JSON.stringify(r, null, 2)) + '</pre></details></div>';
  }

  function onScore(ev) {
    ev.preventDefault();
    var out = $('#ccSingleOut'), btn = $('#ccScoreBtn');
    var hist = $('#cc_history').value.trim() ? csvToObjects($('#cc_history').value) : [];
    btn.disabled = true; out.innerHTML = '<div class="empty-note"><span class="cc-spinner"></span>Scoring…</div>';
    ccFetch('/api/card/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ transaction: readForm(), history: hist }) })
      .then(renderSingle, function (e) { out.innerHTML = '<div class="banner-msg error"><div>' + errorHtml(e) + '</div></div>'; })
      .then(function () { btn.disabled = false; });
  }

  // ---------------------------------------------------------------- batch prediction
  var lastBatch = null;

  function renderBatch(resp, inputs, showAll, onlyFlagged) {
    var s = resp.summary, rows = resp.results.map(function (r, i) { return { r: r, x: inputs[i] || {} }; });
    if (onlyFlagged) rows = rows.filter(function (o) { return o.r.is_fraud; });
    var shown = showAll ? rows : rows.slice(0, SHOW_ROWS);
    var body = shown.map(function (o) {
      return '<tr class="' + (o.r.is_fraud ? 'cc-fraud-row' : '') + '"><td class="mono">' + (o.r.index + 1) + '</td><td class="mono">' + esc(o.x.card) + '</td><td class="mono">' + esc(o.x.ts) + '</td><td class="mono">' + esc(o.x.amount) + '</td><td class="mono">' + esc(o.x.channel) + '</td><td class="mono">' + esc(o.x.merchant) + '</td>' +
        '<td><span class="cc-tag ' + (o.r.is_fraud ? 'fraud' : 'legit') + '">' + esc(o.r.classification) + '</span></td><td class="mono">' + esc(sci(o.r.model_score)) + '</td><td class="mono">' + esc(pct(o.r.legit_percentile, 2)) + '</td><td class="mono">' + esc(o.r.prior_transactions_on_card) + '</td></tr>';
    }).join('');
    $('#ccBatchOut').innerHTML = '<div class="cc-panel"><h3>Batch summary</h3>' +
      '<div class="cc-kv"><div><div class="k">Transactions</div><div class="v">' + esc(s.transactions) + '</div></div><div><div class="k">Flagged FRAUD</div><div class="v">' + esc(s.flagged_fraud) + '</div></div>' +
      '<div><div class="k">LEGITIMATE</div><div class="v">' + esc(s.legitimate) + '</div></div><div><div class="k">Flagged rate</div><div class="v">' + esc(pct(s.flagged_rate, 2)) + '</div></div>' +
      '<div><div class="k">Model score min / mean / max</div><div class="v" style="font-size:13px">' + esc(sci(s.model_score.min)) + ' / ' + esc(sci(s.model_score.mean)) + ' / ' + esc(sci(s.model_score.max)) + '</div></div>' +
      '<div><div class="k">Threshold</div><div class="v">' + esc(sci(resp.threshold)) + '</div></div></div>' +
      '<div class="cc-note">The flagged rate is the share of THIS file above the threshold; it is not a measured fraud rate or precision. ' + esc(resp.disclaimer) + '</div>' +
      '<div class="cc-row"><button type="button" class="run-btn" id="ccDownload">Download results (CSV)</button>' +
      '<label class="checkbox-row"><input type="checkbox" id="ccOnlyFlagged"' + (onlyFlagged ? ' checked' : '') + '> show flagged only</label></div></div>' +
      '<div class="cc-panel"><h3>Transaction-level results</h3><p class="cc-sub">' + esc(rows.length) + ' row(s)' + (!showAll && rows.length > SHOW_ROWS ? ' — showing the first ' + SHOW_ROWS : '') + '.</p>' +
      '<div class="cc-table-wrap"><table class="data"><thead><tr><th>#</th><th>card</th><th>ts</th><th>amount</th><th>channel</th><th>merchant</th><th>classification</th><th>model score</th><th>percentile vs legit</th><th>prior txns</th></tr></thead><tbody>' + body + '</tbody></table></div>' +
      (!showAll && rows.length > SHOW_ROWS ? '<div class="cc-row"><button type="button" class="cc-chip" id="ccShowAll">Show all ' + esc(rows.length) + ' rows</button></div>' : '') + '</div>';
    $('#ccDownload').onclick = function () {
      var extra = ['classification', 'is_fraud', 'model_score', 'legit_percentile', 'prior_transactions_on_card'];
      var data = resp.results.map(function (r, i) { var o = Object.assign({}, inputs[i]); extra.forEach(function (k) { o[k] = r[k]; }); return o; });
      download('card_fraud_predictions.csv', toCsv(COLS.concat(extra), data));
    };
    $('#ccOnlyFlagged').onchange = function () { renderBatch(resp, inputs, showAll, this.checked); };
    var sa = $('#ccShowAll'); if (sa) sa.onclick = function () { renderBatch(resp, inputs, true, onlyFlagged); };
  }

  function onBatch() {
    var text = $('#ccBatchText').value, out = $('#ccBatchOut'), btn = $('#ccBatchBtn');
    if (!text.trim()) { out.innerHTML = '<div class="banner-msg error"><div>Paste a CSV or choose a file first.</div></div>'; return; }
    var inputs = csvToObjects(text);
    btn.disabled = true; out.innerHTML = '<div class="empty-note"><span class="cc-spinner"></span>Scoring ' + inputs.length + ' transactions…</div>';
    ccFetch('/api/card/predict-batch', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ csv_text: text }) })
      .then(function (resp) { lastBatch = { resp: resp, inputs: inputs }; renderBatch(resp, inputs, false, false); },
            function (e) { out.innerHTML = '<div class="banner-msg error"><div>' + errorHtml(e) + '</div></div>'; })
      .then(function () { btn.disabled = false; });
  }

  // ---------------------------------------------------------------- tabs + module switch
  function selectTab(name) {
    [['Single', 'ccTabSingle', 'ccPanelSingle'], ['Batch', 'ccTabBatch', 'ccPanelBatch'], ['Model', 'ccTabModel', 'ccPanelModel']].forEach(function (t) {
      var on = t[0] === name; $('#' + t[1]).setAttribute('aria-selected', on ? 'true' : 'false'); $('#' + t[2]).hidden = !on;
    });
    if (name === 'Model' && !modelCard) loadModelCard();
  }

  var built = false;
  function build() {
    if (built) return; built = true;
    $('#cardModule').innerHTML = HTML;
    $('#ccTabSingle').onclick = function () { selectTab('Single'); };
    $('#ccTabBatch').onclick = function () { selectTab('Batch'); };
    $('#ccTabModel').onclick = function () { selectTab('Model'); };
    $('#ccForm').addEventListener('submit', onScore);
    $('#ccExLegit').onclick = function () { var e = exLegit(); fillForm(e.tx); $('#cc_history').value = toCsv(COLS, e.history); };
    $('#ccExBurst').onclick = function () { var e = exBurst(); fillForm(e.tx); $('#cc_history').value = toCsv(COLS, e.history); };
    $('#ccClear').onclick = function () { $('#ccForm').reset(); $('#cc_history').value = ''; $('#ccSingleOut').innerHTML = ''; };
    $('#ccBatchSample').onclick = function () { $('#ccBatchText').value = toCsv(COLS, exBatch()); };
    $('#ccBatchTemplate').onclick = function () { download('card_transactions_template.csv', toCsv(COLS, [exLegit().tx])); };
    $('#ccBatchBtn').onclick = onBatch;
    $('#ccFile').onchange = function () {
      var f = this.files && this.files[0]; if (!f) return;
      var rd = new FileReader(); rd.onload = function () { $('#ccBatchText').value = String(rd.result); }; rd.readAsText(f);
    };
    $('#ccExLegit').click();
  }

  function setModule(name, persist) {
    var card = name === 'card';
    document.body.setAttribute('data-module', card ? 'card' : 'payment');
    $('#paymentModule').hidden = card; $('#cardModule').hidden = !card;
    $('#modTabPayment').setAttribute('aria-selected', card ? 'false' : 'true'); $('#modTabCard').setAttribute('aria-selected', card ? 'true' : 'false');
    document.title = card ? 'AI Defense Lab — Credit Card Fraud Console' : 'AI Defense Lab — Payment Fraud Console';
    if (persist) { try { localStorage.setItem(MODULE_KEY, card ? 'card' : 'payment'); } catch (e) { /* storage may be unavailable */ } }
    if (card) { build(); refreshStatus(); }
  }

  function init() {
    if (!$('#cardModule') || !$('#paymentModule')) return;
    $('#modTabPayment').onclick = function () { setModule('payment', true); };
    $('#modTabCard').onclick = function () { setModule('card', true); };
    var apiInput = document.getElementById('apiBase');
    if (apiInput) apiInput.addEventListener('change', function () { if (!$('#cardModule').hidden) refreshStatus(); });
    var want = 'payment';
    try { var q = new URLSearchParams(location.search).get('module'); want = q || localStorage.getItem(MODULE_KEY) || 'payment'; } catch (e) { /* ignore */ }
    setModule(want === 'card' ? 'card' : 'payment', false);
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();

  // exposed for tests only (pure helpers, no side effects)
  window.__cardUi = { csvToObjects: csvToObjects, toCsv: toCsv, parseCsvRows: parseCsvRows, exBurst: exBurst, exLegit: exLegit, exBatch: exBatch };
})();
