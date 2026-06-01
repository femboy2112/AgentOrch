function esc(value) {
    return String(value ?? '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
}

function asNumber(value) {
    const n = Number(value);
    return Number.isFinite(n) ? n : null;
}

function formatCounters(counters) {
    const inTokens = Number(counters?.in_tokens || 0).toLocaleString();
    const outTokens = Number(counters?.out_tokens || 0).toLocaleString();
    const cost = Number(counters?.cost_usd || 0);
    if (cost > 0) {
        return `in ${inTokens} | out ${outTokens} | $${cost.toFixed(4)}`;
    }
    return `in ${inTokens} | out ${outTokens}`;
}

function formatDuration(secondsRaw) {
    const total = Math.max(0, Math.floor(Number(secondsRaw) || 0));
    const h = Math.floor(total / 3600);
    const m = Math.floor((total % 3600) / 60);
    const s = total % 60;
    if (h > 0) {
        return `${h}h ${m}m ${s}s`;
    }
    if (m > 0) {
        return `${m}m ${s}s`;
    }
    return `${s}s`;
}

function iconForStatus(status) {
    if (status === 'done') {
        return '[ OK ]';
    }
    if (status === 'failed') {
        return '[FAIL]';
    }
    if (status === 'active') {
        return '[ RUN ]';
    }
    return '[ .. ]';
}

function latestTransition(events) {
    for (let i = events.length - 1; i >= 0; i -= 1) {
        const event = events[i];
        if (event.kind !== 'lifecycle') {
            continue;
        }
        if (event.data?.event !== 'orchestration_transition') {
            continue;
        }
        const orchestration = event.data?.orchestration;
        if (orchestration && typeof orchestration === 'object') {
            return orchestration;
        }
    }
    return null;
}

function statusLine(snapshot) {
    const transition = latestTransition(snapshot.events || []);
    if (!transition) {
        if (snapshot.inferred) {
            return 'Running inferred lifecycle timeline...';
        }
        return 'Waiting for orchestration telemetry...';
    }
    const phase = String(transition.phase || '');
    const action = String(transition.action || '');
    if (phase === 'plan' && action === 'started') {
        return 'Planning...';
    }
    if (phase === 'plan' && action === 'completed') {
        return 'Plan completed.';
    }
    if (phase === 'step') {
        const idx = transition.step_index || '?';
        const total = transition.step_total || '?';
        if (action === 'started') {
            return `Step ${idx} of ${total} in progress`;
        }
        if (action === 'completed') {
            return `Step ${idx} of ${total} completed`;
        }
    }
    if (phase === 'tot' && action === 'branch_selected') {
        const branchTotal = Number(transition.branch_total ?? transition.branchTotal);
        const score = Number(transition.score);
        const scoreText = Number.isFinite(score) ? ` (${score.toFixed(1)}/10)` : '';
        if (Number.isFinite(branchTotal) && branchTotal > 1) {
            return `Explored ${branchTotal} approaches${scoreText}`;
        }
        return `Picked the best approach${scoreText}`;
    }
    if (phase === 'adversarial') {
        const idx = transition.step_index || '?';
        const iter = transition.iteration || '?';
        const iterTotal = transition.iteration_total || '?';
        if (action === 'iteration_started') {
            return `Step ${idx} review pass ${iter} of ${iterTotal}`;
        }
        if (action === 'iteration_completed') {
            if (transition.outcome) {
                return `Step ${idx} pass ${iter} of ${iterTotal} ${transition.outcome}`;
            }
            return `Step ${idx} review pass ${iter} of ${iterTotal} complete`;
        }
    }
    if (phase === 'fallback' && action === 'reroute') {
        return `Rerouted ${transition.from_worker || '?'} to ${transition.to_worker || '?'}`;
    }
    return 'Updating run status...';
}

function formatProgress(index, total) {
    const i = Number(index);
    const t = Number(total);
    if (!Number.isFinite(i) || !Number.isFinite(t) || t <= 0) {
        return { label: 'Progress unavailable', pct: 0 };
    }
    const pct = Math.max(0, Math.min(100, Math.round((i / t) * 100)));
    return { label: `${i}/${t}`, pct };
}

function chip(text, cls = '') {
    if (!text) {
        return '';
    }
    return `<span class="plan-chip ${cls}">${esc(text)}</span>`;
}

function kvChip(key, value, cls = '') {
    const keyText = String(key || '').trim();
    const valueText = String(value || '').trim();
    if (!keyText || !valueText) {
        return '';
    }
    return `<span class="plan-chip ${cls}"><span class="plan-chip-key">${esc(keyText)}:</span><span class="plan-chip-value">${esc(valueText)}</span></span>`;
}

function badge(text, cls = '') {
    if (!text) {
        return '';
    }
    return `<span class="plan-badge-item ${cls}">${esc(text)}</span>`;
}

function progressBar(pctRaw, label, options = {}) {
    const pct = Math.max(0, Math.min(100, Number(pctRaw) || 0));
    const cls = String(options.cls || '').trim();
    const fillCls = String(options.fillCls || '').trim();
    const empty = options.empty === true;
    if (empty) {
        return `<div class="plan-progress plan-progress-empty ${esc(cls)}"><span>${esc(label || 'Progress unavailable')}</span></div>`;
    }
    return `
        <div class="plan-progress ${esc(cls)}" role="img" aria-label="progress ${esc(label || '')}">
            <div class="plan-progress-track">
                <div class="plan-progress-fill ${esc(fillCls)}" style="width:${pct}%"></div>
            </div>
            <span class="plan-progress-label">${esc(label || '')}</span>
        </div>
    `;
}

function cleanFieldValue(value) {
    const text = String(value || '').trim();
    if (!text) {
        return '';
    }
    return text.toLowerCase() === 'n/a' ? '' : text;
}

function modelEffortChips(meta) {
    const model = cleanFieldValue(meta?.model);
    const effort = cleanFieldValue(meta?.effort);
    return [
        model ? kvChip('model', model, 'plan-chip-model') : '',
        effort ? kvChip('effort', effort, 'plan-chip-model') : '',
    ].filter(Boolean).join('');
}

function usageChip(meta) {
    const inTokens = Number(meta?.inTokens || 0);
    const outTokens = Number(meta?.outTokens || 0);
    const costUsd = Number(meta?.costUsd || 0);
    if (inTokens <= 0 && outTokens <= 0 && costUsd <= 0) {
        return '';
    }
    const parts = [];
    if (inTokens > 0 || outTokens > 0) {
        parts.push(`in ${inTokens.toLocaleString()} / out ${outTokens.toLocaleString()}`);
    }
    if (costUsd > 0) {
        parts.push(`$${costUsd.toFixed(4)}`);
    }
    return parts.join(' · ');
}

function durationChip(node) {
    const started = asNumber(node.meta?.startedTs ?? node.started_ts);
    const ended = asNumber(node.meta?.endedTs ?? node.ended_ts);
    if (node.status === 'active' && started) {
        return `<span class="plan-chip plan-chip-timer"><span class="plan-chip-key">elapsed:</span><span class="plan-chip-value plan-live-timer" data-live-elapsed="1" data-start-ts="${started}">${esc(formatDuration((Date.now() / 1000) - started))}</span></span>`;
    }
    if (started && ended && ended >= started) {
        return kvChip('duration', formatDuration(ended - started), 'plan-chip-timer');
    }
    return '';
}

function outcomeBadge(meta) {
    const outcome = String(meta?.outcome || '').trim();
    if (outcome === 'verified' || meta?.verified === true) {
        return badge('verified ✓', 'plan-badge-ok');
    }
    if (outcome === 'approved' || meta?.approved === true) {
        return badge('approved ✓', 'plan-badge-good');
    }
    if (outcome === 'stalled') {
        return badge('stalled ⚠', 'plan-badge-warn');
    }
    if (outcome) {
        return badge(outcome, 'plan-badge-neutral');
    }
    return '';
}

function totSummaryLabel(meta) {
    const branchTotal = Number(meta?.branchTotal);
    const score = Number(meta?.score);
    const scoreText = Number.isFinite(score) ? ` (${score.toFixed(1)}/10)` : '';
    if (Number.isFinite(branchTotal) && branchTotal > 1) {
        return `Explored ${branchTotal} approaches · kept the best${scoreText}`;
    }
    return `Picked the best approach${scoreText}`;
}

function summarizeDurationSeconds(snapshot, meta) {
    const candidates = [
        meta?.duration_s,
        meta?.duration,
        meta?.durationSeconds,
    ];
    for (const value of candidates) {
        const n = Number(value);
        if (Number.isFinite(n) && n >= 0) {
            return n;
        }
    }
    let minStart = Number.POSITIVE_INFINITY;
    let maxEnd = 0;
    const now = Date.now() / 1000;
    for (const node of snapshot.nodes || []) {
        const started = asNumber(node.meta?.startedTs ?? node.started_ts);
        if (started && started > 0) {
            minStart = Math.min(minStart, started);
        }
        const ended = asNumber(node.meta?.endedTs ?? node.ended_ts);
        if (ended && ended > 0) {
            maxEnd = Math.max(maxEnd, ended);
        } else if (node.status === 'active' && started) {
            maxEnd = Math.max(maxEnd, now);
        }
    }
    if (Number.isFinite(minStart) && maxEnd >= minStart) {
        return maxEnd - minStart;
    }
    return null;
}

function changedFilesCount(meta) {
    if (Array.isArray(meta?.changed_files)) {
        return meta.changed_files.length;
    }
    const numeric = Number(meta?.changed_files_count ?? meta?.changedFiles);
    if (Number.isFinite(numeric) && numeric >= 0) {
        return Math.floor(numeric);
    }
    return null;
}

function deriveRunSummaryStatus(snapshot, meta) {
    const statuses = (snapshot.nodes || []).map((node) => String(node.status || ''));
    if (statuses.includes('failed') || meta?.success === false) {
        return { icon: '✗', text: 'RUN FAILED' };
    }
    if (statuses.includes('active')) {
        return { icon: '▶', text: 'RUNNING' };
    }
    const planNode = (snapshot.nodes || []).find((node) => node.id === 'plan');
    if (meta?.success === true || planNode?.status === 'done') {
        return { icon: '✅', text: 'RUN COMPLETE' };
    }
    return { icon: '▶', text: 'RUNNING' };
}

function stepProgressText(stepNodes, planNode, snapshot) {
    const totalRaw = Number(
        planNode?.meta?.step_total
        ?? snapshot?.step_total
        ?? stepNodes.length
        ?? 0,
    );
    const total = Number.isFinite(totalRaw) && totalRaw > 0 ? totalRaw : 0;
    if (total <= 0) {
        return '';
    }
    const done = stepNodes.filter((node) => node.status === 'done').length;
    const active = stepNodes.filter((node) => node.status === 'active').length;
    const completed = Math.min(total, done + active);
    return `${completed}/${total} steps`;
}

function stepTotals(stepNodes, planNode, snapshot) {
    const totalRaw = Number(
        planNode?.meta?.step_total
        ?? snapshot?.step_total
        ?? stepNodes.length
        ?? 0,
    );
    const total = Number.isFinite(totalRaw) && totalRaw > 0 ? totalRaw : 0;
    const done = stepNodes.filter((node) => node.status === 'done').length;
    const active = stepNodes.filter((node) => node.status === 'active').length;
    return { total, done, active };
}

function planProgress(stepNodes, planNode, snapshot) {
    const totals = stepTotals(stepNodes, planNode, snapshot);
    if (totals.total <= 0) {
        return progressBar(0, 'Progress unavailable', { empty: true });
    }
    const ratio = (totals.done + (totals.active > 0 ? 0.5 : 0)) / totals.total;
    const pct = Math.round(Math.max(0, Math.min(1, ratio)) * 100);
    const label = `${totals.done}/${totals.total} complete`;
    return progressBar(pct, label, { cls: 'plan-progress-plan' });
}

function stepProgress(node) {
    if (node.status === 'active') {
        return progressBar(100, 'running', {
            cls: 'plan-progress-running',
            fillCls: 'plan-progress-fill-running',
        });
    }
    if (node.status === 'done') {
        return progressBar(100, 'done');
    }
    if (node.status === 'failed') {
        return progressBar(100, 'failed', { cls: 'plan-progress-failed' });
    }
    return progressBar(0, 'pending');
}

function normalizeStreamItems(lines) {
    if (!Array.isArray(lines)) {
        return [];
    }
    const out = [];
    for (const raw of lines) {
        if (!raw || typeof raw !== 'object') {
            continue;
        }
        const worker = String(raw.worker || '').trim();
        const model = String(raw.model || '').trim();
        const text = String(raw.text || '').trim();
        if (!text) {
            continue;
        }
        const prefix = [worker, model].filter(Boolean).join('/');
        out.push(prefix ? `[${prefix}] ${text}` : text);
    }
    return out;
}

function streamBoxHtml(nodeId) {
    return `
        <div class="plan-stream" data-stream-node="${esc(nodeId)}">
            <div class="plan-stream-lines"></div>
        </div>
    `;
}

function objectiveHtml(runMeta) {
    const objective = String(runMeta?.objective || '').trim();
    if (!objective) {
        return '';
    }
    const lineCount = objective.split('\n').length;
    const longText = objective.length > 700 || lineCount > 10;
    return `
        <details class="friendly-objective" ${longText ? '' : 'open'}>
            <summary>Objective</summary>
            <pre>${esc(objective)}</pre>
        </details>
    `;
}

function buildFriendlyHero(snapshot, meta, stepNodes, planNode) {
    const status = deriveRunSummaryStatus(snapshot, meta);
    const parts = [`${status.icon} ${status.text}`];
    const progress = stepProgressText(stepNodes, planNode, snapshot);
    if (progress) {
        parts.push(progress);
    }
    const durationSeconds = summarizeDurationSeconds(snapshot, meta);
    if (durationSeconds !== null) {
        parts.push(formatDuration(durationSeconds));
    }
    const files = changedFilesCount(meta);
    if (files !== null) {
        parts.push(`${files} file${files === 1 ? '' : 's'} changed`);
    }
    return `> ${parts.join(' · ')}`;
}

function fallbackReasonBadge(meta) {
    const category = String(meta?.reasonCategory || '').trim();
    if (!category) {
        return '';
    }
    if (category === 'usage') {
        return badge('usage', 'plan-badge-usage');
    }
    if (category === 'stalled') {
        return badge('stalled', 'plan-badge-warn');
    }
    if (category === 'verbose') {
        return badge('verbose', 'plan-badge-neutral');
    }
    if (category === 'error') {
        return badge('error', 'plan-badge-bad');
    }
    return badge(category, 'plan-badge-neutral');
}

function collectStepGroups(snapshot) {
    const nodeMap = new Map((snapshot.nodes || []).map((node) => [node.id, node]));
    const children = new Map();
    for (const edge of snapshot.edges || []) {
        if (!children.has(edge.from)) {
            children.set(edge.from, []);
        }
        children.get(edge.from).push(edge.to);
    }
    const stepNodes = (snapshot.nodes || [])
        .filter((node) => node.type === 'step' && node.id.startsWith('step:'))
        .sort((a, b) => {
            const ai = Number(a.meta?.step_index ?? a.id.split(':')[1] ?? 0);
            const bi = Number(b.meta?.step_index ?? b.id.split(':')[1] ?? 0);
            return ai - bi;
        });
    return { nodeMap, children, stepNodes };
}

function stepRoadmapHtml(planNode, stepNodes) {
    const titles = Array.isArray(planNode?.meta?.stepTitles) ? planNode.meta.stepTitles : [];
    if (titles.length === 0) {
        return '';
    }
    const statusByIndex = new Map();
    for (const node of stepNodes) {
        const idx = Number(node.meta?.step_index ?? node.id.split(':')[1] ?? 0);
        if (Number.isFinite(idx) && idx > 0) {
            statusByIndex.set(idx, node.status);
        }
    }
    const lines = titles.map((title, i) => {
        const idx = i + 1;
        const status = statusByIndex.get(idx) || 'pending';
        return `<li class="plan-roadmap-item" data-status="${esc(status)}"><span class="plan-roadmap-index">${idx}.</span><span>${esc(title)}</span></li>`;
    }).join('');
    return `<ol class="plan-roadmap">${lines}</ol>`;
}

function nodeHtml(node, context) {
    const meta = node.meta || {};
    const modelEffort = modelEffortChips(meta);
    const usage = usageChip(meta);
    const duration = durationChip(node);
    const streamLines = normalizeStreamItems(meta.stream);
    const showStream = streamLines.length > 0 || node.status === 'active';
    const activeCaret = node.status === 'active'
        ? '<span class="plan-running-caret" aria-hidden="true">▌</span>'
        : '';

    if (node.type === 'plan') {
        const roadmap = stepRoadmapHtml(node, context.stepNodes);
        return `
            <div class="plan-node ${node.status === 'active' ? 'plan-node-running' : ''}" data-status="${esc(node.status)}" data-type="${esc(node.type)}">
                <span class="plan-node-icon">${iconForStatus(node.status)}</span>
                <div class="plan-node-body">
                    <div class="plan-node-label">${esc(node.label)}${activeCaret}</div>
                    ${planProgress(context.stepNodes || [], node, context.snapshot || {})}
                    ${roadmap}
                    ${showStream ? streamBoxHtml(node.id) : ''}
                </div>
            </div>
        `;
    }

    if (node.type === 'step') {
        const idx = Number(meta.step_index || node.id.split(':')[1] || 0) || '?';
        const total = Number(meta.step_total || context.stepTotal || 0) || '?';
        const title = String(meta.title || node.label || `Step ${idx}/${total}`);
        const activity = node.status === 'active' && meta.activity
            ? `<div class="plan-activity">${esc(meta.activity)}</div>`
            : '';
        const chips = [
            modelEffort,
            usage ? kvChip('usage', usage, 'plan-chip-usage') : '',
            duration,
        ].filter(Boolean).join('');
        return `
            <div class="plan-node ${node.status === 'active' ? 'plan-node-running' : ''}" data-status="${esc(node.status)}" data-type="${esc(node.type)}">
                <span class="plan-node-icon">${iconForStatus(node.status)}</span>
                <div class="plan-node-body">
                    <div class="plan-node-label">${esc(title)}${activeCaret}</div>
                    ${stepProgress(node, idx, total)}
                    ${chips ? `<div class="plan-chip-row">${chips}</div>` : ''}
                    ${activity}
                    ${showStream ? streamBoxHtml(node.id) : ''}
                </div>
            </div>
        `;
    }

    if (node.type === 'tot') {
        return `
            <div class="plan-node ${node.status === 'active' ? 'plan-node-running' : ''}" data-status="${esc(node.status)}" data-type="${esc(node.type)}">
                <span class="plan-node-icon">${iconForStatus(node.status)}</span>
                <div class="plan-node-body">
                    <div class="plan-node-label">${esc(totSummaryLabel(meta))}${activeCaret}</div>
                    ${showStream ? streamBoxHtml(node.id) : ''}
                </div>
            </div>
        `;
    }

    if (node.type === 'iteration') {
        const iter = Number(meta.iteration || 0) || '?';
        const iterTotal = Number(meta.iteration_total || 0) || '?';
        const iterStats = formatProgress(iter, iterTotal);
        const iterProgress = iterStats.label === 'Progress unavailable'
            ? progressBar(0, 'Review progress unavailable', { empty: true })
            : progressBar(iterStats.pct, iterStats.label);
        const chips = [
            outcomeBadge(meta),
            modelEffort,
        ].filter(Boolean).join('');
        return `
            <div class="plan-node ${node.status === 'active' ? 'plan-node-running' : ''}" data-status="${esc(node.status)}" data-type="${esc(node.type)}">
                <span class="plan-node-icon">${iconForStatus(node.status)}</span>
                <div class="plan-node-body">
                    <div class="plan-node-label">Review pass ${esc(iter)} of ${esc(iterTotal)}${activeCaret}</div>
                    ${iterProgress}
                    ${chips ? `<div class="plan-chip-row">${chips}</div>` : ''}
                    ${showStream ? streamBoxHtml(node.id) : ''}
                </div>
            </div>
        `;
    }

    if (node.type === 'fallback') {
        const reason = meta.reason ? `<div class="plan-node-note">${esc(meta.reason)}</div>` : '';
        const attempt = Number(meta.attempt || 0);
        const attemptTotal = Number(meta.attemptTotal || 0);
        const attemptText = (attempt > 0 && attemptTotal > 0) ? `attempt ${attempt}/${attemptTotal}` : '';
        const chips = [
            fallbackReasonBadge(meta),
            attemptText ? chip(attemptText, 'plan-chip-usage') : '',
        ].filter(Boolean).join('');
        return `
            <div class="plan-node ${node.status === 'active' ? 'plan-node-running' : ''}" data-status="${esc(node.status)}" data-type="${esc(node.type)}">
                <span class="plan-node-icon">${iconForStatus(node.status)}</span>
                <div class="plan-node-body">
                    <div class="plan-node-label">${esc(node.label)}${activeCaret}</div>
                    ${chips ? `<div class="plan-chip-row">${chips}</div>` : ''}
                    ${reason}
                    ${showStream ? streamBoxHtml(node.id) : ''}
                </div>
            </div>
        `;
    }

    return `
        <div class="plan-node ${node.status === 'active' ? 'plan-node-running' : ''}" data-status="${esc(node.status)}" data-type="${esc(node.type)}">
            <span class="plan-node-icon">${iconForStatus(node.status)}</span>
            <div class="plan-node-body">
                <div class="plan-node-label">${esc(node.label)}${activeCaret}</div>
                ${showStream ? streamBoxHtml(node.id) : ''}
            </div>
        </div>
    `;
}

export class FriendlyPlanRenderer {
    constructor(container, options = {}) {
        this.container = container;
        this.runMeta = options.runMeta && typeof options.runMeta === 'object' ? options.runMeta : {};
        this.lastSignature = '';
        this._timerInterval = null;
        this._streamState = new Map();
        this._streamTickHandle = null;
        this._typingQueue = [];
    }

    setRunMeta(meta) {
        this.runMeta = meta && typeof meta === 'object' ? meta : {};
        this.lastSignature = '';
    }

    _updateLiveTimers() {
        if (!this.container || this.container.hidden) {
            return;
        }
        const timerNodes = this.container.querySelectorAll('[data-live-elapsed="1"][data-start-ts]');
        const nowTs = Date.now() / 1000;
        for (const el of timerNodes) {
            const start = Number(el.dataset.startTs || 0);
            if (!Number.isFinite(start) || start <= 0) {
                continue;
            }
            el.textContent = formatDuration(nowTs - start);
        }
    }

    _syncTimerTicker() {
        const hasLiveTimers = this.container.querySelector('[data-live-elapsed="1"]') !== null;
        if (hasLiveTimers && !this._timerInterval) {
            this._timerInterval = setInterval(() => this._updateLiveTimers(), 1000);
            return;
        }
        if (!hasLiveTimers && this._timerInterval) {
            clearInterval(this._timerInterval);
            this._timerInterval = null;
        }
    }

    _ensureStreamTicker() {
        if (this._streamTickHandle) {
            return;
        }
        this._streamTickHandle = setInterval(() => {
            if (this._typingQueue.length === 0) {
                this._stopStreamTicker();
                return;
            }
            const queue = this._typingQueue.slice();
            this._typingQueue = [];
            for (const job of queue) {
                if (!job || !job.el || !this.container.contains(job.el)) {
                    continue;
                }
                const remaining = job.text.length - job.pos;
                if (remaining <= 0) {
                    continue;
                }
                const burst = Math.min(6, Math.max(1, Math.ceil(remaining / 8)));
                const nextPos = Math.min(job.text.length, job.pos + burst);
                job.pos = nextPos;
                job.el.textContent = job.text.slice(0, job.pos);
                const scroller = job.scroller;
                if (scroller) {
                    scroller.scrollTop = scroller.scrollHeight;
                }
                if (job.pos < job.text.length) {
                    this._typingQueue.push(job);
                }
            }
            if (this._typingQueue.length === 0) {
                this._stopStreamTicker();
            }
        }, 40);
    }

    _stopStreamTicker() {
        if (!this._streamTickHandle) {
            return;
        }
        clearInterval(this._streamTickHandle);
        this._streamTickHandle = null;
    }

    _syncStreams(snapshot) {
        if (!this.container || this.container.hidden) {
            return;
        }
        const activeNodes = (snapshot?.nodes || []).filter((node) => node?.status === 'active');
        const hasActiveStep = activeNodes.some((node) => node?.type === 'step');
        const nextStreams = new Map();
        for (const node of snapshot?.nodes || []) {
            if (!node || !node.id) {
                continue;
            }
            const lines = normalizeStreamItems(node.meta?.stream);
            if (lines.length > 0 || node.status === 'active') {
                nextStreams.set(String(node.id), lines);
            }
        }
        const liveTail = normalizeStreamItems(snapshot?.liveTail);
        if (!hasActiveStep && liveTail.length > 0) {
            nextStreams.set('__live_tail__', liveTail);
        }

        for (const nodeId of Array.from(this._streamState.keys())) {
            if (!nextStreams.has(nodeId)) {
                this._streamState.delete(nodeId);
            }
        }

        const boxes = this.container.querySelectorAll('[data-stream-node]');
        for (const box of boxes) {
            const nodeId = String(box.dataset.streamNode || '');
            if (!nodeId) {
                continue;
            }
            const linesWrap = box.querySelector('.plan-stream-lines');
            if (!linesWrap) {
                continue;
            }
            const targetLines = nextStreams.get(nodeId) || [];
            let state = this._streamState.get(nodeId);
            if (!state) {
                state = { lines: [] };
                this._streamState.set(nodeId, state);
            }

            let overlap = Math.min(state.lines.length, targetLines.length);
            while (overlap > 0) {
                const left = state.lines.slice(state.lines.length - overlap).join('\n');
                const right = targetLines.slice(0, overlap).join('\n');
                if (left === right) {
                    break;
                }
                overlap -= 1;
            }

            if (overlap === 0 && state.lines.length > 0 && targetLines.length > 0) {
                linesWrap.textContent = '';
                state.lines = [];
            } else if (overlap < state.lines.length) {
                const keep = state.lines.length - overlap;
                for (let i = 0; i < keep; i += 1) {
                    if (linesWrap.firstChild) {
                        linesWrap.removeChild(linesWrap.firstChild);
                    }
                }
                state.lines = state.lines.slice(state.lines.length - overlap);
            }

            const delta = targetLines.slice(overlap);
            if (delta.length === 0) {
                continue;
            }

            for (let i = 0; i < delta.length; i += 1) {
                const text = delta[i];
                const lineEl = document.createElement('div');
                lineEl.className = 'plan-stream-line';
                const newestIncoming = i === delta.length - 1;
                if (newestIncoming) {
                    lineEl.textContent = '';
                    linesWrap.appendChild(lineEl);
                    this._typingQueue.push({
                        el: lineEl,
                        text,
                        pos: 0,
                        scroller: box,
                    });
                } else {
                    lineEl.textContent = text;
                    linesWrap.appendChild(lineEl);
                }
            }

            const maxLines = 40;
            while (linesWrap.children.length > maxLines) {
                linesWrap.removeChild(linesWrap.firstChild);
            }
            state.lines = targetLines.slice(-maxLines);
            box.scrollTop = box.scrollHeight;
        }

        if (this._typingQueue.length > 0) {
            this._ensureStreamTicker();
        } else {
            this._stopStreamTicker();
        }
    }

    render(snapshot) {
        if (!snapshot || typeof snapshot !== 'object') {
            this.container.innerHTML = '<div class="plan-empty">No run data yet.</div>';
            return;
        }
        const signature = JSON.stringify({
            nodes: snapshot.nodes || [],
            edges: snapshot.edges || [],
            counters: snapshot.counters || {},
            inferred: snapshot.inferred,
            limited_detail: snapshot.limited_detail,
            events: (snapshot.events || []).length,
        });
        if (signature === this.lastSignature) {
            this._updateLiveTimers();
            this._syncTimerTicker();
            this._syncStreams(snapshot);
            return;
        }
        this.lastSignature = signature;

        const { nodeMap, children, stepNodes } = collectStepGroups(snapshot);
        const planNode = nodeMap.get('plan');
        const planHtml = planNode ? nodeHtml(planNode, { stepNodes, stepTotal: planNode?.meta?.step_total, snapshot }) : '';
        const heroLine = buildFriendlyHero(snapshot, this.runMeta, stepNodes, planNode);
        const objectiveBox = objectiveHtml(this.runMeta);
        const badges = snapshot.inferred
            ? '<span class="plan-badge">inferred timeline - limited detail</span>'
            : (snapshot.limited_detail
                ? '<span class="plan-badge">older run - limited detail</span>'
                : '');
        const renderedIds = new Set();
        const activeStep = stepNodes.find((node) => node.status === 'active');

        let stepsHtml = '';
        for (const step of stepNodes) {
            renderedIds.add(step.id);
            const childIds = children.get(step.id) || [];
            const childNodes = childIds.map((id) => nodeMap.get(id)).filter(Boolean);
            for (const child of childNodes) {
                renderedIds.add(child.id);
            }
            const childHtml = childNodes.map((child) => nodeHtml(child, { stepNodes, stepTotal: step.meta?.step_total, snapshot })).join('');
            stepsHtml += `
                <section class="plan-step-group">
                    <div class="plan-step-main">
                        ${nodeHtml(step, { stepNodes, stepTotal: step.meta?.step_total, snapshot })}
                    </div>
                    ${childHtml ? `<div class="plan-step-children">${childHtml}</div>` : ''}
                </section>
            `;
        }

        const rootChildren = (children.get('plan') || [])
            .map((id) => nodeMap.get(id))
            .filter((node) => node && !renderedIds.has(node.id))
            .map((node) => {
                renderedIds.add(node.id);
                return nodeHtml(node, { stepNodes, stepTotal: planNode?.meta?.step_total, snapshot });
            })
            .join('');

        const dangling = (snapshot.nodes || [])
            .filter((node) => node.id !== 'plan' && !renderedIds.has(node.id))
            .map((node) => {
                renderedIds.add(node.id);
                return nodeHtml(node, { stepNodes, stepTotal: planNode?.meta?.step_total, snapshot });
            })
            .join('');

        const showLiveTail = !activeStep && Array.isArray(snapshot.liveTail) && snapshot.liveTail.length > 0;
        const preservedStreams = new Map();
        for (const el of this.container.querySelectorAll('[data-stream-node]')) {
            const id = String(el.dataset.streamNode || '');
            if (id) {
                preservedStreams.set(id, el);
            }
        }

        this.container.innerHTML = `
            <div class="friendly-plan">
                <div class="friendly-head">
                    <div class="friendly-hero">${esc(heroLine)}</div>
                    <div class="friendly-status">${esc(statusLine(snapshot))}</div>
                    <div class="friendly-counters">${esc(formatCounters(snapshot.counters || {}))}</div>
                    ${badges ? `<div class="friendly-badges">${badges}</div>` : ''}
                    ${objectiveBox}
                    ${showLiveTail ? `<div class="friendly-live-tail">${streamBoxHtml('__live_tail__')}</div>` : ''}
                </div>
                <div class="friendly-diagram ${stepNodes.length > 0 ? 'friendly-diagram-has-steps' : ''}">
                    ${planHtml ? `<div class="plan-flow-head">${planHtml}</div>` : ''}
                    <div class="plan-flow-body">
                        ${rootChildren}
                        ${stepsHtml}
                        ${dangling}
                    </div>
                </div>
            </div>
        `;

        for (const placeholder of this.container.querySelectorAll('[data-stream-node]')) {
            const id = String(placeholder.dataset.streamNode || '');
            const reused = preservedStreams.get(id);
            if (reused && reused !== placeholder) {
                placeholder.replaceWith(reused);
            }
        }

        this._updateLiveTimers();
        this._syncTimerTicker();
        this._syncStreams(snapshot);
    }
}
