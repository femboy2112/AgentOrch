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
        return '[ok]';
    }
    if (status === 'failed') {
        return '[x]';
    }
    if (status === 'active') {
        return '[*]';
    }
    return '[ ]';
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
        return `Picked branch ${transition.selected_branch || '?'}${transition.score !== undefined ? ` (score ${transition.score})` : ''}`;
    }
    if (phase === 'adversarial') {
        const idx = transition.step_index || '?';
        const iter = transition.iteration || '?';
        const iterTotal = transition.iteration_total || '?';
        if (action === 'iteration_started') {
            return `Step ${idx} critic round ${iter}/${iterTotal}`;
        }
        if (action === 'iteration_completed') {
            if (transition.outcome) {
                return `Step ${idx} round ${iter}/${iterTotal} ${transition.outcome}`;
            }
            return `Step ${idx} critic round ${iter}/${iterTotal} complete`;
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

function badge(text, cls = '') {
    if (!text) {
        return '';
    }
    return `<span class="plan-badge-item ${cls}">${esc(text)}</span>`;
}

function progressBar(current, total) {
    const p = formatProgress(current, total);
    if (p.label === 'Progress unavailable') {
        return '<div class="plan-progress plan-progress-empty"><span>Progress unavailable</span></div>';
    }
    return `
        <div class="plan-progress" role="img" aria-label="progress ${esc(p.label)}">
            <div class="plan-progress-track">
                <div class="plan-progress-fill" style="width:${p.pct}%"></div>
            </div>
            <span class="plan-progress-label">${esc(p.label)}</span>
        </div>
    `;
}

function cleanModelEffort(meta) {
    const model = String(meta?.model || '').trim();
    const effort = String(meta?.effort || '').trim();
    const modelSafe = model && model !== 'n/a' ? model : '';
    const effortSafe = effort && effort !== 'n/a' ? effort : '';
    if (!modelSafe && !effortSafe) {
        return '';
    }
    return [modelSafe, effortSafe].filter(Boolean).join(' / ');
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
    return parts.join(' / ');
}

function durationChip(node) {
    const started = asNumber(node.meta?.startedTs ?? node.started_ts);
    const ended = asNumber(node.meta?.endedTs ?? node.ended_ts);
    if (node.status === 'active' && started) {
        return `<span class="plan-chip plan-chip-timer plan-live-timer" data-live-elapsed="1" data-start-ts="${started}">elapsed ${esc(formatDuration((Date.now() / 1000) - started))}</span>`;
    }
    if (started && ended && ended >= started) {
        return chip(`duration ${formatDuration(ended - started)}`, 'plan-chip-timer');
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
    const modelEffort = cleanModelEffort(meta);
    const usage = usageChip(meta);
    const duration = durationChip(node);

    if (node.type === 'plan') {
        const roadmap = stepRoadmapHtml(node, context.stepNodes);
        return `
            <div class="plan-node" data-status="${esc(node.status)}" data-type="${esc(node.type)}">
                <span class="plan-node-icon">${iconForStatus(node.status)}</span>
                <div class="plan-node-body">
                    <div class="plan-node-label">${esc(node.label)}</div>
                    ${roadmap}
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
            modelEffort ? chip(modelEffort, 'plan-chip-model') : '',
            usage ? chip(usage, 'plan-chip-usage') : '',
            duration,
        ].filter(Boolean).join('');
        return `
            <div class="plan-node" data-status="${esc(node.status)}" data-type="${esc(node.type)}">
                <span class="plan-node-icon">${iconForStatus(node.status)}</span>
                <div class="plan-node-body">
                    <div class="plan-node-label">${esc(title)}</div>
                    ${progressBar(idx, total)}
                    ${chips ? `<div class="plan-chip-row">${chips}</div>` : ''}
                    ${activity}
                </div>
            </div>
        `;
    }

    if (node.type === 'tot') {
        const selectedBranch = meta.selected_branch ?? '?';
        const branchTotal = meta.branchTotal ?? '?';
        const selector = meta.selector ? ` by ${meta.selector}` : '';
        const score = Number.isFinite(Number(meta.score)) ? `, score ${Number(meta.score).toFixed(3)}` : '';
        return `
            <div class="plan-node" data-status="${esc(node.status)}" data-type="${esc(node.type)}">
                <span class="plan-node-icon">${iconForStatus(node.status)}</span>
                <div class="plan-node-body">
                    <div class="plan-node-label">Picked branch ${esc(selectedBranch)} of ${esc(branchTotal)}${esc(score)}${esc(selector)}</div>
                </div>
            </div>
        `;
    }

    if (node.type === 'iteration') {
        const iter = Number(meta.iteration || 0) || '?';
        const iterTotal = Number(meta.iteration_total || 0) || '?';
        const iterProgress = Number(iter) > 0 && Number(iterTotal) > 0
            ? progressBar(iter, iterTotal)
            : '<div class="plan-progress plan-progress-empty"><span>Critic progress unavailable</span></div>';
        const chips = [
            outcomeBadge(meta),
            modelEffort ? chip(modelEffort, 'plan-chip-model') : '',
        ].filter(Boolean).join('');
        return `
            <div class="plan-node" data-status="${esc(node.status)}" data-type="${esc(node.type)}">
                <span class="plan-node-icon">${iconForStatus(node.status)}</span>
                <div class="plan-node-body">
                    <div class="plan-node-label">Critic round ${esc(iter)}/${esc(iterTotal)}</div>
                    ${iterProgress}
                    ${chips ? `<div class="plan-chip-row">${chips}</div>` : ''}
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
            <div class="plan-node" data-status="${esc(node.status)}" data-type="${esc(node.type)}">
                <span class="plan-node-icon">${iconForStatus(node.status)}</span>
                <div class="plan-node-body">
                    <div class="plan-node-label">${esc(node.label)}</div>
                    ${chips ? `<div class="plan-chip-row">${chips}</div>` : ''}
                    ${reason}
                </div>
            </div>
        `;
    }

    return `
        <div class="plan-node" data-status="${esc(node.status)}" data-type="${esc(node.type)}">
            <span class="plan-node-icon">${iconForStatus(node.status)}</span>
            <div class="plan-node-body">
                <div class="plan-node-label">${esc(node.label)}</div>
            </div>
        </div>
    `;
}

export class FriendlyPlanRenderer {
    constructor(container) {
        this.container = container;
        this.lastSignature = '';
        this._timerInterval = null;
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
            el.textContent = `elapsed ${formatDuration(nowTs - start)}`;
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
            return;
        }
        this.lastSignature = signature;

        const { nodeMap, children, stepNodes } = collectStepGroups(snapshot);
        const planNode = nodeMap.get('plan');
        const planHtml = planNode ? nodeHtml(planNode, { stepNodes, stepTotal: planNode?.meta?.step_total }) : '';
        const badges = snapshot.inferred
            ? '<span class="plan-badge">inferred timeline - limited detail</span>'
            : (snapshot.limited_detail
                ? '<span class="plan-badge">older run - limited detail</span>'
                : '');
        const renderedIds = new Set();

        let stepsHtml = '';
        for (const step of stepNodes) {
            renderedIds.add(step.id);
            const childIds = children.get(step.id) || [];
            const childNodes = childIds.map((id) => nodeMap.get(id)).filter(Boolean);
            for (const child of childNodes) {
                renderedIds.add(child.id);
            }
            const childHtml = childNodes.map((child) => nodeHtml(child, { stepNodes, stepTotal: step.meta?.step_total })).join('');
            stepsHtml += `
                <section class="plan-step-group">
                    ${nodeHtml(step, { stepNodes, stepTotal: step.meta?.step_total })}
                    ${childHtml ? `<div class="plan-step-children">${childHtml}</div>` : ''}
                </section>
            `;
        }

        const rootChildren = (children.get('plan') || [])
            .map((id) => nodeMap.get(id))
            .filter((node) => node && !renderedIds.has(node.id))
            .map((node) => {
                renderedIds.add(node.id);
                return nodeHtml(node, { stepNodes, stepTotal: planNode?.meta?.step_total });
            })
            .join('');

        const dangling = (snapshot.nodes || [])
            .filter((node) => node.id !== 'plan' && !renderedIds.has(node.id))
            .map((node) => {
                renderedIds.add(node.id);
                return nodeHtml(node, { stepNodes, stepTotal: planNode?.meta?.step_total });
            })
            .join('');

        this.container.innerHTML = `
            <div class="friendly-plan">
                <div class="friendly-head">
                    <div class="friendly-status">${esc(statusLine(snapshot))}</div>
                    <div class="friendly-counters">${esc(formatCounters(snapshot.counters || {}))}</div>
                    ${badges}
                </div>
                <div class="friendly-diagram">
                    ${planHtml}
                    ${rootChildren}
                    ${stepsHtml}
                    ${dangling}
                </div>
            </div>
        `;

        this._updateLiveTimers();
        this._syncTimerTicker();
    }
}
