function toNumber(value) {
    const n = Number(value);
    return Number.isFinite(n) ? n : 0;
}

function usageFromText(text) {
    const raw = String(text || '');
    const inMatch = raw.match(/\bin(?:_tokens)?\s*[=:]\s*(\d+)/i);
    const outMatch = raw.match(/\bout(?:_tokens)?\s*[=:]\s*(\d+)/i);
    const costMatch = raw.match(/\bcost(?:_usd)?\s*[=:]\s*([0-9]*\.?[0-9]+)/i);
    return {
        in_tokens: inMatch ? Number(inMatch[1]) : 0,
        out_tokens: outMatch ? Number(outMatch[1]) : 0,
        cost_usd: costMatch ? Number(costMatch[1]) : undefined,
    };
}

function truncateText(value, maxLen = 160) {
    const raw = String(value || '').trim();
    if (raw.length <= maxLen) {
        return raw;
    }
    return `${raw.slice(0, Math.max(0, maxLen - 3)).trimEnd()}...`;
}

function usageFromEvent(event) {
    const textUsage = usageFromText(event.text);
    const inTokens = toNumber(event.data?.in_tokens) || textUsage.in_tokens;
    const outTokens = toNumber(event.data?.out_tokens) || textUsage.out_tokens;
    const eventCostRaw = event.data?.cost_usd;
    const eventCostAlt = event.data?.cost;
    const eventCost = eventCostRaw !== undefined && eventCostRaw !== null ? eventCostRaw : eventCostAlt;
    const cost = toNumber(eventCost) || toNumber(textUsage.cost_usd);
    return { inTokens, outTokens, cost };
}

function normalizeEvent(workerEvent, runId) {
    const evt = workerEvent && typeof workerEvent === 'object' ? workerEvent : {};
    const data = evt.data && typeof evt.data === 'object' ? evt.data : {};
    return {
        ts: Number(evt.ts || 0),
        run_id: String(evt.run_id || runId || ''),
        worker: String(evt.worker || ''),
        model: String(evt.model || ''),
        effort: String(evt.effort || ''),
        branch: evt.branch ?? null,
        kind: String(evt.kind || 'message'),
        text: String(evt.text || ''),
        data,
    };
}

function createState(runId) {
    return {
        run_id: runId,
        events: [],
        nodes: new Map(),
        nodeOrder: [],
        edges: [],
        edgeSet: new Set(),
        counters: { in_tokens: 0, out_tokens: 0, cost_usd: 0 },
        inferred: false,
        has_orchestration: false,
        current_step: null,
        step_total: null,
        fallback_count: 0,
        inferred_count: 0,
        steps_seen: 0,
        steps_with_title: 0,
        listeners: new Set(),
        seen_ids: new Set(),
    };
}

export class RunEventStore {
    constructor() {
        this._runs = new Map();
    }

    _ensure(runId) {
        if (!this._runs.has(runId)) {
            this._runs.set(runId, createState(runId));
        }
        return this._runs.get(runId);
    }

    _notify(state) {
        for (const cb of Array.from(state.listeners)) {
            try {
                cb();
            } catch (_err) {
                // Listener failures should not break ingestion.
            }
        }
    }

    _setNode(state, nodeId, patch) {
        const existing = state.nodes.get(nodeId);
        if (!existing) {
            state.nodeOrder.push(nodeId);
        }
        state.nodes.set(nodeId, {
            id: nodeId,
            type: patch.type || existing?.type || 'step',
            label: patch.label || existing?.label || nodeId,
            status: patch.status || existing?.status || 'pending',
            started_ts: patch.started_ts ?? existing?.started_ts,
            ended_ts: patch.ended_ts ?? existing?.ended_ts,
            meta: { ...(existing?.meta || {}), ...(patch.meta || {}) },
        });
    }

    _ensureEdge(state, from, to) {
        if (!from || !to) {
            return;
        }
        const key = `${from}->${to}`;
        if (state.edgeSet.has(key)) {
            return;
        }
        state.edgeSet.add(key);
        state.edges.push({ from, to });
    }

    _markActiveStepDone(state, skipNodeId) {
        for (const nodeId of state.nodeOrder) {
            if (nodeId === skipNodeId) {
                continue;
            }
            const node = state.nodes.get(nodeId);
            if (node?.type === 'step' && node.status === 'active') {
                node.status = 'done';
                node.ended_ts = node.ended_ts || Date.now() / 1000;
            }
        }
    }

    _foldOrchestration(state, event) {
        const data = event.data || {};
        if (data.event !== 'orchestration_transition') {
            return false;
        }
        const o = data.orchestration && typeof data.orchestration === 'object' ? data.orchestration : {};
        const phase = String(o.phase || '');
        const action = String(o.action || '');
        const stepIndex = Number.isFinite(Number(o.step_index)) ? Number(o.step_index) : state.current_step;
        const stepTotal = Number.isFinite(Number(o.step_total)) ? Number(o.step_total) : state.step_total;
        if (Number.isFinite(stepTotal)) {
            state.step_total = stepTotal;
        }

        if (phase === 'plan' && action === 'completed') {
            const total = Number.isFinite(state.step_total) ? state.step_total : '?';
            const stepTitles = Array.isArray(o.step_titles)
                ? o.step_titles.map((item) => truncateText(item, 120)).filter(Boolean)
                : undefined;
            this._setNode(state, 'plan', {
                type: 'plan',
                label: `Plan (${total} steps)`,
                status: 'done',
                ended_ts: event.ts,
                meta: { step_total: state.step_total, stepTitles },
            });
            return true;
        }
        if (phase === 'plan' && action === 'started') {
            this._setNode(state, 'plan', {
                type: 'plan',
                label: 'Plan',
                status: 'active',
                started_ts: event.ts,
            });
            return true;
        }
        if (phase === 'step' && action === 'started') {
            const idx = Number.isFinite(stepIndex) ? stepIndex : (state.current_step || 1);
            const total = Number.isFinite(state.step_total) ? state.step_total : '?';
            const nodeId = `step:${idx}`;
            state.current_step = idx;
            state.steps_seen += 1;
            if (o.step_title) {
                state.steps_with_title += 1;
            }
            this._markActiveStepDone(state, nodeId);
            this._setNode(state, nodeId, {
                type: 'step',
                label: `Step ${idx}/${total}`,
                status: 'active',
                started_ts: event.ts,
                meta: {
                    step_index: idx,
                    step_total: state.step_total,
                    title: truncateText(o.step_title, 120),
                    model: o.model ? String(o.model) : undefined,
                    effort: o.effort ? String(o.effort) : undefined,
                    startedTs: event.ts,
                },
            });
            this._ensureEdge(state, 'plan', nodeId);
            return true;
        }
        if (phase === 'step' && action === 'completed') {
            const idx = Number.isFinite(stepIndex) ? stepIndex : state.current_step;
            if (idx !== null && idx !== undefined) {
                this._setNode(state, `step:${idx}`, {
                    type: 'step',
                    status: 'done',
                    ended_ts: event.ts,
                    meta: {
                        title: truncateText(o.step_title, 120),
                        endedTs: event.ts,
                    },
                });
            }
            return true;
        }
        if (phase === 'tot' && action === 'branch_selected') {
            const idx = Number.isFinite(stepIndex) ? stepIndex : state.current_step;
            const branch = Number.isFinite(Number(o.selected_branch)) ? Number(o.selected_branch) : '?';
            const nodeId = `tot:${idx ?? 'root'}`;
            this._setNode(state, nodeId, {
                type: 'tot',
                label: `Picked branch ${branch}`,
                status: 'done',
                ended_ts: event.ts,
                meta: {
                    selected_branch: o.selected_branch,
                    branchTotal: Number.isFinite(Number(o.branch_total)) ? Number(o.branch_total) : undefined,
                    selector: o.selector ? String(o.selector) : undefined,
                    score: Number.isFinite(Number(o.score)) ? Number(o.score) : undefined,
                    scores: Array.isArray(o.scores) ? o.scores : undefined,
                },
            });
            this._ensureEdge(state, idx ? `step:${idx}` : 'plan', nodeId);
            return true;
        }
        if (phase === 'adversarial' && action === 'iteration_started') {
            const idx = Number.isFinite(stepIndex) ? stepIndex : state.current_step;
            const iter = Number.isFinite(Number(o.iteration)) ? Number(o.iteration) : 1;
            const iterTotal = Number.isFinite(Number(o.iteration_total)) ? Number(o.iteration_total) : '?';
            const nodeId = `iter:${idx ?? 'root'}:${iter}`;
            this._setNode(state, nodeId, {
                type: 'iteration',
                label: `Critic iter ${iter}/${iterTotal}`,
                status: 'active',
                started_ts: event.ts,
                meta: {
                    iteration: iter,
                    iteration_total: o.iteration_total,
                    model: o.model ? String(o.model) : undefined,
                    effort: o.effort ? String(o.effort) : undefined,
                },
            });
            this._ensureEdge(state, idx ? `step:${idx}` : 'plan', nodeId);
            return true;
        }
        if (phase === 'adversarial' && action === 'iteration_completed') {
            const idx = Number.isFinite(stepIndex) ? stepIndex : state.current_step;
            const iter = Number.isFinite(Number(o.iteration)) ? Number(o.iteration) : 1;
            this._setNode(state, `iter:${idx ?? 'root'}:${iter}`, {
                type: 'iteration',
                status: 'done',
                ended_ts: event.ts,
                meta: {
                    outcome: o.outcome ? String(o.outcome) : undefined,
                    verified: o.verified === true ? true : (o.verified === false ? false : undefined),
                    approved: o.approved === true ? true : (o.approved === false ? false : undefined),
                },
            });
            return true;
        }
        if (phase === 'fallback' && action === 'reroute') {
            state.fallback_count += 1;
            const nodeId = `fallback:${state.fallback_count}`;
            const fromWorker = String(o.from_worker || '?');
            const toWorker = String(o.to_worker || '?');
            this._setNode(state, nodeId, {
                type: 'fallback',
                label: `Reroute ${fromWorker} -> ${toWorker}`,
                status: 'failed',
                started_ts: event.ts,
                ended_ts: event.ts,
                meta: {
                    reason: o.reason || '',
                    from_worker: o.from_worker,
                    to_worker: o.to_worker,
                    reasonCategory: o.reason_category ? String(o.reason_category) : undefined,
                    attempt: Number.isFinite(Number(o.attempt)) ? Number(o.attempt) : undefined,
                    attemptTotal: Number.isFinite(Number(o.attempt_total)) ? Number(o.attempt_total) : undefined,
                },
            });
            const parent = state.current_step ? `step:${state.current_step}` : 'plan';
            this._ensureEdge(state, parent, nodeId);
            return true;
        }

        return false;
    }

    _foldInferredLifecycle(state, event) {
        if (event.kind !== 'lifecycle' || state.has_orchestration) {
            return;
        }
        const data = event.data && typeof event.data === 'object' ? event.data : {};
        const lifecycleEvent = String(data.event || '');
        if (lifecycleEvent !== 'agent_started' && lifecycleEvent !== 'agent_finished') {
            return;
        }

        state.inferred = true;
        this._setNode(state, 'plan', {
            type: 'plan',
            label: 'Inferred Timeline',
            status: 'active',
        });

        if (lifecycleEvent === 'agent_started') {
            state.inferred_count += 1;
            const nodeId = `infer:${state.inferred_count}`;
            this._setNode(state, nodeId, {
                type: 'step',
                label: `${event.worker || 'worker'} started`,
                status: 'active',
                started_ts: event.ts,
                meta: { worker: event.worker || '' },
            });
            this._ensureEdge(state, 'plan', nodeId);
            return;
        }

        for (let i = state.nodeOrder.length - 1; i >= 0; i -= 1) {
            const node = state.nodes.get(state.nodeOrder[i]);
            if (node?.id?.startsWith('infer:') && node.status === 'active') {
                node.status = 'done';
                node.ended_ts = event.ts;
                break;
            }
        }
        let hasActive = false;
        for (const nodeId of state.nodeOrder) {
            const node = state.nodes.get(nodeId);
            if (node?.id?.startsWith('infer:') && node.status === 'active') {
                hasActive = true;
                break;
            }
        }
        if (!hasActive) {
            this._setNode(state, 'plan', { type: 'plan', status: 'done', ended_ts: event.ts });
        }
    }

    _foldCounters(state, event) {
        if (event.kind !== 'usage') {
            return;
        }
        const { inTokens, outTokens, cost } = usageFromEvent(event);
        state.counters.in_tokens += inTokens;
        state.counters.out_tokens += outTokens;
        state.counters.cost_usd += cost;
    }

    _applyFrontendMeta(state, nodes) {
        const stepNodes = nodes
            .filter((node) => node.type === 'step' && node.id.startsWith('step:'))
            .sort((a, b) => {
                const ai = Number(a.meta?.step_index ?? a.id.split(':')[1] ?? 0);
                const bi = Number(b.meta?.step_index ?? b.id.split(':')[1] ?? 0);
                return ai - bi;
            });
        const byStepId = new Map(stepNodes.map((node) => [node.id, node]));
        const activeStep = stepNodes
            .filter((node) => node.status === 'active')
            .sort((a, b) => Number(b.started_ts || 0) - Number(a.started_ts || 0))[0] || null;

        for (let i = 0; i < stepNodes.length; i += 1) {
            const node = stepNodes[i];
            const next = stepNodes[i + 1];
            const startedTs = Number(node.started_ts || node.meta?.startedTs || 0) || undefined;
            const endedTs = Number(node.ended_ts || node.meta?.endedTs || 0) || undefined;
            const nextStartedTs = Number(next?.started_ts || next?.meta?.startedTs || 0) || undefined;
            node.meta.startedTs = startedTs;
            node.meta.endedTs = endedTs;
            node.meta.inTokens = 0;
            node.meta.outTokens = 0;
            node.meta.costUsd = 0;
            node._usageWindowEndTs = endedTs || nextStartedTs || (node.status === 'active' ? Number.POSITIVE_INFINITY : undefined);
        }

        for (const event of state.events) {
            if (event.kind !== 'usage') {
                continue;
            }
            for (const node of stepNodes) {
                const startedTs = Number(node.meta.startedTs || 0);
                const windowEnd = Number(node._usageWindowEndTs);
                const hasWindowEnd = node._usageWindowEndTs !== undefined;
                if (!Number.isFinite(startedTs) || !hasWindowEnd) {
                    continue;
                }
                if (event.ts < startedTs) {
                    continue;
                }
                if (event.ts > windowEnd) {
                    continue;
                }
                const usage = usageFromEvent(event);
                node.meta.inTokens += usage.inTokens;
                node.meta.outTokens += usage.outTokens;
                node.meta.costUsd += usage.cost;
                break;
            }
        }

        if (activeStep) {
            const startedTs = Number(activeStep.meta?.startedTs || 0);
            let latestActivity = '';
            for (let i = state.events.length - 1; i >= 0; i -= 1) {
                const event = state.events[i];
                if (event.kind !== 'message') {
                    continue;
                }
                if (event.ts < startedTs) {
                    break;
                }
                if (!event.text || event.worker === 'orchestrator') {
                    continue;
                }
                latestActivity = truncateText(event.text, 160);
                break;
            }
            if (latestActivity) {
                activeStep.meta.activity = latestActivity;
            } else {
                delete activeStep.meta.activity;
            }
        }

        for (const node of stepNodes) {
            delete node._usageWindowEndTs;
            if (node.meta.inTokens <= 0) {
                delete node.meta.inTokens;
            }
            if (node.meta.outTokens <= 0) {
                delete node.meta.outTokens;
            }
            if (node.meta.costUsd <= 0) {
                delete node.meta.costUsd;
            }
            if (!node.meta.title) {
                const idx = node.meta?.step_index ?? node.id.split(':')[1] ?? '?';
                const total = node.meta?.step_total ?? state.step_total ?? '?';
                node.meta.title = `Step ${idx}/${total}`;
            }
        }

        const planNode = nodes.find((node) => node.id === 'plan');
        if (planNode && !Array.isArray(planNode.meta?.stepTitles)) {
            const titles = stepNodes
                .map((node) => node.meta?.title)
                .filter(Boolean);
            if (titles.length > 0) {
                planNode.meta.stepTitles = titles;
            }
        }

        for (const node of nodes) {
            if (node.type === 'tot') {
                if (node.meta?.branchTotal !== undefined) {
                    node.meta.branchTotal = Number(node.meta.branchTotal) || undefined;
                }
                if (node.meta?.score !== undefined) {
                    node.meta.score = Number(node.meta.score);
                    if (!Number.isFinite(node.meta.score)) {
                        delete node.meta.score;
                    }
                }
            }
            if (node.type === 'iteration') {
                if (!node.meta?.outcome) {
                    if (node.meta?.verified === true) {
                        node.meta.outcome = 'verified';
                    } else if (node.meta?.approved === true) {
                        node.meta.outcome = 'approved';
                    }
                }
            }
            if (node.type === 'fallback') {
                if (!node.meta?.reasonCategory && node.meta?.reason) {
                    if (String(node.meta.reason).toLowerCase().includes('usage')) {
                        node.meta.reasonCategory = 'usage';
                    }
                }
            }
            if (byStepId.has(node.id) && node.meta?.model === 'n/a') {
                delete node.meta.model;
            }
            if (byStepId.has(node.id) && node.meta?.effort === 'n/a') {
                delete node.meta.effort;
            }
        }
    }

    _rebuildGraph(state) {
        state.nodes.clear();
        state.nodeOrder = [];
        state.edges = [];
        state.edgeSet.clear();
        state.inferred = false;
        state.current_step = null;
        state.step_total = null;
        state.fallback_count = 0;
        state.inferred_count = 0;
        state.steps_seen = 0;
        state.steps_with_title = 0;
        for (const event of state.events) {
            const handled = this._foldOrchestration(state, event);
            if (!handled) {
                this._foldInferredLifecycle(state, event);
            }
        }
    }

    append(runId, workerEvent, eventId) {
        const state = this._ensure(runId);
        if (eventId !== null && eventId !== undefined) {
            const key = String(eventId);
            if (state.seen_ids.has(key)) {
                return;
            }
            state.seen_ids.add(key);
        }

        const event = normalizeEvent(workerEvent, runId);
        state.events.push(event);
        this._foldCounters(state, event);

        const isOrchestration = this._foldOrchestration(state, event);
        if (isOrchestration) {
            if (!state.has_orchestration) {
                state.has_orchestration = true;
                this._rebuildGraph(state);
            } else {
                state.inferred = false;
            }
        } else {
            this._foldInferredLifecycle(state, event);
        }

        this._notify(state);
    }

    snapshot(runId) {
        const state = this._ensure(runId);
        const nodes = state.nodeOrder
            .map((id) => state.nodes.get(id))
            .filter(Boolean)
            .map((node) => ({ ...node, meta: { ...(node.meta || {}) } }));
        this._applyFrontendMeta(state, nodes);
        const counters = {
            in_tokens: state.counters.in_tokens,
            out_tokens: state.counters.out_tokens,
        };
        if (state.counters.cost_usd > 0) {
            counters.cost_usd = state.counters.cost_usd;
        }
        return {
            run_id: state.run_id,
            events: state.events.slice(),
            nodes,
            edges: state.edges.slice(),
            counters,
            inferred: Boolean(state.inferred),
            limited_detail: Boolean(
                state.has_orchestration
                && state.steps_seen > 0
                && state.steps_with_title === 0,
            ),
        };
    }

    subscribe(runId, cb) {
        const state = this._ensure(runId);
        state.listeners.add(cb);
        return () => state.listeners.delete(cb);
    }
}

export const runEventStore = new RunEventStore();
