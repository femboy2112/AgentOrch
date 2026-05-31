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
            this._setNode(state, 'plan', {
                type: 'plan',
                label: `Plan (${total} steps)`,
                status: 'done',
                ended_ts: event.ts,
                meta: { step_total: state.step_total },
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
            this._markActiveStepDone(state, nodeId);
            this._setNode(state, nodeId, {
                type: 'step',
                label: `Step ${idx}/${total}`,
                status: 'active',
                started_ts: event.ts,
                meta: { step_index: idx, step_total: state.step_total },
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
                meta: { selected_branch: o.selected_branch },
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
                meta: { iteration: iter, iteration_total: o.iteration_total },
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
                meta: { reason: o.reason || '', from_worker: o.from_worker, to_worker: o.to_worker },
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
        const textUsage = usageFromText(event.text);
        const inTokens = toNumber(event.data?.in_tokens) || textUsage.in_tokens;
        const outTokens = toNumber(event.data?.out_tokens) || textUsage.out_tokens;
        const cost = toNumber(event.data?.cost_usd ?? event.data?.cost) || toNumber(textUsage.cost_usd);
        state.counters.in_tokens += inTokens;
        state.counters.out_tokens += outTokens;
        state.counters.cost_usd += cost;
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
        };
    }

    subscribe(runId, cb) {
        const state = this._ensure(runId);
        state.listeners.add(cb);
        return () => state.listeners.delete(cb);
    }
}

export const runEventStore = new RunEventStore();
