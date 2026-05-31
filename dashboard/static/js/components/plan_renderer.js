function esc(value) {
    return String(value ?? '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
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
            return `Step ${idx} of ${total} - in progress`;
        }
        if (action === 'completed') {
            return `Step ${idx} of ${total} completed`;
        }
    }
    if (phase === 'tot' && action === 'branch_selected') {
        return `Step ${transition.step_index || '?'} selected branch ${transition.selected_branch || '?'}`;
    }
    if (phase === 'adversarial') {
        const idx = transition.step_index || '?';
        const iter = transition.iteration || '?';
        const iterTotal = transition.iteration_total || '?';
        if (action === 'iteration_started') {
            return `Step ${idx} - refining (critic round ${iter}/${iterTotal})`;
        }
        if (action === 'iteration_completed') {
            return `Step ${idx} - critic round ${iter}/${iterTotal} complete`;
        }
    }
    if (phase === 'fallback' && action === 'reroute') {
        return `Rerouted ${transition.from_worker || '?'} -> ${transition.to_worker || '?'}`;
    }
    return 'Updating run status...';
}

function nodeHtml(node) {
    const reason = node.type === 'fallback' && node.meta?.reason
        ? `<div class="plan-node-note">${esc(node.meta.reason)}</div>`
        : '';
    return `
        <div class="plan-node" data-status="${esc(node.status)}" data-type="${esc(node.type)}">
            <span class="plan-node-icon">${iconForStatus(node.status)}</span>
            <div class="plan-node-body">
                <div class="plan-node-label">${esc(node.label)}</div>
                ${reason}
            </div>
        </div>
    `;
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
            const ai = Number(a.id.split(':')[1] || 0);
            const bi = Number(b.id.split(':')[1] || 0);
            return ai - bi;
        });
    return { nodeMap, children, stepNodes };
}

export class FriendlyPlanRenderer {
    constructor(container) {
        this.container = container;
        this.lastSignature = '';
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
            events: (snapshot.events || []).length,
        });
        if (signature === this.lastSignature) {
            return;
        }
        this.lastSignature = signature;

        const { nodeMap, children, stepNodes } = collectStepGroups(snapshot);
        const planNode = nodeMap.get('plan');
        const planHtml = planNode ? nodeHtml(planNode) : '';
        const badges = snapshot.inferred
            ? '<span class="plan-badge">inferred timeline - limited detail</span>'
            : '';
        const renderedIds = new Set();

        let stepsHtml = '';
        for (const step of stepNodes) {
            renderedIds.add(step.id);
            const childIds = children.get(step.id) || [];
            const childNodes = childIds.map((id) => nodeMap.get(id)).filter(Boolean);
            for (const child of childNodes) {
                renderedIds.add(child.id);
            }
            const childHtml = childNodes.map((node) => nodeHtml(node)).join('');
            stepsHtml += `
                <section class="plan-step-group">
                    ${nodeHtml(step)}
                    ${childHtml ? `<div class="plan-step-children">${childHtml}</div>` : ''}
                </section>
            `;
        }

        const rootChildren = (children.get('plan') || [])
            .map((id) => nodeMap.get(id))
            .filter((node) => node && !renderedIds.has(node.id))
            .map((node) => {
                renderedIds.add(node.id);
                return nodeHtml(node);
            })
            .join('');

        const dangling = (snapshot.nodes || [])
            .filter((node) => node.id !== 'plan' && !renderedIds.has(node.id))
            .map((node) => {
                renderedIds.add(node.id);
                return nodeHtml(node);
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
    }
}
