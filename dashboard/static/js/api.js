export async function postDispatch(data) {
    const res = await fetch('/api/dispatch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    });
    if (!res.ok) {
        const text = await res.text();
        throw new Error(`Dispatch failed: ${res.status} ${text}`);
    }
    return res.json();
}

export async function getLive() {
    const res = await fetch('/api/live');
    if (!res.ok) throw new Error('Live fetch failed');
    return res.json();
}

export async function getRuns(limitOrOptions = 3) {
    const options = typeof limitOrOptions === 'number' ? { limit: limitOrOptions } : (limitOrOptions || {});
    const params = new URLSearchParams();
    if (options.limit !== undefined && options.limit !== null) params.set('limit', String(options.limit));
    if (options.before) params.set('before', String(options.before));
    if (options.q) params.set('q', String(options.q));
    if (options.mode) params.set('mode', String(options.mode));
    if (options.success) params.set('success', String(options.success));
    if (options.worker) params.set('worker', String(options.worker));
    const res = await fetch(`/api/runs?${params.toString()}`);
    if (!res.ok) {
        throw new Error('Runs fetch failed');
    }
    return res.json();
}

export async function getRun(runId) {
    const res = await fetch(`/api/runs/${encodeURIComponent(runId)}`);
    if (!res.ok) {
        throw new Error(`Run detail fetch failed: ${res.status}`);
    }
    return res.json();
}

export async function getArtifact(runId, name) {
    const res = await fetch(`/api/runs/${encodeURIComponent(runId)}/artifact/${encodeURIComponent(name)}`);
    if (!res.ok) {
        throw new Error(`Artifact fetch failed: ${res.status}`);
    }
    return res.text();
}

export async function getBudget(mode, gen, critic) {
    const params = new URLSearchParams();
    if (mode) params.set('mode', mode);
    if (gen) params.set('generator', gen);
    if (critic) params.set('critic', critic);

    const res = await fetch(`/api/dispatch/budget?${params.toString()}`);
    if (!res.ok) return null;
    return res.json();
}
