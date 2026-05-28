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

export async function getRuns(limit = 3) {
    const params = new URLSearchParams();
    params.set('limit', String(limit));
    const res = await fetch(`/api/runs?${params.toString()}`);
    if (!res.ok) {
        throw new Error('Runs fetch failed');
    }
    return res.json();
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
