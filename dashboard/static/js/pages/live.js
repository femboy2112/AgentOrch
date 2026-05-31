import { getBudget, getLive } from '../api.js';
import { subscribe } from '../sse.js';
import { mountRunRenderers } from '../components/mode_render.js';

function elapsed(startedAt) {
    const age = Math.max(0, Date.now() / 1000 - Number(startedAt || 0));
    const m = Math.floor(age / 60);
    const s = Math.floor(age % 60);
    return `${m}m ${s}s`;
}

export async function renderLive(container) {
    container.innerHTML = `<h2>Live Runs</h2><div id="live-list">Loading...</div>`;

    let runs;
    try {
        const resp = await getLive();
        runs = resp.running || [];
    } catch (_err) {
        container.innerHTML += `<p style="color:var(--bad)" title="The live endpoint did not return active dispatch data.">Failed to load live runs. Refresh and try again.</p>`;
        return;
    }

    const list = document.getElementById('live-list');
    if (runs.length === 0) {
        list.innerHTML = '<p title="No dispatch currently reports as running from the dashboard API.">No active dispatches right now.</p>';
        return;
    }
    list.innerHTML = '';

    const focusRunId = sessionStorage.getItem('dashboard:last_run_id');
    const expandAll = runs.length === 1;

    // Build every card synchronously (no awaits) so the DOM is fully
    // rendered before any network call fires. The previous sequential
    // `await getBudget(...)` inside the loop blocked SSE subscription
    // for runs N+1..K behind runs 1..N — visibly slow with multiple
    // active runs.
    const wires = [];
    for (const run of runs) {
        const card = document.createElement('section');
        card.className = 'card';

        const details = document.createElement('details');
        details.open = expandAll || run.run_id === focusRunId;

        const summary = document.createElement('summary');
        summary.className = 'live-summary';
        const sourceLabel = run.source === 'cli' ? '<span class="source-cli">CLI</span>' : '';
        summary.innerHTML = `
            ${sourceLabel}<strong>${run.run_id}</strong>
            <span class="live-meta">mode=${run.mode} gen=${(run.generator || []).join(',')}</span>
            <span id="elapsed-${run.run_id}" class="live-meta">${elapsed(run.started_at)}</span>
            <span id="tokens-${run.run_id}" class="live-meta">tokens=0</span>
            <span id="budget-${run.run_id}" class="live-meta">budget=0%</span>
        `;
        details.appendChild(summary);

        const streamHost = document.createElement('div');
        streamHost.id = `live-stream-${run.run_id}`;
        details.appendChild(streamHost);
        card.appendChild(details);
        list.appendChild(card);

        const elapsedEl = summary.querySelector(`#elapsed-${run.run_id}`);
        const tokensEl = summary.querySelector(`#tokens-${run.run_id}`);
        const budgetEl = summary.querySelector(`#budget-${run.run_id}`);

        const timer = setInterval(() => {
            elapsedEl.textContent = elapsed(run.started_at);
        }, 1000);
        details.addEventListener('toggle', () => {
            if (!document.body.contains(details)) {
                clearInterval(timer);
            }
        });

        const rendererView = mountRunRenderers(streamHost, run.run_id);
        wires.push({ run, rendererView, tokensEl, budgetEl });
    }

    // Now fire every budget fetch in parallel; whichever resolves first
    // updates its budget cell. None blocks any other run's SSE subscription.
    for (const { run, rendererView, tokensEl, budgetEl } of wires) {
        let outTokens = 0;
        let maxBytes = 0;
        // Start the SSE subscription immediately — don't wait on the
        // (slow, parallelizable) budget fetch. Budget cell updates
        // independently when its fetch resolves.
        subscribe(
            run.run_id,
            (e) => {
                rendererView.appendEvent(e);
                if (e.kind === 'usage') {
                    outTokens += Number(e.data?.out_tokens || 0);
                    tokensEl.textContent = `tokens=${outTokens}`;
                    if (maxBytes > 0) {
                        const approxOutBytes = outTokens * 4;
                        const pct = Math.min(999, Math.round((approxOutBytes / maxBytes) * 100));
                        budgetEl.textContent = `budget=${pct}%`;
                    }
                }
            },
            () => {
                budgetEl.textContent = `${budgetEl.textContent} done`;
            },
            (err) => {
                console.error('SSE error', err);
            }
        );
        // Budget is a soft signal; failures don't affect the stream.
        getBudget(run.mode, (run.generator || []).join(','), (run.critic || []).join(','))
            .then((b) => {
                if (b && b.rows) {
                    maxBytes = b.rows.reduce((sum, row) => sum + Number(row.max_output_bytes || 0), 0);
                }
            })
            .catch(() => {});
    }
}
