import { getArtifact, getRun } from '../api.js';
import { mountRunRenderers } from '../components/mode_render.js';

function esc(value) {
    return String(value ?? '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
}

function parseEventsNdjson(raw) {
    const out = [];
    for (const line of raw.split('\n')) {
        const trimmed = line.trim();
        if (!trimmed) {
            continue;
        }
        try {
            out.push(JSON.parse(trimmed));
        } catch (_err) {
            // Ignore partial trailing writes while a run is still appending.
        }
    }
    return out;
}

function formatChain(value) {
    if (Array.isArray(value)) {
        return value.join(',');
    }
    return String(value || '');
}

export async function renderRunDetail(container, runId) {
    container.innerHTML = '<h2>Run Detail</h2><div class="card">Loading...</div>';

    let payload;
    try {
        payload = await getRun(runId);
    } catch (_err) {
        container.innerHTML = `<h2>Run Detail</h2><div class="card">Run not found: ${esc(runId)}</div>`;
        return;
    }

    const meta = payload.meta || {};
    const tabs = ['Stream', 'Prompt', 'Stdout', 'Stderr', 'Diff', 'Quality'];

    container.innerHTML = `
        <h2>Run ${esc(runId)}</h2>
        <div class="card run-header">
            <div><strong>Mode:</strong> ${esc(meta.mode || '')}</div>
            <div><strong>Generator:</strong> ${esc(formatChain(meta.generator))}</div>
            <div><strong>Critic:</strong> ${esc(formatChain(meta.critic || ''))}</div>
            <div><strong>Duration:</strong> ${Number(meta.duration_s || 0).toFixed(2)}s</div>
            <div><strong>Status:</strong> ${meta.success ? 'ok' : 'fail'}</div>
            <div><strong>Changed Files:</strong> ${Array.isArray(meta.changed_files) ? meta.changed_files.length : 0}</div>
        </div>
        <div class="card">
            <div class="tab-bar">
                ${tabs.map((name, idx) => `<button class="btn tab-btn" data-tab="${name.toLowerCase()}" data-active="${idx === 0 ? '1' : '0'}">${name}</button>`).join('')}
            </div>
            <div id="run-detail-body" class="run-detail-body"></div>
        </div>
    `;

    const bodyEl = document.getElementById('run-detail-body');
    const tabButtons = Array.from(container.querySelectorAll('.tab-btn'));
    let streamViewDestroy = null;

    async function renderTab(tab) {
        for (const btn of tabButtons) {
            btn.dataset.active = btn.dataset.tab === tab ? '1' : '0';
        }

        if (tab === 'stream') {
            if (streamViewDestroy) {
                streamViewDestroy();
                streamViewDestroy = null;
            }
            const host = document.createElement('div');
            bodyEl.innerHTML = '';
            bodyEl.appendChild(host);
            const rendererView = mountRunRenderers(host, runId);
            streamViewDestroy = () => rendererView.destroy();
            try {
                const raw = await getArtifact(runId, 'events');
                for (const evt of parseEventsNdjson(raw)) {
                    rendererView.appendEvent(evt);
                }
            } catch (_err) {
                const msg = document.createElement('p');
                msg.textContent = 'Unable to load events stream.';
                bodyEl.appendChild(msg);
            }
            return;
        }

        if (streamViewDestroy) {
            streamViewDestroy();
            streamViewDestroy = null;
        }

        if (tab === 'quality') {
            const pretty = JSON.stringify(meta.quality || {}, null, 2);
            bodyEl.innerHTML = `<pre class="artifact-text">${esc(pretty)}</pre>`;
            return;
        }

        const artifactMap = {
            prompt: 'prompt',
            stdout: 'stdout',
            stderr: 'stderr',
            diff: 'diff',
        };
        const artifact = artifactMap[tab];
        if (!artifact) {
            bodyEl.innerHTML = '<p>Unknown tab.</p>';
            return;
        }

        bodyEl.innerHTML = '<p>Loading artifact...</p>';
        try {
            const raw = await getArtifact(runId, artifact);
            bodyEl.innerHTML = `<pre class="artifact-text">${esc(raw)}</pre>`;
        } catch (_err) {
            bodyEl.innerHTML = '<p>Artifact unavailable.</p>';
        }
    }

    tabButtons.forEach((btn) => {
        btn.addEventListener('click', () => {
            renderTab(btn.dataset.tab);
        });
    });

    await renderTab('stream');
}
