import { getRuns } from '../api.js';

function esc(value) {
    return String(value ?? '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
}

function chainToText(value) {
    if (Array.isArray(value)) {
        return value.join(',');
    }
    return String(value || '').trim();
}

function renderEmptyState(label, title) {
    return `<span title="${esc(title)}">${esc(label)}</span>`;
}

function renderChainCell(value, roleName) {
    const text = chainToText(value);
    if (text) {
        return esc(text);
    }
    return renderEmptyState(
        `${roleName} chain not recorded`,
        `This run metadata did not include a ${roleName.toLowerCase()} chain.`,
    );
}

function formatStart(ts) {
    if (!ts) {
        return null;
    }
    const date = new Date(Number(ts) * 1000);
    if (Number.isNaN(date.getTime())) {
        return null;
    }
    return date.toLocaleString();
}

function renderStartCell(ts) {
    const text = formatStart(ts);
    if (text) {
        return esc(text);
    }
    return renderEmptyState('Start time unavailable', 'No valid start timestamp was recorded for this run.');
}

export async function renderRuns(container) {
    container.innerHTML = `
        <h2>Runs</h2>
        <div class="card runs-controls">
            <div class="runs-grid">
                <label>
                    Search
                    <input type="text" id="runs-q" placeholder="run id, mode, chain, instruction">
                </label>
                <label>
                    Mode
                    <select id="runs-mode">
                        <option value="">all</option>
                        <option value="direct">direct</option>
                        <option value="adversarial">adversarial</option>
                        <option value="feedback">feedback</option>
                        <option value="cascade">cascade</option>
                        <option value="master">master</option>
                    </select>
                </label>
                <label>
                    Success
                    <select id="runs-success">
                        <option value="">all</option>
                        <option value="true">success</option>
                        <option value="false">failed</option>
                    </select>
                </label>
                <label>
                    Worker
                    <input type="text" id="runs-worker" placeholder="codex, agy, grok">
                </label>
            </div>
            <div class="runs-actions">
                <button id="runs-apply" class="btn">Apply Filters</button>
                <button id="runs-reset" class="btn">Reset</button>
            </div>
        </div>
        <div class="card">
            <table class="runs-table">
                <thead>
                    <tr>
                        <th>Run</th>
                        <th>Mode</th>
                        <th>Generator</th>
                        <th>Critic</th>
                        <th>Duration</th>
                        <th>Status</th>
                        <th>Files</th>
                        <th>Started</th>
                    </tr>
                </thead>
                <tbody id="runs-body">
                    <tr><td colspan="8">Loading...</td></tr>
                </tbody>
            </table>
            <div class="runs-footer">
                <button id="runs-more" class="btn" style="display:none">Load more</button>
            </div>
        </div>
    `;

    const qEl = document.getElementById('runs-q');
    const modeEl = document.getElementById('runs-mode');
    const successEl = document.getElementById('runs-success');
    const workerEl = document.getElementById('runs-worker');
    const bodyEl = document.getElementById('runs-body');
    const moreBtn = document.getElementById('runs-more');

    let nextBefore = null;

    async function fetchPage({ reset }) {
        if (reset) {
            nextBefore = null;
            bodyEl.innerHTML = '<tr><td colspan="8">Loading...</td></tr>';
        }

        const params = {
            limit: 50,
            before: reset ? null : nextBefore,
            q: qEl.value.trim() || null,
            mode: modeEl.value || null,
            success: successEl.value || null,
            worker: workerEl.value.trim() || null,
        };

        let payload;
        try {
            payload = await getRuns(params);
        } catch (_err) {
            bodyEl.innerHTML = '<tr><td colspan="8"><span title="The runs list endpoint did not return data.">Failed to load runs. Try again.</span></td></tr>';
            moreBtn.style.display = 'none';
            return;
        }

        if (reset) {
            bodyEl.innerHTML = '';
        }

        const rows = payload.runs || [];
        if (rows.length === 0 && reset) {
            bodyEl.innerHTML = '<tr><td colspan="8"><span title="No rows matched the current filters.">No runs match these filters.</span></td></tr>';
        }

        for (const row of rows) {
            const tr = document.createElement('tr');
            tr.className = 'runs-row';
            tr.tabIndex = 0;
            tr.innerHTML = `
                <td>${esc(row.run_id)}</td>
                <td>${esc(row.mode)}</td>
                <td>${renderChainCell(row.generator, 'Generator')}</td>
                <td>${renderChainCell(row.critic, 'Critic')}</td>
                <td>${Number(row.duration_s || 0).toFixed(2)}s</td>
                <td>${row.success ? 'ok' : 'fail'}</td>
                <td>${Number(row.changed_files_count || 0)}</td>
                <td>${renderStartCell(row.started_at)}</td>
            `;
            tr.addEventListener('click', () => {
                window.location.hash = `#/runs/${row.run_id}`;
            });
            tr.addEventListener('keydown', (evt) => {
                if (evt.key === 'Enter' || evt.key === ' ') {
                    evt.preventDefault();
                    window.location.hash = `#/runs/${row.run_id}`;
                }
            });
            bodyEl.appendChild(tr);

            if (row.instruction_preview) {
                const preview = document.createElement('tr');
                preview.className = 'runs-preview';
                preview.innerHTML = `<td colspan="8">${esc(row.instruction_preview)}</td>`;
                bodyEl.appendChild(preview);
            }
        }

        nextBefore = payload.next_before || null;
        moreBtn.style.display = nextBefore ? 'inline-flex' : 'none';
    }

    document.getElementById('runs-apply').addEventListener('click', () => fetchPage({ reset: true }));
    document.getElementById('runs-reset').addEventListener('click', () => {
        qEl.value = '';
        modeEl.value = '';
        successEl.value = '';
        workerEl.value = '';
        fetchPage({ reset: true });
    });
    moreBtn.addEventListener('click', () => fetchPage({ reset: false }));

    await fetchPage({ reset: true });
}
