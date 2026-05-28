import { getBudget, getRuns, postDispatch } from '../api.js';
import { subscribe } from '../sse.js';
import { StreamRenderer } from '../components/stream.js';

const WORKERS = ['codex', 'agy', 'grok', 'claude'];

function csvToList(raw) {
    return raw
        .split(',')
        .map((s) => s.trim())
        .filter(Boolean);
}

export function renderDispatch(container) {
    container.innerHTML = `
        <h2>Dispatch</h2>
        <div class="card">
            <div class="form-group">
                <label for="d-inst">Instruction</label>
                <textarea id="d-inst" rows="14"></textarea>
            </div>
            <div class="form-group">
                <label for="d-mode">Mode</label>
                <select id="d-mode">
                    <option value="direct">direct</option>
                    <option value="adversarial" selected>adversarial</option>
                    <option value="feedback">feedback</option>
                    <option value="cascade">cascade</option>
                    <option value="master">master</option>
                </select>
            </div>
            <div class="form-group">
                <label for="d-gen">Generator Chain</label>
                <input type="text" id="d-gen" list="workers" value="codex,agy,grok">
            </div>
            <div class="form-group" id="d-critic-group">
                <label for="d-critic">Critic Chain</label>
                <input type="text" id="d-critic" list="workers" value="agy,codex,grok">
            </div>
            <datalist id="workers">
                ${WORKERS.map((w) => `<option value="${w}"></option>`).join('')}
            </datalist>
            <div class="form-group">
                <label for="d-test-cmd">Test Command</label>
                <input type="text" id="d-test-cmd" placeholder="python -m pytest -q">
            </div>
            <div class="form-group inline-checks">
                <label><input type="checkbox" id="d-web"> web-search</label>
                <label><input type="checkbox" id="d-no-fallback"> no-fallback</label>
                <label><input type="checkbox" id="d-stay" checked> Stay on this page</label>
            </div>
            <div id="d-budget" class="budget-line">Loading budget...</div>
            <button id="d-btn" class="btn btn-primary">DISPATCH</button>
            <div id="d-last3" class="last-runs"></div>
        </div>
        <div id="d-stream-container" class="dispatch-stream"></div>
    `;

    const modeEl = document.getElementById('d-mode');
    const genEl = document.getElementById('d-gen');
    const criticEl = document.getElementById('d-critic');
    const criticGroup = document.getElementById('d-critic-group');
    const budgetEl = document.getElementById('d-budget');

    async function loadLast3() {
        const target = document.getElementById('d-last3');
        target.textContent = '';
        try {
            const resp = await getRuns(3);
            if (!resp.runs || resp.runs.length === 0) {
                target.textContent = 'No previous dispatches yet.';
                return;
            }
            const title = document.createElement('p');
            title.textContent = 'Last 3 dispatches:';
            target.appendChild(title);
            const ul = document.createElement('ul');
            ul.className = 'last-runs-list';
            for (const run of resp.runs) {
                const li = document.createElement('li');
                const a = document.createElement('a');
                a.href = `#/runs/${run.run_id}`;
                a.textContent = `${run.run_id} (${run.mode})`;
                li.appendChild(a);
                ul.appendChild(li);
            }
            target.appendChild(ul);
        } catch (_err) {
            target.textContent = 'Could not load recent dispatches.';
        }
    }

    async function updateBudget() {
        const mode = modeEl.value;
        criticGroup.style.display = mode === 'adversarial' ? 'block' : 'none';
        const b = await getBudget(mode, genEl.value, criticEl.value);
        if (b && b.rows) {
            budgetEl.innerHTML = '<strong>Budget:</strong> ' + b.rows.map((r) =>
                `${r.worker} (${r.model}) ${Math.round(r.max_output_bytes / 1024)}kb/${Math.round(r.stall_seconds)}s`
            ).join(', ');
        } else {
            budgetEl.textContent = 'Budget data unavailable.';
        }
    }

    modeEl.addEventListener('change', updateBudget);
    genEl.addEventListener('input', updateBudget);
    criticEl.addEventListener('input', updateBudget);
    updateBudget();
    loadLast3();

    document.getElementById('d-btn').addEventListener('click', async () => {
        const inst = document.getElementById('d-inst').value;
        const mode = modeEl.value;
        const gen = csvToList(genEl.value);
        const critic = csvToList(criticEl.value);
        const stay = document.getElementById('d-stay').checked;
        const noFallback = document.getElementById('d-no-fallback').checked;
        const webSearch = document.getElementById('d-web').checked;
        const testCmd = document.getElementById('d-test-cmd').value.trim();

        const payload = {
            instruction: inst,
            mode,
            generator_chain: gen.length ? gen : undefined,
            fallback: !noFallback,
            web_search: webSearch,
            test_cmd: testCmd || undefined,
        };
        if (mode === 'adversarial') {
            payload.critic_chain = critic.length ? critic : undefined;
        }

        try {
            const res = await postDispatch(payload);
            sessionStorage.setItem('dashboard:last_run_id', res.run_id);

            if (!stay) {
                window.location.hash = '#/live';
                return;
            }

            const holder = document.getElementById('d-stream-container');
            holder.style.display = 'block';
            holder.innerHTML = `<h3>Run: ${res.run_id}</h3><div id="d-stream"></div>`;
            const renderer = new StreamRenderer(document.getElementById('d-stream'));
            subscribe(
                res.run_id,
                (event) => renderer.appendEvent(event),
                () => {
                    const p = document.createElement('p');
                    p.textContent = '[done]';
                    renderer.viewport.appendChild(p);
                },
                (err) => {
                    console.error('SSE error', err);
                }
            );
            loadLast3();
        } catch (err) {
            alert(err.message);
        }
    });
}
