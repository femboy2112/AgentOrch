import { postDispatch, getBudget } from '../api.js';
import { SSEClient } from '../sse.js';
import { StreamRenderer } from '../components/stream.js';

export function renderDispatch(container) {
    container.innerHTML = `
        <h2>Dispatch</h2>
        <div class="card">
            <div class="form-group">
                <label>Instruction</label>
                <textarea id="d-inst" rows="4">Create a hello world python script</textarea>
            </div>
            <div class="form-group">
                <label>Mode</label>
                <select id="d-mode">
                    <option value="direct">direct</option>
                    <option value="adversarial" selected>adversarial</option>
                </select>
            </div>
            <div class="form-group">
                <label>Generator Chain</label>
                <input type="text" id="d-gen" value="codex,agy,grok">
            </div>
            <div class="form-group" id="d-critic-group">
                <label>Critic Chain</label>
                <input type="text" id="d-critic" value="agy,codex">
            </div>
            <div class="form-group">
                <label>
                    <input type="checkbox" id="d-stay" checked> Stay on this page
                </label>
            </div>
            <div id="d-budget" style="margin-bottom: var(--s-4); font-size: 0.9em; color: var(--fg-1);">
                Loading budget...
            </div>
            <button id="d-btn" class="btn btn-primary">DISPATCH</button>
        </div>
        <div id="d-stream-container" style="margin-top: 20px; display: none;"></div>
    `;

    const modeEl = document.getElementById('d-mode');
    const genEl = document.getElementById('d-gen');
    const criticEl = document.getElementById('d-critic');
    const criticGroup = document.getElementById('d-critic-group');
    const budgetEl = document.getElementById('d-budget');

    async function updateBudget() {
        const mode = modeEl.value;
        if (mode !== 'adversarial') {
            criticGroup.style.display = 'none';
        } else {
            criticGroup.style.display = 'block';
        }

        const b = await getBudget(mode, genEl.value, criticEl.value);
        if (b && b.rows) {
            budgetEl.innerHTML = '<strong>Budget:</strong> ' + b.rows.map(r => 
                `${r.worker} (${r.model}): ${Math.round(r.max_output_bytes/1024)}kb`
            ).join(', ');
        } else {
            budgetEl.innerHTML = 'Budget data unavailable.';
        }
    }

    modeEl.addEventListener('change', updateBudget);
    genEl.addEventListener('input', updateBudget);
    criticEl.addEventListener('input', updateBudget);
    updateBudget();

    document.getElementById('d-btn').addEventListener('click', async () => {
        const inst = document.getElementById('d-inst').value;
        const mode = modeEl.value;
        const gen = genEl.value.split(',').map(s => s.trim()).filter(Boolean);
        const critic = criticEl.value.split(',').map(s => s.trim()).filter(Boolean);
        const stay = document.getElementById('d-stay').checked;

        const payload = {
            instruction: inst,
            mode: mode,
            generator_chain: gen.length ? gen : undefined
        };
        if (mode === 'adversarial') {
            payload.critic_chain = critic.length ? critic : undefined;
        }

        try {
            const res = await postDispatch(payload);
            
            if (stay) {
                const sContainer = document.getElementById('d-stream-container');
                sContainer.style.display = 'block';
                sContainer.innerHTML = `<h3>Run: ${res.run_id}</h3><div id="d-stream"></div>`;
                const renderer = new StreamRenderer(document.getElementById('d-stream'));
                const sse = new SSEClient(`/api/sse/${res.run_id}`);
                sse.subscribe(
                    (e) => renderer.appendEvent(e),
                    (e) => {
                        const p = document.createElement('p');
                        p.textContent = '[done]';
                        renderer.container.appendChild(p);
                    },
                    (e) => console.error('SSE Error', e)
                );
            } else {
                window.location.hash = '#/live';
            }
        } catch (err) {
            alert(err.message);
        }
    });
}
