import { getLive } from '../api.js';
import { SSEClient } from '../sse.js';
import { StreamRenderer } from '../components/stream.js';

export async function renderLive(container) {
    container.innerHTML = `<h2>Live Runs</h2><div id="live-list">Loading...</div>`;
    
    try {
        const res = await getLive();
        const list = document.getElementById('live-list');
        list.innerHTML = '';
        if (!res.running || res.running.length === 0) {
            list.innerHTML = '<p>No running dispatches.</p>';
            return;
        }
        res.running.forEach(run => {
            const card = document.createElement('div');
            card.className = 'card';
            
            card.innerHTML = `
                <h3>Run: ${run.run_id}</h3>
                <p style="color: var(--fg-1); margin-top: 0;">Mode: ${run.mode} | Generator: ${run.generator?.join(',') || 'N/A'}</p>
                <div id="live-stream-${run.run_id}"></div>
            `;
            list.appendChild(card);

            const renderer = new StreamRenderer(document.getElementById(`live-stream-${run.run_id}`));
            const sse = new SSEClient(`/api/sse/${run.run_id}`);
            sse.subscribe(
                (e) => renderer.appendEvent(e),
                (e) => {
                    const p = document.createElement('p');
                    p.textContent = `[done]`;
                    renderer.container.appendChild(p);
                },
                (e) => console.error('SSE Error', e)
            );
        });
    } catch (err) {
        container.innerHTML += `<p style="color:var(--bad)">Failed to load live runs.</p>`;
    }
}
