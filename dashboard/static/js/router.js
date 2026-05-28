import { renderDispatch } from './pages/dispatch.js';
import { renderLive } from './pages/live.js';
import { renderRuns } from './pages/runs.js';
import { renderRunDetail } from './pages/run_detail.js';

export function initRouter() {
    window.addEventListener('hashchange', handleRoute);
    handleRoute();
}

function handleRoute() {
    const hash = window.location.hash || '#/dispatch';
    const app = document.getElementById('app');
    app.innerHTML = '';

    if (hash === '#/dispatch') {
        renderDispatch(app);
        return;
    }
    if (hash === '#/live') {
        renderLive(app);
        return;
    }
    if (hash === '#/runs') {
        renderRuns(app);
        return;
    }
    if (hash.startsWith('#/runs/')) {
        const runId = decodeURIComponent(hash.slice('#/runs/'.length));
        if (!runId) {
            window.location.hash = '#/runs';
            return;
        }
        renderRunDetail(app, runId);
        return;
    }

    window.location.hash = '#/dispatch';
}
