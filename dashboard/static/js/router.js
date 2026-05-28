import { renderDispatch } from './pages/dispatch.js';
import { renderLive } from './pages/live.js';

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
    } else if (hash === '#/live') {
        renderLive(app);
    } else if (hash === '#/runs' || hash.startsWith('#/runs/')) {
        app.innerHTML = '<h2>Runs (Phase 4)</h2>';
    } else {
        window.location.hash = '#/dispatch';
    }
}
