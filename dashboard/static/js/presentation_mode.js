const MODE_KEY = 'dashboard:presentation_mode';
const DEFAULT_MODE = 'professional';
const MODES = new Set(['professional', 'friendly']);

function normalizeMode(mode) {
    return MODES.has(mode) ? mode : DEFAULT_MODE;
}

export function getMode() {
    return normalizeMode(localStorage.getItem(MODE_KEY));
}

export function setMode(mode) {
    const next = normalizeMode(mode);
    localStorage.setItem(MODE_KEY, next);
    const evt = new CustomEvent('modechange', { detail: { mode: next } });
    window.dispatchEvent(evt);
    document.dispatchEvent(evt);
}

export function onModeChange(cb) {
    const handler = (evt) => cb(normalizeMode(evt?.detail?.mode));
    window.addEventListener('modechange', handler);
    return () => window.removeEventListener('modechange', handler);
}
