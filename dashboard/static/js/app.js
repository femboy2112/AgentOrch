import { initRouter } from './router.js';
import { getMode, onModeChange, setMode } from './presentation_mode.js';

const THEME_KEY = 'dashboard:theme';

function applyTheme(theme) {
    const isLight = theme === 'light';
    document.documentElement.classList.toggle('light', isLight);
    const btn = document.getElementById('theme-toggle');
    if (btn) {
        btn.textContent = isLight ? 'Use Dark' : 'Use Light';
    }
}

function applyPresentationMode(mode) {
    const btn = document.getElementById('presentation-toggle');
    if (!btn) {
        return;
    }
    const friendly = mode === 'friendly';
    btn.textContent = friendly ? 'Friendly View' : 'Professional View';
}

const savedTheme = localStorage.getItem(THEME_KEY);
applyTheme(savedTheme === 'light' ? 'light' : 'dark');

document.getElementById('theme-toggle').addEventListener('click', () => {
    const nowLight = !document.documentElement.classList.contains('light');
    const next = nowLight ? 'light' : 'dark';
    localStorage.setItem(THEME_KEY, next);
    applyTheme(next);
});

const savedMode = getMode();
applyPresentationMode(savedMode);

document.getElementById('presentation-toggle').addEventListener('click', () => {
    const next = getMode() === 'friendly' ? 'professional' : 'friendly';
    setMode(next);
});

onModeChange((mode) => {
    applyPresentationMode(mode);
});

initRouter();
