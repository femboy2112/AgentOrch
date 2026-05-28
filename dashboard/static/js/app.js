import { initRouter } from './router.js';

const THEME_KEY = 'dashboard:theme';

function applyTheme(theme) {
    const isLight = theme === 'light';
    document.documentElement.classList.toggle('light', isLight);
    const btn = document.getElementById('theme-toggle');
    if (btn) {
        btn.textContent = isLight ? 'Use Dark' : 'Use Light';
    }
}

const savedTheme = localStorage.getItem(THEME_KEY);
applyTheme(savedTheme === 'light' ? 'light' : 'dark');

document.getElementById('theme-toggle').addEventListener('click', () => {
    const nowLight = !document.documentElement.classList.contains('light');
    const next = nowLight ? 'light' : 'dark';
    localStorage.setItem(THEME_KEY, next);
    applyTheme(next);
});

initRouter();
