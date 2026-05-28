import { initRouter } from './router.js';

document.getElementById('theme-toggle').addEventListener('click', () => {
    document.documentElement.classList.toggle('light');
});

initRouter();
