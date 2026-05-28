const FILTERS = [
    { key: 'reasoning', label: 'reasoning' },
    { key: 'messages', label: 'messages' },
    { key: 'tools', label: 'tools' },
    { key: 'usage', label: 'usage' },
    { key: 'stderr', label: 'stderr' },
];

function toClock(ts) {
    return new Date(ts * 1000).toLocaleTimeString();
}

function eventFilterKind(e) {
    if (e.kind === 'reasoning') {
        return 'reasoning';
    }
    if (e.kind === 'tool_call' || e.kind === 'tool_result') {
        return 'tools';
    }
    if (e.kind === 'usage') {
        return 'usage';
    }
    if (e.kind === 'stderr' || e.kind === 'watchdog') {
        return 'stderr';
    }
    return 'messages';
}

export class StreamRenderer {
    constructor(container) {
        this.container = container;
        this.lastEvent = null;
        this.lastReasoningP = null;
        this.autoScroll = true;
        this.view = 'flow';
        this.filters = {
            reasoning: true,
            messages: true,
            tools: true,
            usage: true,
            stderr: true,
        };
        this._build();
    }

    _build() {
        this.container.innerHTML = '';

        this.toolbar = document.createElement('div');
        this.toolbar.className = 'stream-toolbar';

        const pills = document.createElement('div');
        pills.className = 'stream-pills';
        for (const item of FILTERS) {
            const b = document.createElement('button');
            b.className = 'btn btn-pill';
            b.textContent = item.label;
            b.dataset.filter = item.key;
            b.dataset.active = '1';
            b.addEventListener('click', () => {
                this.filters[item.key] = !this.filters[item.key];
                b.dataset.active = this.filters[item.key] ? '1' : '0';
            });
            pills.appendChild(b);
        }

        const layoutToggle = document.createElement('button');
        layoutToggle.className = 'btn';
        layoutToggle.textContent = 'Cards View';
        layoutToggle.addEventListener('click', () => {
            this.view = this.view === 'flow' ? 'cards' : 'flow';
            this.viewport.classList.toggle('stream-cards', this.view === 'cards');
            layoutToggle.textContent = this.view === 'flow' ? 'Cards View' : 'Flow View';
            this.lastReasoningP = null;
            this.lastEvent = null;
        });

        this.toolbar.appendChild(pills);
        this.toolbar.appendChild(layoutToggle);

        this.viewport = document.createElement('div');
        this.viewport.className = 'stream-viewport';
        this.viewport.addEventListener('scroll', () => {
            const delta = this.viewport.scrollHeight - this.viewport.scrollTop - this.viewport.clientHeight;
            this.autoScroll = delta < 20;
            this.jumpBtn.style.display = this.autoScroll ? 'none' : 'inline-flex';
        });

        this.jumpBtn = document.createElement('button');
        this.jumpBtn.className = 'btn stream-jump';
        this.jumpBtn.textContent = '▼ jump to live';
        this.jumpBtn.style.display = 'none';
        this.jumpBtn.addEventListener('click', () => {
            this.viewport.scrollTop = this.viewport.scrollHeight;
            this.autoScroll = true;
            this.jumpBtn.style.display = 'none';
        });

        this.container.appendChild(this.toolbar);
        this.container.appendChild(this.viewport);
        this.container.appendChild(this.jumpBtn);
    }

    _appendNode(node) {
        this.viewport.appendChild(node);
        if (this.autoScroll) {
            this.viewport.scrollTop = this.viewport.scrollHeight;
        } else {
            this.jumpBtn.style.display = 'inline-flex';
        }
    }

    _newLine(text, cls = '') {
        const p = document.createElement('p');
        p.className = cls;
        p.textContent = text;
        return p;
    }

    appendEvent(e) {
        const filterKey = eventFilterKind(e);
        if (!this.filters[filterKey]) {
            return;
        }

        const stamp = toClock(e.ts);
        const branchTag = e.branch !== null && e.branch !== undefined ? `branch=${e.branch} ` : '';

        if (this.view === 'flow' && e.kind === 'reasoning') {
            const canMerge = (
                this.lastEvent?.kind === 'reasoning'
                && this.lastEvent?.worker === e.worker
                && this.lastEvent?.branch === e.branch
                && (e.ts - this.lastEvent.ts) <= 1.5
            );
            if (canMerge && this.lastReasoningP) {
                this.lastReasoningP.textContent += e.text;
            } else {
                this.lastReasoningP = this._newLine(
                    `${stamp}  ${branchTag}${e.worker} / ${e.model} / ${e.effort}  ·  reasoning ▾\n${e.text}`
                );
                this._appendNode(this.lastReasoningP);
            }
            this.lastEvent = e;
            return;
        }

        this.lastReasoningP = null;
        let line = '';
        let cls = '';

        if (e.kind === 'reasoning') {
            line = `${stamp}  ${branchTag}${e.worker} / ${e.model} / ${e.effort}  ·  reasoning ▾\n${e.text}`;
        } else if (e.kind === 'tool_call') {
            line = `  ▸ tool_call  ${e.data?.name}  ${JSON.stringify(e.data?.args || {})}`;
        } else if (e.kind === 'tool_result') {
            line = `  ◂ tool_result  ${e.data?.name}  ${e.data?.summary || ''}`;
        } else if (e.kind === 'usage') {
            line = `  ▴ usage  in=${e.data?.in_tokens ?? 0} out=${e.data?.out_tokens ?? 0}`;
        } else if (e.kind === 'watchdog') {
            cls = 'stream-watchdog';
            line = `⚠ [watchdog:verbose] ${e.text} ${e.data?.reason || ''}`;
        } else if (e.kind === 'lifecycle') {
            line = `[lifecycle] ${e.data?.event} ${e.data?.detail || ''}`;
        } else {
            line = `${stamp} [${e.kind}] ${e.text}`;
        }

        const node = this._newLine(line, this.view === 'cards' ? `stream-card ${cls}` : cls);
        this._appendNode(node);
        this.lastEvent = e;
    }
}
