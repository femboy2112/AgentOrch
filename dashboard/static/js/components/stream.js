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
        // Pending event queue + a single requestAnimationFrame flush. Fast
        // reasoning streams (50+ events/sec from codex) would otherwise
        // dominate the main thread with per-event textContent += and
        // scrollTop = scrollHeight, starving toolbar button clicks.
        this._pending = [];
        this._flushScheduled = false;
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
        // Scroll is set ONCE per RAF flush, not per node. The flush method
        // handles autoScroll after the batch is fully appended; per-event
        // scroll was the main culprit for click starvation.
    }

    _newLine(text, cls = '') {
        const p = document.createElement('p');
        p.className = cls;
        p.textContent = text;
        return p;
    }

    // Public API: queue an event for the next animation-frame flush.
    // This is the only entry point the SSE handler should call.
    appendEvent(e) {
        this._pending.push(e);
        this._scheduleFlush();
    }

    _scheduleFlush() {
        if (this._flushScheduled) {
            return;
        }
        this._flushScheduled = true;
        const flush = () => {
            this._flushScheduled = false;
            const batch = this._pending;
            this._pending = [];
            if (batch.length === 0) {
                return;
            }
            // Use a DocumentFragment to coalesce DOM writes into one
            // synchronous append at the end. Saves a layout pass per node.
            const fragment = document.createDocumentFragment();
            // The last-reasoning-paragraph buffer state is updated in-place
            // as we walk the batch — it may point either at an already-in-
            // DOM node (from a prior flush) or at a node we just created
            // inside this fragment. Either way `.textContent +=` mutates
            // the same element; the browser only re-renders at frame end.
            for (const e of batch) {
                this._renderOne(e, fragment);
            }
            if (fragment.childNodes.length > 0) {
                this.viewport.appendChild(fragment);
            }
            if (this.autoScroll) {
                this.viewport.scrollTop = this.viewport.scrollHeight;
            } else if (batch.length > 0) {
                this.jumpBtn.style.display = 'inline-flex';
            }
        };
        if (typeof requestAnimationFrame === 'function') {
            requestAnimationFrame(flush);
        } else {
            // Headless test envs (jsdom) don't always provide RAF; fall
            // back to a microtask so tests behave deterministically.
            Promise.resolve().then(flush);
        }
    }

    _renderOne(e, fragment) {
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
                fragment.appendChild(this.lastReasoningP);
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
        fragment.appendChild(node);
        this.lastEvent = e;
    }
}
