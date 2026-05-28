export class StreamRenderer {
    constructor(container) {
        this.container = container;
        this.container.classList.add('stream-renderer');
        this.container.style.fontFamily = 'var(--font-mono)';
        this.container.style.fontSize = '14px';
        this.container.style.lineHeight = '1.55';
        this.container.style.maxWidth = '80ch';
        this.container.style.padding = 'var(--s-4)';
        this.container.style.background = 'var(--bg-0)';
        this.container.style.borderRadius = 'var(--r-1)';
        this.container.style.overflowY = 'auto';
        this.container.style.height = '400px';
        this.container.style.border = '1px solid var(--line)';

        this.lastEvent = null;
        this.lastReasoningP = null;
    }

    appendEvent(e) {
        const d = new Date(e.ts * 1000).toLocaleTimeString();
        
        if (e.kind === 'reasoning') {
            if (this.lastEvent?.kind === 'reasoning' && this.lastEvent?.worker === e.worker && (e.ts - this.lastEvent.ts) <= 1.5) {
                this.lastReasoningP.textContent += e.text;
            } else {
                this.lastReasoningP = document.createElement('p');
                this.lastReasoningP.textContent = `${d}  ${e.worker} / ${e.model} / ${e.effort}  ·  reasoning ▾\n` + e.text;
                this.container.appendChild(this.lastReasoningP);
            }
        } else if (e.kind === 'tool_call') {
            const p = document.createElement('p');
            p.textContent = `  ▸ tool_call  ${e.data?.name}  ${JSON.stringify(e.data?.args || {})}`;
            this.container.appendChild(p);
        } else if (e.kind === 'tool_result') {
            const p = document.createElement('p');
            p.textContent = `  ◂ tool_result  ${e.data?.name}  ${e.data?.summary || ''}`;
            this.container.appendChild(p);
        } else if (e.kind === 'usage') {
            const p = document.createElement('p');
            p.textContent = `  ▴ usage  in=${e.data?.in_tokens} out=${e.data?.out_tokens}`;
            this.container.appendChild(p);
        } else if (e.kind === 'lifecycle') {
            const p = document.createElement('p');
            p.textContent = `[lifecycle] ${e.data?.event} ${e.data?.detail || ''}`;
            this.container.appendChild(p);
        } else if (e.kind === 'message' || e.kind === 'stderr' || e.kind === 'watchdog') {
            const p = document.createElement('p');
            if (e.kind === 'watchdog') {
                p.style.color = 'var(--bad)';
                p.textContent = `⚠ [watchdog:verbose] ${e.text} ${e.data?.reason || ''}`;
            } else {
                p.textContent = `${d} [${e.kind}] ${e.text}`;
            }
            this.container.appendChild(p);
        }

        this.lastEvent = e;
        this.container.scrollTop = this.container.scrollHeight;
    }
}
