const FRIENDLY_INTERVAL_MS = 333;

function nowMs() {
    if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
        return performance.now();
    }
    return Date.now();
}

export class RenderScheduler {
    constructor({ runId, getMode, getSnapshot, renderActive, friendlyIntervalMs = FRIENDLY_INTERVAL_MS }) {
        this.runId = runId;
        this.getMode = getMode;
        this.getSnapshot = getSnapshot;
        this.renderActive = renderActive;
        this.friendlyIntervalMs = friendlyIntervalMs;
        this._dirty = false;
        this._scheduled = false;
        this._lastFriendlyPaintMs = 0;
    }

    queueUpdate(runId) {
        if (runId !== this.runId) {
            return;
        }
        this._dirty = true;
        this._schedule();
    }

    _schedule() {
        if (this._scheduled) {
            return;
        }
        this._scheduled = true;
        const flush = () => {
            this._scheduled = false;
            if (!this._dirty) {
                return;
            }
            const mode = this.getMode();
            const now = nowMs();
            if (mode === 'friendly' && (now - this._lastFriendlyPaintMs) < this.friendlyIntervalMs) {
                this._schedule();
                return;
            }
            this._dirty = false;
            const snapshot = this.getSnapshot(this.runId);
            this.renderActive(snapshot, mode);
            if (mode === 'friendly') {
                this._lastFriendlyPaintMs = now;
            }
        };
        if (typeof requestAnimationFrame === 'function') {
            requestAnimationFrame(flush);
        } else {
            Promise.resolve().then(flush);
        }
    }
}
