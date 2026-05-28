const MIN_BACKOFF_MS = 500;
const MAX_BACKOFF_MS = 10000;

export class SSEClient {
    constructor(path, eventName = 'worker_event') {
        this.path = path;
        this.eventName = eventName;
        this.es = null;
        this.lastEventId = null;
        this.onEvent = null;
        this.onDone = null;
        this.onError = null;
        this.reconnectTimer = null;
        this.closed = false;
        this.attempt = 0;
    }

    subscribe(onEvent, onDone, onError) {
        this.onEvent = onEvent;
        this.onDone = onDone;
        this.onError = onError;
        this.closed = false;
        this.attempt = 0;
        this._connect();
        return () => this.cancel();
    }

    _buildUrl() {
        if (this.lastEventId === null || this.lastEventId === undefined) {
            return this.path;
        }
        const joiner = this.path.includes('?') ? '&' : '?';
        return `${this.path}${joiner}after_id=${encodeURIComponent(String(this.lastEventId))}`;
    }

    _scheduleReconnect() {
        if (this.closed) {
            return;
        }
        const delay = Math.min(MIN_BACKOFF_MS * (2 ** this.attempt), MAX_BACKOFF_MS);
        this.attempt += 1;
        this.reconnectTimer = setTimeout(() => this._connect(), delay);
    }

    _connect() {
        if (this.closed) {
            return;
        }
        const url = this._buildUrl();
        this.es = new EventSource(url);
        this.es.addEventListener(this.eventName, (e) => {
            const raw = e.lastEventId;
            if (raw !== undefined && raw !== null && raw !== '') {
                const parsed = Number(raw);
                if (Number.isFinite(parsed)) {
                    this.lastEventId = parsed;
                }
            }
            this.attempt = 0;
            if (this.onEvent) {
                this.onEvent(JSON.parse(e.data));
            }
        });
        this.es.addEventListener('done', (e) => {
            this.cancel();
            if (this.onDone) {
                this.onDone(JSON.parse(e.data));
            }
        });
        this.es.onerror = (e) => {
            if (this.es) {
                this.es.close();
                this.es = null;
            }
            if (this.onError) {
                this.onError(e);
            }
            this._scheduleReconnect();
        };
    }

    cancel() {
        this.closed = true;
        if (this.es) {
            this.es.close();
            this.es = null;
        }
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }
    }
}

export function subscribe(runId, onEvent, onDone, onError) {
    const client = new SSEClient(`/api/sse/${runId}`);
    return client.subscribe(onEvent, onDone, onError);
}
