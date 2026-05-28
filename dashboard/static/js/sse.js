export class SSEClient {
    constructor(url) {
        this.url = url;
        this.es = null;
        this.lastEventId = null;
        this.onEvent = null;
        this.onDone = null;
        this.onError = null;
        this.reconnectTimer = null;
    }

    subscribe(onEvent, onDone, onError) {
        this.onEvent = onEvent;
        this.onDone = onDone;
        this.onError = onError;
        this._connect();
        return () => this.cancel();
    }

    _connect() {
        let url = this.url;
        if (this.lastEventId !== null) {
            url += (url.includes('?') ? '&' : '?') + `Last-Event-ID=${encodeURIComponent(this.lastEventId)}`;
        }
        this.es = new EventSource(url);
        this.es.addEventListener('worker_event', (e) => {
            this.lastEventId = e.lastEventId;
            if (this.onEvent) this.onEvent(JSON.parse(e.data));
        });
        this.es.addEventListener('done', (e) => {
            this.cancel();
            if (this.onDone) this.onDone(JSON.parse(e.data));
        });
        this.es.onerror = (e) => {
            this.es.close();
            if (this.onError) this.onError(e);
            this.reconnectTimer = setTimeout(() => this._connect(), 2000);
        };
    }

    cancel() {
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
