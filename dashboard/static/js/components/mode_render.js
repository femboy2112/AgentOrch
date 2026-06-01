import { ProfessionalRenderer } from './stream.js';
import { FriendlyPlanRenderer } from './plan_renderer.js';
import { getMode, onModeChange } from '../presentation_mode.js';
import { runEventStore } from '../run_event_store.js';
import { RenderScheduler } from '../render_scheduler.js';
import { getArtifact } from '../api.js';

function extractObjectiveFromPrompt(raw) {
    const text = String(raw || '').replace(/\r\n/g, '\n').trim();
    if (!text) {
        return '';
    }
    const markerMatch = text.match(/^##\s*Instruction\b/im);
    if (markerMatch && Number.isFinite(markerMatch.index)) {
        const tail = text.slice(markerMatch.index + markerMatch[0].length).trim();
        return tail;
    }
    const lines = text.split('\n');
    const preamblePatterns = [
        /^you are a coding worker operating inside/i,
        /^implement the instruction below/i,
        /^keep the change minimal/i,
        /^do not run `?sudo`?/i,
        /^a separate automated harness/i,
        /^when finished/i,
    ];
    while (lines.length > 0) {
        const current = String(lines[0] || '').trim();
        if (!current) {
            lines.shift();
            continue;
        }
        if (!preamblePatterns.some((pattern) => pattern.test(current))) {
            break;
        }
        lines.shift();
    }
    return lines.join('\n').trim();
}

export function mountRunRenderers(container, runId, options = {}) {
    container.innerHTML = '';

    const professionalHost = document.createElement('div');
    professionalHost.className = 'render-host';
    const friendlyHost = document.createElement('div');
    friendlyHost.className = 'render-host';
    friendlyHost.hidden = true;

    container.appendChild(professionalHost);
    container.appendChild(friendlyHost);

    const professional = new ProfessionalRenderer(professionalHost);
    let friendlyMeta = options.runMeta && typeof options.runMeta === 'object' ? { ...options.runMeta } : {};
    const friendly = new FriendlyPlanRenderer(friendlyHost, { runMeta: friendlyMeta });
    let professionalCursor = 0;

    function applyFriendlyMetaPatch(patch) {
        if (!patch || typeof patch !== 'object') {
            return;
        }
        friendlyMeta = { ...friendlyMeta, ...patch };
        friendly.setRunMeta(friendlyMeta);
    }

    function renderProfessional(snapshot) {
        while (professionalCursor < snapshot.events.length) {
            professional.appendEvent(snapshot.events[professionalCursor]);
            professionalCursor += 1;
        }
    }

    function applyMode(mode) {
        const friendlyMode = mode === 'friendly';
        professionalHost.hidden = friendlyMode;
        friendlyHost.hidden = !friendlyMode;
        const snapshot = runEventStore.snapshot(runId);
        renderProfessional(snapshot);
        if (friendlyMode) {
            friendly.render(snapshot);
        }
    }

    const scheduler = new RenderScheduler({
        runId,
        getMode,
        getSnapshot: () => runEventStore.snapshot(runId),
        renderActive: (snapshot, mode) => {
            if (mode === 'friendly') {
                friendly.render(snapshot);
                return;
            }
            renderProfessional(snapshot);
        },
    });

    const offStore = runEventStore.subscribe(runId, () => {
        const snapshot = runEventStore.snapshot(runId);
        renderProfessional(snapshot);
        scheduler.queueUpdate(runId);
    });
    const offMode = onModeChange((mode) => {
        applyMode(mode);
        scheduler.queueUpdate(runId);
    });

    (async () => {
        try {
            const prompt = await getArtifact(runId, 'prompt');
            const objective = extractObjectiveFromPrompt(prompt);
            if (!objective) {
                return;
            }
            applyFriendlyMetaPatch({ objective });
            scheduler.queueUpdate(runId);
        } catch (_err) {
            // Objective is optional in friendly mode.
        }
    })();

    applyMode(getMode());
    scheduler.queueUpdate(runId);

    return {
        appendEvent(event, eventId) {
            runEventStore.append(runId, event, eventId);
        },
        setRunMeta(meta) {
            applyFriendlyMetaPatch(meta && typeof meta === 'object' ? meta : {});
            scheduler.queueUpdate(runId);
        },
        destroy() {
            offStore();
            offMode();
        },
    };
}
