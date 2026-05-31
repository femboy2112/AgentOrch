import { ProfessionalRenderer } from './stream.js';
import { FriendlyPlanRenderer } from './plan_renderer.js';
import { getMode, onModeChange } from '../presentation_mode.js';
import { runEventStore } from '../run_event_store.js';
import { RenderScheduler } from '../render_scheduler.js';

export function mountRunRenderers(container, runId) {
    container.innerHTML = '';

    const professionalHost = document.createElement('div');
    professionalHost.className = 'render-host';
    const friendlyHost = document.createElement('div');
    friendlyHost.className = 'render-host';
    friendlyHost.hidden = true;

    container.appendChild(professionalHost);
    container.appendChild(friendlyHost);

    const professional = new ProfessionalRenderer(professionalHost);
    const friendly = new FriendlyPlanRenderer(friendlyHost);
    let professionalCursor = 0;

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

    applyMode(getMode());
    scheduler.queueUpdate(runId);

    return {
        appendEvent(event, eventId) {
            runEventStore.append(runId, event, eventId);
        },
        destroy() {
            offStore();
            offMode();
        },
    };
}
