import hashlib
import json
import logging
import os
import re
from typing import Callable, List, Optional, Tuple

from agy_orchestrator.core.agents.agy_agent import AgyAgent
from agy_orchestrator.execution.verifier import QualityVerifier
from agy_orchestrator.workflows.adversarial import AdversarialReview
from agy_orchestrator.workflows.tree_of_thought import TreeOfThought

logger = logging.getLogger(__name__)

# Matches "--- Step N Summary ---" so we can split project_context into
# per-step blocks for the two-tier compaction below. Tolerant of leading
# whitespace and trailing punctuation/numbers; the master writer uses an
# exact format but a future variant shouldn't blow this up.
_STEP_HEADER_RE = re.compile(r"^---\s*Step\s+(\d+)\s+Summary\s*---\s*$", re.MULTILINE)

# How many recent step summaries to keep VERBATIM in compacted context.
# 2 chosen empirically: the adversarial critic on step N+1 needs immediate
# visibility into the file paths and symbol names from N and N-1; older
# detail is fine to digest. Bump via MasterWorkflow(recent_steps_verbatim=K).
DEFAULT_RECENT_STEPS_VERBATIM = 2


def _truncate_orch_title(value: object) -> str:
    return str(value)[:120]


class MasterWorkflow:
    """
    Combines Tree of Thought and Adversarial Review to manage and execute
    large, complex projects accurately.
    """
    def __init__(
        self,
        model: str,
        effort: str,
        branches: int = 3,
        max_iterations: int = 5,
        verifier: Optional[QualityVerifier] = None,
        agent_class=AgyAgent,
        checkpoint_path: Optional[str] = None,
        compaction_interval: int = 6,
        max_context_chars: int = 12000,
        selector: str = "judge",
        working_directory: str = ".",
        recent_steps_verbatim: int = DEFAULT_RECENT_STEPS_VERBATIM,
        event_callback: Optional[Callable[[dict], None]] = None,
    ):
        self.model = model
        self.effort = effort
        self.branches = branches
        self.max_iterations = max_iterations
        self.verifier = verifier
        self.agent_class = agent_class
        # Two-tier compaction: when the running context is digested, the
        # most recent N step summaries are kept VERBATIM and only older
        # steps go through the compactor. Avoids the "I lost the file
        # paths from the step I just did" failure mode that hurt long
        # master runs under the single-digest scheme. Anchor:
        # arxiv 2509.13313 (ReSum) — periodic external summarization with
        # bounded recent context improved long-horizon search by +8.2%.
        self.recent_steps_verbatim = max(0, int(recent_steps_verbatim))
        # Where the verifier should run the test_cmd. Threaded into every
        # AdversarialReview the master spawns so cross-repo dispatches
        # (caller's `out_dir != PROJECT_ROOT`) verify in the right tree.
        self.working_directory = working_directory
        # ToT selection: "judge" (an evaluator scores each branch — right for the
        # diverse code outputs here) or "vote" (free, but rarely clusters for code).
        self.selector = selector
        self.checkpoint_path = checkpoint_path
        # Session compaction: over a long chained run the resumed workflow session
        # (full transcript re-sent every step) and the growing project_context are the
        # token-cost drivers. Every ``compaction_interval`` steps, OR whenever
        # project_context exceeds ``max_context_chars``, condense the context and RESET
        # the session so the accumulated transcript is shed. 0/negative disables.
        self.compaction_interval = int(compaction_interval)
        self.max_context_chars = int(max_context_chars)
        self.event_callback = event_callback

    def _emit_orchestration(self, **fields) -> None:
        cb = self.event_callback
        if cb is None:
            return
        orchestration = {"workflow": "master"}
        for key, value in fields.items():
            if value is not None:
                orchestration[key] = value
        try:
            cb(
                {
                    "kind": "lifecycle",
                    "data": {
                        "event": "orchestration_transition",
                        "orchestration": orchestration,
                    },
                }
            )
        except Exception:
            pass

    def _should_compact(self, steps_since_compaction: int, project_context: str) -> bool:
        if self.compaction_interval and steps_since_compaction >= self.compaction_interval:
            return True
        if self.max_context_chars and len(project_context) >= self.max_context_chars:
            return True
        return False

    def _split_context_into_steps(
        self, project_context: str
    ) -> Tuple[str, List[str]]:
        """Split project_context into (preamble, [per-step block, ...]).

        The preamble is the "Original Goal:" header. Each per-step block is
        the text from one ``--- Step N Summary ---`` line up to (but not
        including) the next one. Order is preserved; if no step markers are
        found the whole string is returned as the preamble with an empty list
        of blocks. The caller can rejoin blocks with ``''.join(...)`` since
        each block already starts with its own ``\n--- Step N ...`` header.
        """
        matches = list(_STEP_HEADER_RE.finditer(project_context))
        if not matches:
            return project_context, []
        # Preamble = everything before the first step header. The header line
        # itself belongs to its step (preserves the leading `\n---` form).
        first_start = matches[0].start()
        # Backwards-include the preceding "\n" so each block reads as a
        # standalone segment when rejoined.
        preamble_end = first_start
        if preamble_end > 0 and project_context[preamble_end - 1] == "\n":
            preamble_end -= 1
        preamble = project_context[:preamble_end]
        blocks: List[str] = []
        for i, m in enumerate(matches):
            start = m.start()
            # Walk back one char if the marker is preceded by "\n" so each
            # block carries its leading newline. This makes joining trivial.
            if start > 0 and project_context[start - 1] == "\n":
                start -= 1
            # The end of block i is "just before block i+1's leading \n",
            # which is matches[i+1].start() - 1. No further walk-back —
            # that boundary newline already lives at start-1 of block i+1.
            if i + 1 < len(matches):
                end = matches[i + 1].start() - 1
            else:
                end = len(project_context)
            blocks.append(project_context[start:end])
        return preamble, blocks

    async def _compact_context(self, initial_prompt: str, project_context: str) -> str:
        """Condense older steps; keep recent N steps verbatim (two-tier).

        For a long master run, the single-digest scheme of the prior version
        had a subtle failure mode: after compaction, the immediately
        preceding step's concrete details (file paths, symbol names) were
        absorbed into a paragraph-prose digest and the next critic
        iteration couldn't tell which exact file the previous step had
        touched. Two-tier compaction sidesteps this — older steps go through
        the digest (cheap context), the LAST ``recent_steps_verbatim`` step
        summaries are kept whole (rich, precise context for the next step).
        Anchor: arxiv 2509.13313.

        Falls open on edge cases — if there aren't enough step markers to
        split, behaves identically to the prior single-digest scheme.
        """
        header = f"Original Goal: {initial_prompt}\n\n=== Accumulated Implementation (compacted) ===\n"
        preamble, blocks = self._split_context_into_steps(project_context)
        keep = self.recent_steps_verbatim

        # No step markers OR fewer steps than we want to keep verbatim:
        # nothing to split on. Compact the whole thing the old way.
        if not blocks or keep == 0 or len(blocks) <= keep:
            to_digest = project_context
            verbatim_suffix = ""
        else:
            older = blocks[:-keep]
            recent = blocks[-keep:]
            to_digest = preamble + "".join(older)
            verbatim_suffix = "".join(recent)

        # Bound the digest input — even with two-tier, an enormous older-tier
        # tail can blow up the compactor prompt. The 24K cap matches the
        # prior behaviour; the recent verbatim block is excluded from this
        # cap because preserving its full content is the whole point.
        compactor = self.agent_class(
            prompt=(
                "Condense the following project progress log into a TIGHT running digest "
                "(<= 1500 words). Preserve the original goal, every file created/modified, key "
                "design decisions, and any names (classes/functions/IDs/APIs) later steps need. "
                "Drop redundancy and full code. Output only the digest.\n\n"
                f"{to_digest[:24000]}"
            ),
            model=self.model,
            effort="low",
        )
        try:
            digest = await compactor.run_async()
            compacted = header + digest.strip() + "\n"
            if verbatim_suffix:
                compacted += "\n=== Recent steps (verbatim) ===" + verbatim_suffix + "\n"
            return compacted
        except Exception as exc:  # robust fallback: keep goal + most-recent tail
            logger.warning("Context compaction failed (%s); truncating to recent tail.", exc)
            tail = project_context[-self.max_context_chars :] if self.max_context_chars else project_context
            return header + tail

    def _checkpoint_key(self, initial_prompt: str) -> str:
        return hashlib.sha256(initial_prompt.encode("utf-8")).hexdigest()

    def _load_checkpoint(
        self, initial_prompt: str
    ) -> Optional[Tuple[List[str], int, str, Optional[str]]]:
        """Return (tasks, completed, project_context, session_id) to resume, or None."""
        path = self.checkpoint_path
        if not path or not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:
            logger.warning("Could not read checkpoint %s: %s", path, exc)
            return None
        if data.get("key") != self._checkpoint_key(initial_prompt):
            logger.info("Checkpoint %s is for a different project; starting fresh.", path)
            return None
        tasks = data.get("tasks") or []
        completed = int(data.get("completed", 0))
        if not tasks or completed >= len(tasks):
            logger.info("Checkpoint shows project already complete; nothing to resume.")
            return None
        return tasks, completed, data.get("project_context", ""), data.get("session_id")

    def _save_checkpoint(
        self,
        initial_prompt: str,
        tasks: List[str],
        completed: int,
        project_context: str,
        session_id: Optional[str],
    ) -> None:
        if not self.checkpoint_path:
            return
        try:
            tmp = self.checkpoint_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "key": self._checkpoint_key(initial_prompt),
                        "tasks": tasks,
                        "completed": completed,
                        "project_context": project_context,
                        "session_id": session_id,
                    },
                    fh,
                )
            os.replace(tmp, self.checkpoint_path)  # atomic: crash mid-write can't corrupt
        except Exception as exc:
            logger.warning("Could not write checkpoint %s: %s", self.checkpoint_path, exc)

    async def execute(self, initial_prompt: str) -> str:
        resumed = self._load_checkpoint(initial_prompt)
        if resumed is not None:
            tasks, start_index, project_context, workflow_session_id = resumed
            logger.info(
                "Resuming Master Workflow from checkpoint at step %d/%d (continuing in place).",
                start_index + 1,
                len(tasks),
            )
        else:
            logger.info("Starting Master Workflow Planning Phase...")

            # 1. Planner Phase
            planner_prompt = (
                f"You are the Lead Architect. Break down the following complex project into a logical sequence of implementation steps.\n"
                f"Output ONLY a valid JSON list of strings, where each string is a detailed prompt for a single step.\n"
                f"Example: [\"Step 1: Setup project structure and core utilities...\", \"Step 2: Implement UI component X...\"]\n\n"
                f"Project Request:\n{initial_prompt}"
            )

            planner = self.agent_class(prompt=planner_prompt, model=self.model, effort="high")
            plan_output = await planner.run_async()

            # Capture session established by planner for reuse across all subsequent calls
            workflow_session_id = getattr(planner, "session_id", None)
            if workflow_session_id:
                logger.info("Workflow session established: %s", workflow_session_id)

            # Extract JSON list from plan_output
            tasks = []
            try:
                start = plan_output.find('[')
                end = plan_output.rfind(']') + 1
                if start != -1 and end != 0:
                    tasks = json.loads(plan_output[start:end])
                else:
                    raise ValueError("No JSON array found.")
            except Exception as e:
                logger.warning(f"Failed to parse Planner output as JSON: {e}. Defaulting to a single step.")
                tasks = [initial_prompt]

            logger.info(f"Project broken down into {len(tasks)} steps.")
            step_titles = [_truncate_orch_title(task) for task in tasks]
            self._emit_orchestration(
                phase="plan",
                action="completed",
                step_total=len(tasks),
                step_titles=step_titles,
            )

            project_context = f"Original Goal: {initial_prompt}\n\n=== Accumulated Implementation ===\n"
            start_index = 0
            self._save_checkpoint(initial_prompt, tasks, start_index, project_context, workflow_session_id)

        # 2. Execution Loop
        steps_since_compaction = 0
        for i in range(start_index, len(tasks)):
            task = tasks[i]
            logger.info(f"--- Executing Step {i+1}/{len(tasks)} ---")
            logger.info(f"Task description: {task[:100]}...")
            self._emit_orchestration(
                phase="step",
                action="started",
                step_index=i + 1,
                step_total=len(tasks),
                step_title=_truncate_orch_title(task),
                model=self.model,
                effort=self.effort,
            )

            step_prompt = (
                f"You are implementing Step {i+1} of a larger project.\n\n"
                f"Project Context (What has been built so far):\n{project_context}\n\n"
                f"Current Task to implement NOW:\n{task}"
            )

            # Phase A: Tree of Thought (Exploration). Skip entirely when branches<=1
            # — exploring + judging a single branch is pure overhead; go straight to
            # the adversarial refinement of the step prompt (cheap-mode early-exit).
            if self.branches <= 1:
                logger.info("Phase A: skipped (branches<=1) — refining the step directly.")
                best_tot_output = ""
            else:
                logger.info("Phase A: Tree of Thought Exploration")
                # ToT branches run fresh sessions — concurrent --fork-session on the same parent
                # causes race conditions. They're throwaway explorers anyway.
                tot_branches = [
                    self.agent_class(prompt=step_prompt, model=self.model, effort=self.effort)
                    for _ in range(self.branches)
                ]
                # The judge clones this evaluator per branch and scores them in
                # parallel with independent (sessionless) instances, so do NOT bind
                # the workflow session here — resuming it would both contaminate
                # per-branch scoring and pollute the main thread.
                tot_evaluator = self.agent_class(prompt="", model=self.model, effort="high")
                tot = TreeOfThought(
                    tot_branches,
                    tot_evaluator,
                    selector=self.selector,
                    event_callback=self.event_callback,
                )
                best_tot_output = await tot.execute()

            # Phase B: Adversarial Review (Refinement) — resume main workflow session
            logger.info("Phase B: Adversarial Review Refinement")
            gen_kwargs = dict(model=self.model, effort=self.effort)
            critic_kwargs = dict(model=self.model, effort="high")
            if workflow_session_id:
                try:
                    gen_kwargs["session_id"] = workflow_session_id
                    critic_kwargs["session_id"] = workflow_session_id
                except Exception:
                    pass
            adv_generator = self.agent_class(prompt=step_prompt, **gen_kwargs)

            if best_tot_output:
                adv_prompt = (
                    f"{step_prompt}\n\n"
                    f"Please refine, finalize, and perfect the following draft implementation. "
                    f"Ensure it meets the highest standards and resolves any bugs:\n"
                    f"{best_tot_output}"
                )
            else:
                # No exploration draft (branches<=1): implement the step directly.
                adv_prompt = step_prompt

            adv_critic = self.agent_class(prompt="", **critic_kwargs)

            adv = AdversarialReview(
                generator_instance=adv_generator,
                critic_instance=adv_critic,
                verifier=self.verifier,
                max_iterations=self.max_iterations,
                working_directory=self.working_directory,
                event_callback=self.event_callback,
            )

            final_step_output = await adv.execute(adv_prompt)

            logger.info(f"Step {i+1} Completed. Summarizing for project context.")
            self._emit_orchestration(
                phase="step",
                action="completed",
                step_index=i + 1,
                step_total=len(tasks),
                step_title=_truncate_orch_title(task),
            )
            # Summarize the step output to keep project_context compact.
            # Passing full HTML/code outputs into every subsequent prompt balloons to 50KB+.
            summarize_kwargs = dict(model=self.model, effort="low")
            if workflow_session_id:
                try:
                    summarize_kwargs["session_id"] = workflow_session_id
                except Exception:
                    pass
            summarizer = self.agent_class(
                prompt=(
                    f"In 3-5 bullet points, summarize what was just implemented in Step {i+1}.\n"
                    f"Focus on: what files were created/modified, key design decisions, and any\n"
                    f"CSS classes, JS functions, or IDs that other steps should know about.\n"
                    f"Be concise. Do NOT reproduce the full code.\n\nStep output:\n{final_step_output[:8000]}"
                ),
                **summarize_kwargs
            )
            try:
                step_summary = await summarizer.run_async()
                # Update workflow session from summarizer if we don't have one yet
                if not workflow_session_id:
                    workflow_session_id = getattr(summarizer, "session_id", None)
            except Exception as e:
                logger.warning(f"Summarizer failed ({e}), falling back to task description.")
                step_summary = f"Completed: {task[:300]}"
            project_context += f"\n--- Step {i+1} Summary ---\n{step_summary}\n"
            # Persist progress so a crash resumes from the NEXT step, in place.
            self._save_checkpoint(initial_prompt, tasks, i + 1, project_context, workflow_session_id)

            # Session compaction: shed the accumulated transcript + condense context so a
            # multi-hour chained run does not carry an ever-growing session.
            steps_since_compaction += 1
            if self._should_compact(steps_since_compaction, project_context):
                logger.info(
                    "Compacting context + resetting session after step %d/%d (context was %d chars).",
                    i + 1, len(tasks), len(project_context),
                )
                project_context = await self._compact_context(initial_prompt, project_context)
                workflow_session_id = None  # next step starts a fresh session, shedding the transcript
                steps_since_compaction = 0
                self._save_checkpoint(initial_prompt, tasks, i + 1, project_context, workflow_session_id)

        logger.info("Master Workflow Complete!")
        return project_context
