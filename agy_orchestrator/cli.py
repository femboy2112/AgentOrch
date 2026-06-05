import argparse
import asyncio
import logging

from agy_orchestrator.core.agents.agy_agent import AgyAgent
from agy_orchestrator.core.agents.claude_agent import ClaudeAgent
from agy_orchestrator.core.agents.codex_agent import CodexAgent
from agy_orchestrator.core.agents.grok_agent import GrokAgent
from agy_orchestrator.core.profile import UserProfile
from agy_orchestrator.version import version_string
from agy_orchestrator.execution.pipeline import LinearPipeline
from agy_orchestrator.execution.verifier import QualityVerifier
from agy_orchestrator.workflows.adversarial import AdversarialReview
from agy_orchestrator.workflows.master import MasterWorkflow
from agy_orchestrator.workflows.tree_of_thought import TreeOfThought

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

def get_agent_class(agent_name: str):
    if agent_name == "claude":
        return ClaudeAgent
    elif agent_name == "codex":
        return CodexAgent
    elif agent_name == "grok":
        return GrokAgent
    return AgyAgent

def create_agent(agent_class, prompt, model, effort=None):
    if agent_class == AgyAgent:
        return agent_class(prompt=prompt, model=model, effort=effort)
    return agent_class(prompt=prompt, model=model)

async def run_adversarial(args, profile):
    agent_class = get_agent_class(args.agent)
    effort = profile.get_baseline_effort(args.agent)

    print(f"\n[AdversarialReview] Using {args.agent}:{args.model}:{effort}\n")

    generator = create_agent(agent_class, prompt="", model=args.model, effort=effort)
    critic = create_agent(agent_class, prompt="", model=args.model, effort="high")

    verifier = None
    if args.test_cmd:
        verifier = QualityVerifier(test_commands=[args.test_cmd])

    workflow = AdversarialReview(generator, critic, verifier, max_iterations=args.max_iterations)
    result = await workflow.execute(args.prompt)
    print("\n--- Final Verified Output ---\n")
    print(result)

async def run_tot(args, profile):
    agent_class = get_agent_class(args.agent)
    effort = profile.get_baseline_effort(args.agent)

    print(f"\n[TreeOfThought] Using {args.agent}:{args.model}:{effort}\n")

    branches = [
        create_agent(agent_class, prompt=args.prompt, model=args.model, effort=effort)
        for _ in range(args.branches)
    ]
    evaluator = create_agent(agent_class, prompt="", model=args.model, effort="high")

    workflow = TreeOfThought(branches, evaluator)
    result = await workflow.execute()
    print("\n--- Best Selected Output ---\n")
    print(result)

async def run_master(args, profile):
    agent_class = get_agent_class(args.agent)
    effort = profile.get_baseline_effort(args.agent)

    print(f"\n[MasterWorkflow] Using {args.agent}:{args.model}:{effort}\n")

    if getattr(args, "fallback", False):
        # On a provider's persistent failure (usage/quota wall), fall back to the
        # next provider. Chain is the chosen --agent first, then agy -> claude ->
        # codex (deduped, order preserved), cycled so a provider whose usage has
        # since recovered gets retried. See core/agents/fallback_agent.py.
        from agy_orchestrator.core.agents.fallback_agent import make_fallback_agent
        ordered = []
        for cls in (agent_class, AgyAgent, ClaudeAgent, CodexAgent):
            if cls not in ordered:
                ordered.append(cls)
        agent_class = make_fallback_agent(ordered)
        print(f"[MasterWorkflow] fallback chain enabled: {'->'.join(c.__name__ for c in ordered)} (cycled)\n")

    verifier = None
    if args.test_cmd:
        verifier = QualityVerifier(test_commands=[args.test_cmd])

    workflow = MasterWorkflow(
        model=args.model,
        effort=effort,
        branches=args.branches,
        max_iterations=args.max_iterations,
        verifier=verifier,
        agent_class=agent_class,
        checkpoint_path=getattr(args, "checkpoint", None),
        compaction_interval=getattr(args, "compaction_interval", 6),
        max_context_chars=getattr(args, "max_context_chars", 12000),
    )
    result = await workflow.execute(args.prompt)
    print("\n--- Final Verified Output ---\n")
    print(result)

async def run_chain(args, profile):
    instances = []
    use_fallback = getattr(args, "fallback", False)
    for agent_name in args.agents:
        primary = get_agent_class(agent_name)
        effort = profile.get_baseline_effort(agent_name)
        if use_fallback:
            # Wrap each stage in usage-exhaustion fallback: primary -> agy ->
            # claude -> codex (deduped, order preserved). Lets a chain build
            # survive one provider running out of usage mid-run.
            from agy_orchestrator.core.agents.fallback_agent import make_fallback_agent
            ordered = []
            for cls in (primary, AgyAgent, ClaudeAgent, CodexAgent):
                if cls not in ordered:
                    ordered.append(cls)
            fb_class = make_fallback_agent(ordered)
            print(f"[LinearPipeline] {agent_name}+fallback({'->'.join(c.__name__ for c in ordered)}):{args.model}:{effort}")
            instances.append(fb_class(prompt="", model=args.model, effort=effort))
        else:
            print(f"[LinearPipeline] {agent_name}:{args.model}:{effort}")
            instances.append(create_agent(primary, prompt="", model=args.model, effort=effort))

    print()
    pipeline = LinearPipeline(instances)
    result = await pipeline.execute(args.prompt)
    print("\n--- Final Chained Output ---\n")
    print(result)


def main():
    parser = argparse.ArgumentParser(description="Agy Orchestrator - Advanced Multi-Agent Wrapper")
    parser.add_argument(
        "--version", action="version", version=version_string("agy-orchestrator"),
        help="Print the running build (version + git commit) and exit",
    )
    parser.add_argument("--claude-plan", type=str, default="free", help="Claude subscription plan (e.g. '20x max')")
    parser.add_argument("--codex-plan", type=str, default="free", help="Codex subscription plan (e.g. '$100')")
    parser.add_argument("--agy-plan", type=str, default="free", help="Agy subscription plan")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Adversarial Subcommand
    adv_parser = subparsers.add_parser("adversarial", help="Run Adversarial Review workflow")
    adv_parser.add_argument("--prompt", type=str, required=True, help="The goal prompt")
    adv_parser.add_argument("--agent", type=str, choices=["agy", "claude", "codex", "grok"], default="agy", help="Agent type to use")
    adv_parser.add_argument("--model", type=str, default="standard", help="Base model to use")
    adv_parser.add_argument("--test-cmd", type=str, help="Optional programmatic test command")
    adv_parser.add_argument("--max-iterations", type=int, default=5, help="Max loops")

    # Tree of Thought Subcommand
    tot_parser = subparsers.add_parser("tot", help="Run Tree-of-Thought workflow")
    tot_parser.add_argument("--prompt", type=str, required=True, help="The goal prompt")
    tot_parser.add_argument("--agent", type=str, choices=["agy", "claude", "codex", "grok"], default="agy", help="Agent type to use")
    tot_parser.add_argument("--model", type=str, default="standard", help="Base model to use")
    tot_parser.add_argument("--branches", type=int, default=3, help="Number of ToT branches")

    # Master Subcommand
    master_parser = subparsers.add_parser("master", help="Run Master Orchestrator workflow")
    master_parser.add_argument("--prompt", type=str, required=True, help="The goal prompt")
    master_parser.add_argument("--agent", type=str, choices=["agy", "claude", "codex", "grok"], default="agy", help="Agent type to use")
    master_parser.add_argument("--model", type=str, default="standard", help="Base model to use")
    master_parser.add_argument("--test-cmd", type=str, help="Optional programmatic test command")
    master_parser.add_argument("--branches", type=int, default=3, help="Number of ToT branches per task")
    master_parser.add_argument("--max-iterations", type=int, default=5, help="Max loops for adversarial refinement")
    master_parser.add_argument("--fallback", action="store_true", help="On provider usage/quota failure, fall back codex->agy->claude->codex")
    master_parser.add_argument("--checkpoint", type=str, default=None, help="Path to a checkpoint file; resume in place from the last completed step if it exists")
    master_parser.add_argument("--compaction-interval", type=int, default=6, help="Compact context + reset the chained session every N steps (0 disables). Controls token growth over long runs.")
    master_parser.add_argument("--max-context-chars", type=int, default=12000, help="Also compact whenever the accumulated project context exceeds this many chars (0 disables).")

    # Chain Subcommand
    chain_parser = subparsers.add_parser("chain", help="Run Linear Pipeline workflow")
    chain_parser.add_argument("--prompt", type=str, required=True, help="The goal prompt")
    chain_parser.add_argument("--agents", type=str, nargs="+", choices=["agy", "claude", "codex", "grok"], default=["codex"], help="List of agents to chain (default: codex). Standing rule: every run leads with codex so a refreshed codex quota is auto-detected next run.")
    chain_parser.add_argument("--model", type=str, default="standard", help="Base model to use")
    # Standing rule: fallback ON by default (codex->agy->claude->codex, cycling) so codex-primary
    # runs survive usage exhaustion; pass --no-fallback for a plain refinement pipeline.
    chain_parser.add_argument("--fallback", action=argparse.BooleanOptionalAction, default=True, help="Wrap each stage in usage-exhaustion fallback: primary->agy->claude->codex (cycling). Default on; --no-fallback to disable.")

    args = parser.parse_args()

    profile = UserProfile(claude_plan=args.claude_plan, codex_plan=args.codex_plan, agy_plan=args.agy_plan)

    if args.command == "adversarial":
        asyncio.run(run_adversarial(args, profile))
    elif args.command == "tot":
        asyncio.run(run_tot(args, profile))
    elif args.command == "master":
        asyncio.run(run_master(args, profile))
    elif args.command == "chain":
        asyncio.run(run_chain(args, profile))

if __name__ == "__main__":
    main()
