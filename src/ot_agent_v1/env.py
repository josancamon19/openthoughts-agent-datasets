"""
Daytona environment management for OpenThoughts Agent using Harbor framework.

Uses Harbor's AgentFactory to support multiple agents dynamically.

## Supported Agents (via Harbor AgentFactory)

### External Agents (run outside container, interact via tmux):
- terminus-2: Full trajectory support, writes trajectory.json locally

### Installed Agents (run inside container):
- claude-code: Full trajectory support (converts native JSONL → trajectory.json)
- swe-agent: Full trajectory support
- openhands: Full trajectory support
- gemini-cli: Full trajectory support
- aider, cline-cli, codex, cursor-cli, goose, opencode, qwen-coder: Limited/no trajectory

## Model Format
- External agents (terminus-2): LiteLLM format "provider/model" (e.g., "anthropic/claude-opus-4-5-20251101")
- Installed agents: Varies by agent (claude-code uses direct model name)

## Testing Agents
- SSH into container to verify installation: `which claude`
- Check install logs: `cat /installed-agent/install.sh`
- Check agent output: `ls -la /logs/agent/`
"""

import asyncio
import concurrent.futures
import io
import json
import os
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Callable
from uuid import uuid4

from dotenv import load_dotenv
from harbor.agents.factory import AgentFactory, AgentName
from harbor.agents.installed.base import BaseInstalledAgent
from harbor.environments.daytona import DaytonaEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.task.task import Task
from harbor.models.trial.paths import EnvironmentPaths, TrialPaths

load_dotenv()

DAYTONA_API_URL = "https://app.daytona.io/api"

# Default model for all agents (Opus 4.5)
DEFAULT_MODEL = "claude-opus-4-5-20251101"
DEFAULT_MODEL_LITELLM = "anthropic/claude-opus-4-5-20251101"

# Enable extended thinking for Claude Code (thinking traces)
# Set to desired budget, e.g., 10000 tokens for thinking
DEFAULT_MAX_THINKING_TOKENS = "10000"

# Default Gemini model
DEFAULT_GEMINI_MODEL = "gemini/gemini-3-flash-preview"

# Default OpenAI model for Codex
DEFAULT_OPENAI_MODEL = "gpt-5.1-codex-max"

# Default OpenHands model (LiteLLM format)
DEFAULT_OPENHANDS_MODEL = "anthropic/claude-opus-4-5-20251101"

# Default Cline model (format: provider:model-id)
DEFAULT_CLINE_MODEL = "anthropic:claude-opus-4-5-20251101"

# Agent configuration - display names and default models
# Using Harbor's AgentName values as keys
AGENT_CONFIG = {
    "claude-code": {
        "display_name": "Claude Code",
        "api_key_env": "ANTHROPIC_API_KEY",
        "default_model": DEFAULT_MODEL,  # Direct Anthropic format
    },
    "terminus-2": {
        "display_name": "Terminus2",
        "api_key_env": "ANTHROPIC_API_KEY",
        "default_model": DEFAULT_MODEL_LITELLM,  # LiteLLM format
    },
    "cursor-cli": {
        "display_name": "Cursor CLI",
        "api_key_env": "CURSOR_API_KEY",
        "default_model": DEFAULT_MODEL_LITELLM,  # LiteLLM format: provider/model
    },
    "gemini-cli": {
        "display_name": "Gemini CLI",
        "api_key_env": "GEMINI_API_KEY",
        "default_model": DEFAULT_GEMINI_MODEL,  # Gemini format
    },
    "codex": {
        "display_name": "Codex (OpenAI)",
        "api_key_env": "OPENAI_API_KEY",
        "default_model": DEFAULT_OPENAI_MODEL,  # OpenAI model name
    },
    "openhands": {
        "display_name": "OpenHands",
        "api_key_env": "ANTHROPIC_API_KEY",  # Uses LLM_API_KEY internally, derived from model
        "default_model": DEFAULT_OPENHANDS_MODEL,  # LiteLLM format
    },
    "cline-cli": {
        "display_name": "Cline CLI",
        "api_key_env": "ANTHROPIC_API_KEY",  # We'll copy this to API_KEY for Cline
        "default_model": DEFAULT_CLINE_MODEL,  # Format: provider:model-id
    },
}

# List of supported agent names (for UI dropdowns, etc.)
SUPPORTED_AGENTS = list(AGENT_CONFIG.keys())


def extract_task_to_tempdir(task_binary: bytes) -> Path | None:
    """Extract task tarball to a temp directory and return the path."""
    try:
        tmpdir = tempfile.mkdtemp(prefix="rl_task_")
        tar_io = io.BytesIO(task_binary)
        with tarfile.open(fileobj=tar_io, mode="r:gz") as tar:
            tar.extractall(tmpdir, filter="data")
        return Path(tmpdir)
    except Exception:
        return None


async def create_harbor_daytona_env(api_key: str, task_dir: Path) -> dict:
    """
    Create a Daytona sandbox using Harbor's task infrastructure.
    This properly handles all task files including seeds, tests, and solution.
    """
    # Set Daytona credentials in environment
    os.environ["DAYTONA_API_KEY"] = api_key
    os.environ["DAYTONA_API_URL"] = DAYTONA_API_URL

    # Load task using Harbor's Task class
    task = Task(task_dir)

    # Create temporary trial paths
    temp_trial_dir = tempfile.mkdtemp(prefix="harbor_trial_")
    trial_paths = TrialPaths(trial_dir=Path(temp_trial_dir))

    # Create DaytonaEnvironment
    env = DaytonaEnvironment(
        environment_dir=task.paths.environment_dir,
        environment_name=task.name,
        session_id=str(uuid4()),
        trial_paths=trial_paths,
        task_env_config=task.config.environment,
    )

    # Start the environment (builds container with all files)
    await env.start(force_build=True)

    # Upload solution and tests
    if task.paths.solution_dir.exists():
        await env.upload_dir(
            task.paths.solution_dir, str(EnvironmentPaths.solution_dir)
        )
    if task.paths.tests_dir.exists():
        await env.upload_dir(task.paths.tests_dir, str(EnvironmentPaths.tests_dir))

    # Get SSH access
    ssh_access = await env._sandbox.create_ssh_access()

    return {
        "sandbox_id": env._sandbox.id,
        "ssh_command": f"ssh {ssh_access.token}@ssh.app.daytona.io",
        "task_name": task.name,
        "environment": env,
    }


def get_agent_config(agent_name: str) -> dict:
    """Get configuration for an agent by name."""
    if agent_name not in AGENT_CONFIG:
        raise ValueError(
            f"Unknown agent: {agent_name}. Supported agents: {SUPPORTED_AGENTS}"
        )
    return AGENT_CONFIG[agent_name]


def is_installed_agent(agent) -> bool:
    """Check if an agent is an installed agent (runs inside container)."""
    return isinstance(agent, BaseInstalledAgent)


async def spin_up_environment(
    task_dir: Path,
    daytona_api_key: str,
    agent_name: str,
    model_name: str | None = None,
    status_fn: Callable[[str], None] = print,
):
    # Get agent configuration
    agent_config = get_agent_config(agent_name)

    # Use agent-specific default model if not provided
    if model_name is None:
        model_name = agent_config["default_model"]

    # Set Daytona credentials
    os.environ["DAYTONA_API_KEY"] = daytona_api_key
    os.environ["DAYTONA_API_URL"] = DAYTONA_API_URL

    # Enable extended thinking for Claude Code (thinking traces)
    if "MAX_THINKING_TOKENS" not in os.environ:
        os.environ["MAX_THINKING_TOKENS"] = DEFAULT_MAX_THINKING_TOKENS

    # Verify API key based on agent requirements
    api_key_env = agent_config["api_key_env"]
    if not os.environ.get(api_key_env):
        raise ValueError(f"{api_key_env} not set in environment")

    # Cline CLI expects API_KEY env var, copy from ANTHROPIC_API_KEY
    if agent_name == "cline-cli" and os.environ.get("ANTHROPIC_API_KEY"):
        os.environ["API_KEY"] = os.environ["ANTHROPIC_API_KEY"]

    # Load task
    task = Task(task_dir)

    # Create logs directory for agent

    # Create trial paths
    trial_dir = Path(tempfile.mkdtemp(prefix="harbor_trial_"))
    trial_paths = TrialPaths(trial_dir=trial_dir)

    # Create DaytonaEnvironment
    session_id = str(uuid4())
    env = DaytonaEnvironment(
        environment_dir=task.paths.environment_dir,
        environment_name=task.name,
        session_id=session_id,
        trial_paths=trial_paths,
        task_env_config=task.config.environment,
    )

    status_fn(f"Starting environment for task: {task.name}")
    await env.start(force_build=True)

    # Get SSH access early so user can connect while agent runs
    ssh_access = await env._sandbox.create_ssh_access()
    ssh_command = f"ssh {ssh_access.token}@ssh.app.daytona.io"
    status_fn(f"SSH ready: {ssh_command}")
    if task.paths.solution_dir.exists():
        status_fn("Uploading task files...")
        await env.upload_dir(
            task.paths.solution_dir, str(EnvironmentPaths.solution_dir)
        )

    if task.paths.tests_dir.exists():
        await env.upload_dir(task.paths.tests_dir, str(EnvironmentPaths.tests_dir))

    return env, ssh_command


async def install_and_get_agent(
    agent_name: str,
    logs_dir: Path,
    model_name: str,
    env: DaytonaEnvironment,
    status_fn: Callable[[str], None],
):
    agent_config = get_agent_config(agent_name)
    agent_display_name = agent_config["display_name"]

    agent = AgentFactory.create_agent_from_name(
        name=AgentName(agent_name),
        logs_dir=logs_dir,
        model_name=model_name,
    )

    # Create agent context
    context = AgentContext()
    is_installed = is_installed_agent(agent)
    if is_installed:
        status_fn(f"Installing {agent_display_name} agent...")
    else:
        status_fn(f"Setting up {agent_display_name} agent...")
    await agent.setup(env)

    return agent, context


async def run_installed_agent(
    agent, env, context, status_fn: Callable[[str], None], instruction: str
):
    status_fn(f"Running {agent.name}...")
    status_fn(f"{agent.name} is working on the task...")
    await agent.run(instruction, env, context)

    status_fn("Collecting agent trajectory...")
    return str(EnvironmentPaths.agent_dir)  # /logs/agent


async def convert_logs_to_trajectory(
    agent,
    env,
    context,
    logs_dir: Path,
    container_agent_dir: str,
):
    ls_result = await env.exec(
        command=f"ls -la {container_agent_dir} 2>/dev/null || echo 'Directory not found'"
    )
    if ls_result.stdout:
        print(f"[DEBUG] Container agent dir contents:\n{ls_result.stdout}")

    try:
        await env.download_dir(
            source_dir=container_agent_dir,
            target_dir=str(logs_dir),
        )
        print(f"[DEBUG] Downloaded logs to {logs_dir}")
    except Exception as e:
        print(f"[DEBUG] Download dir failed: {e}")

    # Let the agent convert its native logs to trajectory.json
    try:
        print(f"[DEBUG] agent.logs_dir = {agent.logs_dir}")
        print(f"[DEBUG] our logs_dir = {logs_dir}")
        print(
            f"[DEBUG] Before populate_context_post_run, logs_dir contents: {list(logs_dir.iterdir()) if logs_dir.exists() else 'dir not found'}"
        )
        agent.populate_context_post_run(context)
        print("[DEBUG] populate_context_post_run completed")
        print(
            f"[DEBUG] After populate_context_post_run, logs_dir contents: {list(logs_dir.iterdir()) if logs_dir.exists() else 'dir not found'}"
        )
    except Exception as e:
        print(f"[DEBUG] populate_context_post_run failed: {e}")
        import traceback

        traceback.print_exc()

    # Fallback: If trajectory.json wasn't created, try to convert native log files
    trajectory_path = logs_dir / "trajectory.json"
    if not trajectory_path.exists():
        # Try claude-code.txt first
        claude_code_txt = logs_dir / "claude-code.txt"
        if claude_code_txt.exists():
            print(
                "[DEBUG] trajectory.json not found, using claude-code.txt as fallback"
            )
            try:
                lines = claude_code_txt.read_text().strip().split("\n")
                events = [json.loads(line) for line in lines if line.strip()]
                with open(trajectory_path, "w") as f:
                    json.dump(
                        {"events": events, "source": "claude-code.txt"}, f, indent=2
                    )
                print(
                    f"[DEBUG] Created trajectory.json from claude-code.txt ({len(events)} events)"
                )
            except Exception as e:
                print(f"[DEBUG] Failed to convert claude-code.txt: {e}")

        # Try codex.txt if still no trajectory
        if not trajectory_path.exists():
            codex_txt = logs_dir / "codex.txt"
            if codex_txt.exists():
                print("[DEBUG] trajectory.json not found, using codex.txt as fallback")
                try:
                    lines = codex_txt.read_text().strip().split("\n")
                    events = [json.loads(line) for line in lines if line.strip()]
                    with open(trajectory_path, "w") as f:
                        json.dump(
                            {"events": events, "source": "codex.txt"}, f, indent=2
                        )
                    print(
                        f"[DEBUG] Created trajectory.json from codex.txt ({len(events)} events)"
                    )
                except Exception as e:
                    print(f"[DEBUG] Failed to convert codex.txt: {e}")

        # Try cursor-cli.txt if still no trajectory
        if not trajectory_path.exists():
            cursor_cli_txt = logs_dir / "cursor-cli.txt"
            if cursor_cli_txt.exists():
                print(
                    "[DEBUG] trajectory.json not found, using cursor-cli.txt as fallback"
                )
                try:
                    content = cursor_cli_txt.read_text()
                    # cursor-cli.txt is raw output, wrap it as a single event
                    with open(trajectory_path, "w") as f:
                        json.dump(
                            {
                                "events": [{"type": "output", "content": content}],
                                "source": "cursor-cli.txt",
                            },
                            f,
                            indent=2,
                        )
                    print(
                        f"[DEBUG] Created trajectory.json from cursor-cli.txt ({len(content)} chars)"
                    )
                except Exception as e:
                    print(f"[DEBUG] Failed to convert cursor-cli.txt: {e}")

        # Try cline.txt if still no trajectory
        if not trajectory_path.exists():
            cline_txt = logs_dir / "cline.txt"
            if cline_txt.exists():
                print(
                    "[DEBUG] trajectory.json not found, using cline.txt as fallback"
                )
                try:
                    content = cline_txt.read_text()
                    # cline.txt is raw output, wrap it as a single event
                    with open(trajectory_path, "w") as f:
                        json.dump(
                            {
                                "events": [{"type": "output", "content": content}],
                                "source": "cline.txt",
                            },
                            f,
                            indent=2,
                        )
                    print(
                        f"[DEBUG] Created trajectory.json from cline.txt ({len(content)} chars)"
                    )
                except Exception as e:
                    print(f"[DEBUG] Failed to convert cline.txt: {e}")

        # Try openhands.trajectory.json if still no trajectory (native OpenHands format)
        if not trajectory_path.exists():
            openhands_traj = logs_dir / "openhands.trajectory.json"
            if openhands_traj.exists():
                print(
                    "[DEBUG] trajectory.json not found, using openhands.trajectory.json as fallback"
                )
                try:
                    with open(openhands_traj) as f:
                        oh_data = json.load(f)
                    # OpenHands native format has a list of events directly
                    with open(trajectory_path, "w") as f:
                        json.dump(
                            {"events": oh_data, "source": "openhands.trajectory.json"},
                            f,
                            indent=2,
                        )
                    print(
                        f"[DEBUG] Created trajectory.json from openhands.trajectory.json ({len(oh_data)} events)"
                    )
                except Exception as e:
                    print(f"[DEBUG] Failed to convert openhands.trajectory.json: {e}")


async def run_verification(env, status_fn: Callable[[str], None], logs_dir: Path):
    status = status_fn

    status("Agent run complete!")

    # Run tests to verify the solution
    status("Running verification tests...")
    test_passed = None
    test_output = None
    try:
        # Create verifier log directory
        await env.exec(command="mkdir -p /logs/verifier")

        # Run the test script
        test_result = await env.exec(
            command="bash /tests/test.sh 2>&1",
            timeout_sec=60,
        )
        test_output = test_result.stdout or ""
        if test_result.stderr:
            test_output += "\n" + test_result.stderr

        # Read the reward file
        reward_result = await env.exec(
            command="cat /logs/verifier/reward.txt 2>/dev/null || echo 0"
        )
        reward_str = (reward_result.stdout or "0").strip()
        test_passed = reward_str == "1"

        if test_passed:
            status("✅ Tests PASSED!")
        else:
            status(f"❌ Tests FAILED (reward={reward_str})")
    except Exception as e:
        status(f"⚠️ Test verification error: {e}")
        test_passed = None

    # Load trajectory if available
    trajectory = None
    trajectory_path = logs_dir / "trajectory.json"
    print(f"[DEBUG] Looking for trajectory at: {trajectory_path}")
    print(
        f"[DEBUG] logs_dir contents: {list(logs_dir.iterdir()) if logs_dir.exists() else 'dir not found'}"
    )
    if trajectory_path.exists():
        print("[DEBUG] Found trajectory.json, loading...")
        with open(trajectory_path) as f:
            trajectory = json.load(f)
        print(
            f"[DEBUG] Loaded trajectory with keys: {trajectory.keys() if trajectory else 'None'}"
        )
    else:
        print(f"[DEBUG] trajectory.json NOT FOUND at {trajectory_path}")

    return trajectory, test_passed, test_output


async def run_agent(
    task_dir: Path,
    instruction: str,
    daytona_api_key: str,
    agent_name: str = "claude-code",
    model_name: str | None = None,
    status_log: list = None,
) -> dict:
    """
    Run an agent on a task in a Daytona environment.

    Args:
        task_dir: Path to the extracted task directory
        instruction: The task instruction to give the agent
        daytona_api_key: Daytona API key
        agent_name: Harbor agent name (default: "claude-code")
            Supported: "claude-code", "terminus-2"
        model_name: Model to use (default: agent-specific default)
        status_log: Optional list to collect status messages

    Returns:
        Dict with sandbox_id, trajectory, and agent context
    """
    agent_config = get_agent_config(agent_name)
    agent_display_name = agent_config["display_name"]

    if model_name is None:
        model_name = agent_config["default_model"]

    # Load task
    task = Task(task_dir)

    def status(msg):
        if status_log is not None:
            status_log.append(msg)
        print(f"[*] {msg}")

    status_fn = status
    env, ssh_command = await spin_up_environment(
        task_dir, daytona_api_key, agent_name, model_name, status_fn
    )
    logs_dir = Path(tempfile.mkdtemp(prefix=f"{agent_name}_logs_"))
    agent, context = await install_and_get_agent(
        agent_name, logs_dir, model_name, env, status_fn
    )

    await run_installed_agent(agent, env, context, status_fn, instruction)

    # Collect agent trajectory
    container_agent_dir = str(EnvironmentPaths.agent_dir)  # /logs/agent
    await convert_logs_to_trajectory(agent, env, context, logs_dir, container_agent_dir)

    # Run verification (status messages are handled inside run_verification)
    trajectory, test_passed, test_output = await run_verification(
        env, status_fn, logs_dir
    )

    return {
        "sandbox_id": env._sandbox.id,
        "ssh_command": ssh_command,
        "task_name": task.name,
        "agent_name": agent_name,
        "agent_display_name": agent_display_name,
        "model_name": model_name,
        "logs_dir": str(logs_dir),
        "trajectory": trajectory,
        # "raw_output": raw_output,
        "test_passed": test_passed,
        "test_output": test_output,
        "context": {
            "cost_usd": context.cost_usd,
            "n_input_tokens": context.n_input_tokens,
            "n_output_tokens": context.n_output_tokens,
            "n_cache_tokens": context.n_cache_tokens,
        },
        "environment": env,
    }


def run_async(coro):
    """Run async function in sync context - safe for Streamlit.

    Always runs in a fresh thread to avoid event loop caching issues
    with the Daytona SDK between Streamlit reruns.
    """
    # Clear cached modules that might hold stale event loop references
    modules_to_clear = [
        k
        for k in sys.modules.keys()
        if k.startswith(("daytona", "harbor.environments"))
    ]
    for mod in modules_to_clear:
        sys.modules.pop(mod, None)

    # Always run in a new thread with a fresh event loop
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(asyncio.run, coro)
        return future.result()
