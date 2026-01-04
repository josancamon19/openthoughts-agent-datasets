"""
Daytona environment management for OpenThoughts Agent using Harbor framework.
"""

import asyncio
import io
import json
import os
import tarfile
import tempfile
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

load_dotenv()

DAYTONA_API_URL = "https://app.daytona.io/api"


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

    from harbor.environments.daytona import DaytonaEnvironment
    from harbor.models.task.task import Task
    from harbor.models.trial.paths import EnvironmentPaths, TrialPaths

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


async def run_claude_code_agent(
    task_dir: Path,
    instruction: str,
    daytona_api_key: str,
    on_status: callable = None,
) -> dict:
    """
    Run Claude Code agent on a task in a Daytona environment.

    Args:
        task_dir: Path to the extracted task directory
        instruction: The task instruction to give the agent
        daytona_api_key: Daytona API key
        on_status: Optional callback for status updates (for Streamlit)

    Returns:
        Dict with sandbox_id, trajectory, and agent context
    """

    def status(msg):
        if on_status:
            on_status(msg)
        print(f"[*] {msg}")

    # Set Daytona credentials
    os.environ["DAYTONA_API_KEY"] = daytona_api_key
    os.environ["DAYTONA_API_URL"] = DAYTONA_API_URL

    # Verify Anthropic key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise ValueError("ANTHROPIC_API_KEY not set in environment")

    # Import Harbor components
    from harbor.agents.installed.claude_code import ClaudeCode
    from harbor.environments.daytona import DaytonaEnvironment
    from harbor.models.agent.context import AgentContext
    from harbor.models.task.task import Task
    from harbor.models.trial.paths import EnvironmentPaths, TrialPaths

    # Load task
    task = Task(task_dir)

    # Create logs directory for agent
    logs_dir = Path(tempfile.mkdtemp(prefix="claude_agent_logs_"))

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

    status(f"Starting environment for task: {task.name}")
    await env.start(force_build=True)

    # Upload solution and tests
    if task.paths.solution_dir.exists():
        status("Uploading solution...")
        await env.upload_dir(
            task.paths.solution_dir, str(EnvironmentPaths.solution_dir)
        )

    if task.paths.tests_dir.exists():
        status("Uploading tests...")
        await env.upload_dir(task.paths.tests_dir, str(EnvironmentPaths.tests_dir))

    # Create agent instance
    agent = ClaudeCode(logs_dir=logs_dir)

    # Setup agent (installs claude-code in container)
    status("Installing Claude Code agent in container (this may take 1-2 min)...")
    await agent.setup(env)

    # Create agent context
    context = AgentContext()

    # Run the agent
    status("Running agent...")
    await agent.run(instruction, env, context)

    # Get SSH access
    ssh_access = await env._sandbox.create_ssh_access()

    # Load trajectory if available
    trajectory = None
    trajectory_path = logs_dir / "trajectory.json"
    if trajectory_path.exists():
        with open(trajectory_path) as f:
            trajectory = json.load(f)

    return {
        "sandbox_id": env._sandbox.id,
        "ssh_command": f"ssh {ssh_access.token}@ssh.app.daytona.io",
        "task_name": task.name,
        "logs_dir": str(logs_dir),
        "trajectory": trajectory,
        "context": {
            "cost_usd": context.cost_usd,
            "n_input_tokens": context.n_input_tokens,
            "n_output_tokens": context.n_output_tokens,
            "n_cache_tokens": context.n_cache_tokens,
        },
        "environment": env,
    }


def run_async(coro):
    """Run async function in sync context - safe for Streamlit."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None:
        # Already in an async context - create new thread
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result()
    else:
        # Not in async context - use new event loop
        return asyncio.run(coro)
