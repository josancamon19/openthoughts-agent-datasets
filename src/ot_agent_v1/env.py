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
    status_log: list = None,
) -> dict:
    """
    Run Claude Code agent on a task in a Daytona environment.

    Args:
        task_dir: Path to the extracted task directory
        instruction: The task instruction to give the agent
        daytona_api_key: Daytona API key
        status_log: Optional list to collect status messages

    Returns:
        Dict with sandbox_id, trajectory, and agent context
    """

    def status(msg):
        if status_log is not None:
            status_log.append(msg)
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

    # Get SSH access early so user can connect while agent runs
    ssh_access = await env._sandbox.create_ssh_access()
    ssh_command = f"ssh {ssh_access.token}@ssh.app.daytona.io"
    status(f"🔗 SSH ready: {ssh_command}")

    # Upload solution and tests
    if task.paths.solution_dir.exists():
        status("Uploading task files...")
        await env.upload_dir(
            task.paths.solution_dir, str(EnvironmentPaths.solution_dir)
        )

    if task.paths.tests_dir.exists():
        await env.upload_dir(task.paths.tests_dir, str(EnvironmentPaths.tests_dir))

    # Create agent instance
    agent = ClaudeCode(logs_dir=logs_dir)

    # Setup agent (installs claude-code in container)
    status("Installing Claude Code agent...")
    await agent.setup(env)

    # Create agent context
    context = AgentContext()

    # Run the agent (note: this calls populate_context_post_run but won't find logs yet)
    status("Running agent...")

    # We manually run the commands instead of using agent.run() so we can download logs first
    from harbor.utils.templating import render_prompt_template

    rendered_instruction = instruction
    if agent._prompt_template_path:
        rendered_instruction = render_prompt_template(
            agent._prompt_template_path, instruction
        )

    for i, exec_input in enumerate(
        agent.create_run_agent_commands(rendered_instruction)
    ):
        command_dir = logs_dir / f"command-{i}"
        command_dir.mkdir(parents=True, exist_ok=True)
        (command_dir / "command.txt").write_text(exec_input.command)

        # Only show meaningful progress, not raw commands
        if i == 0:
            status("Setting up agent environment...")
        else:
            status("Agent is working on the task...")

        result = await env.exec(
            command=exec_input.command,
            cwd=exec_input.cwd,
            env=exec_input.env,
            timeout_sec=exec_input.timeout_sec,
        )

        (command_dir / "return-code.txt").write_text(str(result.return_code))
        if result.stdout:
            (command_dir / "stdout.txt").write_text(result.stdout)
        if result.stderr:
            (command_dir / "stderr.txt").write_text(result.stderr)

        # Only report errors
        if result.return_code != 0:
            status(f"⚠️ Command {i} failed with exit code {result.return_code}")

    # Try to download agent logs from container to local logs_dir
    status("Collecting agent trajectory...")
    container_agent_dir = str(EnvironmentPaths.agent_dir)  # /logs/agent

    download_success = False
    try:
        await env.download_dir(
            source_dir=container_agent_dir,
            target_dir=str(logs_dir),
        )
        # Check if we got the session files
        sessions_dir = logs_dir / "sessions"
        if sessions_dir.exists():
            import subprocess

            find_result = subprocess.run(
                ["find", str(sessions_dir), "-type", "f", "-name", "*.jsonl"],
                capture_output=True,
                text=True,
            )
            if find_result.stdout.strip():
                download_success = True
    except Exception:
        pass  # Silently fall back to stdout parsing

    # Fallback: Create sessions directory from stdout if download failed
    if not download_success:
        stdout_file = logs_dir / "command-1" / "stdout.txt"
        if stdout_file.exists():
            # Create the expected sessions directory structure
            sessions_dir = logs_dir / "sessions" / "projects" / "-app"
            sessions_dir.mkdir(parents=True, exist_ok=True)

            # Copy stdout as a JSONL session file
            session_jsonl = sessions_dir / "session.jsonl"
            import shutil

            shutil.copy(stdout_file, session_jsonl)

    # Parse trajectory from logs
    agent.populate_context_post_run(context)
    status("Agent run complete!")

    # Load trajectory if available
    trajectory = None
    trajectory_path = logs_dir / "trajectory.json"
    if trajectory_path.exists():
        with open(trajectory_path) as f:
            trajectory = json.load(f)

    # Read raw agent output for debugging
    raw_output = None
    agent_output_file = logs_dir / "command-1" / "stdout.txt"
    if agent_output_file.exists():
        raw_output = agent_output_file.read_text()

    return {
        "sandbox_id": env._sandbox.id,
        "ssh_command": ssh_command,
        "task_name": task.name,
        "logs_dir": str(logs_dir),
        "trajectory": trajectory,
        "raw_output": raw_output,
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
    import concurrent.futures
    import sys
    
    # Clear cached modules that might hold stale event loop references
    modules_to_clear = [k for k in sys.modules.keys() 
                        if k.startswith(('daytona', 'harbor.environments'))]
    for mod in modules_to_clear:
        sys.modules.pop(mod, None)
    
    # Always run in a new thread with a fresh event loop
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(asyncio.run, coro)
        return future.result()
