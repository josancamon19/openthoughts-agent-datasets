"""
Daytona environment management for OpenThoughts Agent using Harbor framework.
"""

import asyncio
import io
import os
import tarfile
import tempfile
from pathlib import Path
from uuid import uuid4

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
        "environment": env,  # Keep reference for later cleanup if needed
    }


def run_async(coro):
    """Run async function in sync context."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
