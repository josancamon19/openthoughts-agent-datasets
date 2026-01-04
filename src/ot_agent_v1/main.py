"""
OpenThoughts Agent Datasets Dashboard
"""

import io
import os
import tarfile

import streamlit as st
from datasets import load_dataset
from dotenv import load_dotenv

from env import (
    create_harbor_daytona_env,
    extract_task_to_tempdir,
    run_async,
    run_claude_code_agent,
)

load_dotenv()

st.set_page_config(
    page_title="OpenThoughts Agent",
    page_icon="🧠",
    layout="wide",
)

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&display=swap');
    
    .stApp {
        background: #0a0a0f;
        font-family: 'IBM Plex Mono', monospace;
    }
    
    h1, h2, h3 {
        font-family: 'IBM Plex Mono', monospace !important;
        color: #e2e8f0 !important;
    }
    
    .main-title {
        font-size: 2rem;
        font-weight: 600;
        color: #22d3ee;
        margin-bottom: 0.25rem;
    }
    
    .subtitle {
        color: #64748b;
        font-size: 0.9rem;
        margin-bottom: 2rem;
    }
    
    .stat-box {
        background: #111118;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    
    .stat-value {
        font-size: 1.5rem;
        font-weight: 600;
        color: #22d3ee;
    }
    
    .stat-label {
        font-size: 0.75rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .msg-user {
        background: #0c1a2e;
        border-left: 3px solid #3b82f6;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    .msg-assistant {
        background: #1a0c1f;
        border-left: 3px solid #a855f7;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    .msg-system {
        background: #0c1f1a;
        border-left: 3px solid #10b981;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    .msg-role {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.5rem;
        opacity: 0.7;
    }
    
    div[data-testid="stDataFrame"] {
        background: #111118;
        border-radius: 8px;
        padding: 0.5rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ============ SFT Dataset Functions ============


@st.cache_data(show_spinner="Loading SFT dataset from HuggingFace...")
def load_sft_data():
    ds = load_dataset("open-thoughts/OpenThoughts-Agent-v1-SFT", split="train")
    return ds


@st.cache_data(show_spinner="Building SFT index...")
def build_sft_index(_ds):
    """Build task index with metadata."""
    tasks = {}

    for i in range(len(_ds)):
        row = _ds[i]
        task_id = row.get("task", "")
        conv = row.get("conversations", [])

        preview = ""
        assistant_chars = 0
        for msg in conv:
            if msg.get("role") == "assistant":
                assistant_chars += len(msg.get("content", ""))
            if msg.get("role") == "user" and not preview:
                content = msg.get("content", "")
                if "## Goal" in content:
                    goal_idx = content.find("## Goal")
                    goal_text = content[goal_idx + 8 : goal_idx + 150].strip()
                    preview = goal_text.replace("\n", " ").strip()
                elif "## Project Information" in content:
                    proj_idx = content.find("**Project:**")
                    if proj_idx != -1:
                        proj_text = content[proj_idx + 12 : proj_idx + 80].strip()
                        preview = f"Bug fix: {proj_text.split(chr(10))[0]}"
                else:
                    preview = content[:100].replace("\n", " ").strip()

        if task_id.startswith("task_"):
            task_type = "nl2bash"
        elif task_id.startswith("inferredbugs-"):
            task_type = "inferredbugs"
        else:
            task_type = "other"

        tasks[task_id] = {
            "ds_idx": i,
            "task_id": task_id,
            "msgs": len(conv),
            "preview": preview,
            "assistant_tokens": assistant_chars // 4,
            "task_type": task_type,
        }

    def task_sort_key(task_id):
        if task_id.startswith("task_"):
            try:
                return (0, int(task_id.split("_")[1]))
            except (IndexError, ValueError):
                return (0, 0)
        elif task_id.startswith("inferredbugs-"):
            try:
                return (1, int(task_id.split("-")[1]))
            except (IndexError, ValueError):
                return (1, 0)
        return (2, 0)

    sorted_task_ids = sorted(tasks.keys(), key=task_sort_key)
    return tasks, sorted_task_ids


# ============ RL Dataset Functions ============


@st.cache_data(show_spinner="Loading RL dataset from HuggingFace...")
def load_rl_data():
    ds = load_dataset("open-thoughts/OpenThoughts-Agent-v1-RL", split="train")
    return ds


def decode_task_binary(task_binary: bytes) -> dict:
    """Decode the gzipped tar archive and return file contents."""
    files = {}
    try:
        tar_io = io.BytesIO(task_binary)
        with tarfile.open(fileobj=tar_io, mode="r:gz") as tar:
            for member in tar.getmembers():
                if member.isfile():
                    f = tar.extractfile(member)
                    if f:
                        try:
                            content = f.read().decode("utf-8", errors="replace")
                        except Exception:
                            content = "[binary content]"
                        files[member.name] = {
                            "content": content,
                            "size": member.size,
                        }
    except Exception as e:
        files["error"] = {"content": str(e), "size": 0}
    return files


@st.cache_data(show_spinner="Building RL index...")
def build_rl_index(_ds):
    """Build RL task index."""
    tasks = []
    for i in range(len(_ds)):
        row = _ds[i]
        path = row.get("path", "")
        task_binary = row.get("task_binary", b"")

        # Extract task number for sorting
        try:
            task_num = int(path.split("_")[1])
        except (IndexError, ValueError):
            task_num = 0

        tasks.append(
            {
                "ds_idx": i,
                "path": path,
                "task_num": task_num,
                "binary_size": len(task_binary),
            }
        )

    # Sort by task number
    tasks.sort(key=lambda x: x["task_num"])
    return tasks


# ============ SFT Tab ============


def render_sft_tab():
    st.markdown(
        '<div class="main-title">🧠 OpenThoughts-Agent-v1-SFT</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="subtitle">~15,200 conversation traces for supervised fine-tuning<br>'
        '<span style="font-size: 0.75rem; opacity: 0.7;">All traces generated with terminus-2 harness + QuantTrio/GLM-4.6-AWQ model</span></div>',
        unsafe_allow_html=True,
    )

    ds = load_sft_data()
    tasks, sorted_task_ids = build_sft_index(ds)

    # Stats row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"""
        <div class="stat-box">
            <div class="stat-value">{len(ds):,}</div>
            <div class="stat-label">Total Traces</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
        <div class="stat-box">
            <div class="stat-value">{len(tasks):,}</div>
            <div class="stat-label">Unique Tasks</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col3:
        nl2bash_count = sum(1 for t in sorted_task_ids if t.startswith("task_"))
        inferredbugs_count = sum(
            1 for t in sorted_task_ids if t.startswith("inferredbugs-")
        )
        st.markdown(
            f"""
        <div class="stat-box">
            <div class="stat-value">{nl2bash_count:,} / {inferredbugs_count:,}</div>
            <div class="stat-label">nl2bash / InferredBugs</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Filters
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        task_type_filter = st.selectbox(
            "Task Type", ["All", "nl2bash", "inferredbugs"], index=0, key="sft_type"
        )
    with filter_col2:
        sort_by = st.selectbox(
            "Sort by",
            ["Task ID", "Msgs ↑", "Msgs ↓", "Tokens ↑", "Tokens ↓"],
            index=0,
            key="sft_sort",
        )

    # Apply filters
    filtered_task_ids = [
        t
        for t in sorted_task_ids
        if task_type_filter == "All" or tasks[t]["task_type"] == task_type_filter
    ]

    # Apply sorting
    if sort_by == "Msgs ↑":
        filtered_task_ids.sort(key=lambda t: tasks[t]["msgs"])
    elif sort_by == "Msgs ↓":
        filtered_task_ids.sort(key=lambda t: tasks[t]["msgs"], reverse=True)
    elif sort_by == "Tokens ↑":
        filtered_task_ids.sort(key=lambda t: tasks[t]["assistant_tokens"])
    elif sort_by == "Tokens ↓":
        filtered_task_ids.sort(key=lambda t: tasks[t]["assistant_tokens"], reverse=True)

    # Pagination
    page_size = 50
    total_pages = max(1, (len(filtered_task_ids) + page_size - 1) // page_size)
    page = st.number_input(
        "Page", min_value=1, max_value=total_pages, value=1, step=1, key="sft_page"
    )
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, len(filtered_task_ids))

    st.caption(
        f"Showing tasks {start_idx + 1} to {end_idx} of {len(filtered_task_ids):,}"
    )

    # Build table
    table_data = []
    for i in range(start_idx, end_idx):
        task_id = filtered_task_ids[i]
        task = tasks[task_id]
        table_data.append(
            {
                "table_idx": i,
                "task_id": task_id,
                "msgs": task["msgs"],
                "tokens": task["assistant_tokens"],
                "preview": task["preview"][:250]
                + ("..." if len(task["preview"]) >= 250 else ""),
            }
        )

    import pandas as pd

    df = pd.DataFrame(table_data)

    event = st.dataframe(
        df,
        column_config={
            "table_idx": None,
            "task_id": st.column_config.Column("Task ID", width=120),
            "msgs": st.column_config.Column("Msgs", width=60),
            "tokens": st.column_config.Column("Asst Tokens", width=90),
            "preview": st.column_config.Column("Preview", width=1200),
        },
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="sft_table",
    )

    # Show selected task details
    if event.selection and event.selection.rows:
        selected_row = event.selection.rows[0]
        task_id = table_data[selected_row]["task_id"]
        task = tasks[task_id]

        st.divider()
        st.subheader(f"Task: {task_id}")
        st.caption(
            f"{task['msgs']} messages | ~{task['assistant_tokens']:,} assistant tokens"
        )

        sample = ds[task["ds_idx"]]

        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown("**Metadata**")
            for key, value in sample.items():
                if key != "conversations":
                    st.code(f"{key}: {value}", language=None)

        with col2:
            st.markdown("**Conversation**")
            conversations = sample.get("conversations", [])

            is_first_user = True
            for msg in conversations:
                role = msg.get("role", "unknown")
                content = msg.get("content", "").strip()

                if role == "user":
                    css_class = "msg-user"
                    icon = "👤"
                elif role == "assistant":
                    css_class = "msg-assistant"
                    icon = "🤖"
                else:
                    css_class = "msg-system"
                    icon = "⚙️"

                st.markdown(
                    f'<div class="{css_class}"><div class="msg-role">{icon} {role}</div></div>',
                    unsafe_allow_html=True,
                )

                if role == "user" and is_first_user and "Task Description:" in content:
                    is_first_user = False
                    split_idx = content.find("Task Description:")
                    prefix = content[:split_idx].strip()
                    task_content = content[split_idx:].strip()

                    with st.expander("📋 Shared Prefix (boilerplate)", expanded=False):
                        st.code(prefix, language=None)
                    st.markdown("**📌 Actual Task:**")
                    st.code(task_content, language=None)
                else:
                    if role == "user":
                        is_first_user = False
                    if len(content) > 2000:
                        with st.expander(f"View full content ({len(content):,} chars)"):
                            st.code(content, language=None)
                    else:
                        st.code(content, language=None)


# ============ RL Tab ============


def render_rl_tab():
    st.markdown(
        '<div class="main-title">🎮 OpenThoughts-Agent-v1-RL</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="subtitle">728 RL tasks from nl2bash verified dataset<br>'
        '<span style="font-size: 0.75rem; opacity: 0.7;">Each task contains instruction, Dockerfile, tests, and solution</span></div>',
        unsafe_allow_html=True,
    )

    ds = load_rl_data()
    tasks = build_rl_index(ds)

    # Stats row
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"""
        <div class="stat-box">
            <div class="stat-value">{len(ds):,}</div>
            <div class="stat-label">Total Tasks</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col2:
        avg_size = sum(t["binary_size"] for t in tasks) // len(tasks) if tasks else 0
        st.markdown(
            f"""
        <div class="stat-box">
            <div class="stat-value">{avg_size:,} bytes</div>
            <div class="stat-label">Avg Archive Size</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Pagination
    page_size = 50
    total_pages = max(1, (len(tasks) + page_size - 1) // page_size)
    page = st.number_input(
        "Page", min_value=1, max_value=total_pages, value=1, step=1, key="rl_page"
    )
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, len(tasks))

    st.caption(f"Showing tasks {start_idx + 1} to {end_idx} of {len(tasks):,}")

    # Build table
    table_data = []
    for i in range(start_idx, end_idx):
        task = tasks[i]
        table_data.append(
            {
                "table_idx": i,
                "path": task["path"],
                "size": f"{task['binary_size']:,} bytes",
            }
        )

    import pandas as pd

    df = pd.DataFrame(table_data)

    event = st.dataframe(
        df,
        column_config={
            "table_idx": None,
            "path": st.column_config.Column("Task ID", width=120),
            "size": st.column_config.Column("Archive Size", width=120),
        },
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="rl_table",
    )

    # Show selected task details
    if event.selection and event.selection.rows:
        selected_row = event.selection.rows[0]
        task = tasks[start_idx + selected_row]

        st.divider()

        # Header with Open Environment and Run Agent buttons
        header_col1, header_col2, header_col3 = st.columns([2, 1, 1])
        with header_col1:
            st.subheader(f"Task: {task['path']}")
        with header_col2:
            open_env_clicked = st.button(
                "🚀 Open Environment", type="primary", key="open_env_btn"
            )
        with header_col3:
            run_agent_clicked = st.button(
                "🤖 Run Claude Code", type="secondary", key="run_agent_btn"
            )

        # Decode the binary
        row = ds[task["ds_idx"]]
        task_binary = row.get("task_binary", b"")
        files = decode_task_binary(task_binary)

        # Handle Open Environment button
        if open_env_clicked:
            # Check if API key is in environment first, then session state
            env_api_key = os.environ.get("DAYTONA_API_KEY")
            if env_api_key:
                st.session_state.daytona_api_key = env_api_key
                st.session_state.spinning_up_env = True
                st.session_state.selected_task_binary = task_binary
                st.session_state.selected_task_path = task["path"]
            elif (
                "daytona_api_key" not in st.session_state
                or not st.session_state.daytona_api_key
            ):
                st.session_state.show_api_key_dialog = True
            else:
                st.session_state.spinning_up_env = True
                st.session_state.selected_task_binary = task_binary
                st.session_state.selected_task_path = task["path"]

        # API Key dialog
        if st.session_state.get("show_api_key_dialog", False):
            with st.container():
                st.markdown("### 🔐 Daytona API Key Required")
                st.markdown(
                    "Enter your Daytona API key to spin up environments. Get one at [app.daytona.io](https://app.daytona.io)"
                )
                api_key_input = st.text_input(
                    "API Key",
                    type="password",
                    placeholder="dtn_...",
                    key="api_key_input",
                )
                col_save, col_cancel = st.columns(2)
                with col_save:
                    if st.button("Save & Continue", type="primary"):
                        if api_key_input and api_key_input.startswith("dtn_"):
                            st.session_state.daytona_api_key = api_key_input
                            st.session_state.show_api_key_dialog = False
                            st.session_state.spinning_up_env = True
                            st.session_state.selected_task_binary = task_binary
                            st.session_state.selected_task_path = task["path"]
                            st.rerun()
                        else:
                            st.error(
                                "Please enter a valid Daytona API key (starts with dtn_)"
                            )
                with col_cancel:
                    if st.button("Cancel"):
                        st.session_state.show_api_key_dialog = False
                        st.rerun()

        # Spin up environment
        if st.session_state.get("spinning_up_env", False):
            st.session_state.spinning_up_env = False
            task_binary_to_use = st.session_state.get(
                "selected_task_binary", task_binary
            )

            with st.status(
                "🚀 Spinning up Daytona environment via Harbor...", expanded=True
            ) as status:
                st.write("Extracting task files...")
                task_dir = extract_task_to_tempdir(task_binary_to_use)

                if task_dir:
                    dockerfile_path = task_dir / "environment" / "Dockerfile"
                    if dockerfile_path.exists():
                        st.write(f"Task extracted to: {task_dir}")
                        st.write(
                            "Building environment with Harbor (this may take 1-2 minutes)..."
                        )

                        try:
                            result = run_async(
                                create_harbor_daytona_env(
                                    st.session_state.daytona_api_key, task_dir
                                )
                            )
                            status.update(
                                label="✅ Environment ready!", state="complete"
                            )

                            st.success(f"**Task:** `{result['task_name']}` | **Sandbox ID:** `{result['sandbox_id']}`")
                            st.code(result["ssh_command"], language="bash")
                            st.info(
                                "Run the SSH command above in your terminal to connect. "
                                "Solution is at /oracle/, tests at /tests/. "
                                "Environment auto-deletes in 30 minutes."
                            )

                            # Store active sandbox info
                            st.session_state.active_sandbox = result
                        except Exception as e:
                            status.update(
                                label="❌ Failed to create environment", state="error"
                            )
                            st.error(f"Error: {e}")
                    else:
                        status.update(label="❌ No Dockerfile found", state="error")
                        st.error(
                            "This task doesn't have a Dockerfile in the environment/ folder."
                        )
                else:
                    status.update(label="❌ Failed to extract task", state="error")
                    st.error("Could not extract task archive.")

        # Handle Run Agent button
        if run_agent_clicked:
            env_api_key = os.environ.get("DAYTONA_API_KEY")
            if env_api_key:
                st.session_state.daytona_api_key = env_api_key
            if not st.session_state.get("daytona_api_key"):
                st.error("Please set DAYTONA_API_KEY in .env or enter it above")
            elif not os.environ.get("ANTHROPIC_API_KEY"):
                st.error("ANTHROPIC_API_KEY not found in environment")
            else:
                st.session_state.running_agent = True
                st.session_state.agent_task_binary = task_binary
                st.session_state.agent_task_path = task["path"]

        # Run agent flow
        if st.session_state.get("running_agent", False):
            st.session_state.running_agent = False
            agent_task_binary = st.session_state.get("agent_task_binary", task_binary)

            with st.status(
                "🤖 Running Claude Code agent...", expanded=True
            ) as agent_status:
                st.write("Extracting task files...")
                task_dir = extract_task_to_tempdir(agent_task_binary)

                if task_dir:
                    # Read instruction from task
                    instruction_file = task_dir / "instruction.md"
                    if instruction_file.exists():
                        instruction = instruction_file.read_text()
                    else:
                        instruction = "Complete the task in this environment."

                    st.write(f"Instruction: {instruction[:200]}...")

                    try:
                        result = run_async(
                            run_claude_code_agent(
                                task_dir=task_dir,
                                instruction=instruction,
                                daytona_api_key=st.session_state.daytona_api_key,
                                on_status=lambda msg: st.write(msg),
                            )
                        )
                        agent_status.update(
                            label="✅ Agent completed!", state="complete"
                        )

                        # Store result for display
                        st.session_state.agent_result = result
                        st.session_state.active_sandbox = result

                    except Exception as e:
                        agent_status.update(
                            label="❌ Agent failed", state="error"
                        )
                        st.error(f"Error: {e}")
                        import traceback
                        st.code(traceback.format_exc(), language=None)
                else:
                    agent_status.update(label="❌ Failed to extract task", state="error")
                    st.error("Could not extract task archive.")

        # Show agent result with trajectory
        if st.session_state.get("agent_result"):
            result = st.session_state.agent_result
            with st.expander("🤖 Agent Run Result", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Task:** `{result['task_name']}`")
                    st.markdown(f"**Sandbox ID:** `{result['sandbox_id']}`")
                with col2:
                    ctx = result.get("context", {})
                    st.markdown(f"**Input tokens:** {ctx.get('n_input_tokens') or 0:,}")
                    st.markdown(f"**Output tokens:** {ctx.get('n_output_tokens') or 0:,}")

                st.code(result["ssh_command"], language="bash")

                # Show trajectory
                trajectory = result.get("trajectory")
                if trajectory:
                    st.markdown("### 📜 Agent Trajectory")
                    steps = trajectory.get("steps", [])
                    st.caption(f"{len(steps)} steps")

                    for step in steps:
                        source = step.get("source", "unknown")
                        message = step.get("message", "")
                        tool_calls = step.get("tool_calls", [])

                        if source == "agent":
                            icon = "🤖"
                            css_class = "msg-assistant"
                        elif source == "user":
                            icon = "👤"
                            css_class = "msg-user"
                        else:
                            icon = "⚙️"
                            css_class = "msg-system"

                        st.markdown(
                            f'<div class="{css_class}"><div class="msg-role">{icon} {source}</div></div>',
                            unsafe_allow_html=True,
                        )

                        if message:
                            if len(message) > 1000:
                                with st.expander(f"View message ({len(message):,} chars)"):
                                    st.code(message, language=None)
                            else:
                                st.code(message, language=None)

                        for tc in tool_calls:
                            fn_name = tc.get("function_name", "unknown")
                            args = tc.get("arguments", {})
                            st.markdown(f"**Tool:** `{fn_name}`")
                            if args:
                                st.json(args)

                        obs = step.get("observation")
                        if obs and obs.get("results"):
                            for r in obs["results"]:
                                content = r.get("content", "")
                                if content:
                                    with st.expander("📤 Tool output"):
                                        st.code(content[:2000], language=None)
                else:
                    st.warning("No trajectory data available")

        # Show active sandbox if exists
        if st.session_state.get("active_sandbox"):
            with st.expander("🟢 Active Environment", expanded=True):
                sandbox = st.session_state.active_sandbox
                st.markdown(f"**Sandbox ID:** `{sandbox['sandbox_id']}`")
                st.code(sandbox["ssh_command"], language="bash")
                st.caption("Environment auto-deletes in 30 minutes of inactivity.")

        st.caption(f"Archive contains {len(files)} files")

        # Show files in tabs
        if files:
            file_names = list(files.keys())
            file_tabs = st.tabs(file_names)

            for tab, fname in zip(file_tabs, file_names):
                with tab:
                    file_info = files[fname]
                    st.caption(f"{file_info['size']} bytes")

                    # Determine language for syntax highlighting
                    if fname.endswith(".md"):
                        st.markdown(file_info["content"])
                    elif fname.endswith(".json"):
                        st.code(file_info["content"], language="json")
                    elif fname.endswith(".toml"):
                        st.code(file_info["content"], language="toml")
                    elif fname.endswith(".sh"):
                        st.code(file_info["content"], language="bash")
                    elif fname.endswith("Dockerfile"):
                        st.code(file_info["content"], language="dockerfile")
                    else:
                        st.code(file_info["content"], language=None)


# ============ Main ============


def main():
    tab1, tab2 = st.tabs(["📝 SFT Dataset", "🎮 RL Dataset"])

    with tab1:
        render_sft_tab()

    with tab2:
        render_rl_tab()


if __name__ == "__main__":
    main()
