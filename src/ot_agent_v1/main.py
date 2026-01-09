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
    extract_task_to_tempdir,
    run_async,
    run_agent,
    AGENT_CONFIG,
    SUPPORTED_AGENTS,
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

        # Clear previous results when task changes
        if st.session_state.get("current_task_path") != task["path"]:
            st.session_state.current_task_path = task["path"]
            st.session_state.agent_result = None
            st.session_state.active_sandbox = None

        st.divider()

        # Header with task name and controls
        st.subheader(f"Task: {task['path']}")

        # Agent selection and Start button
        control_col1, control_col2, control_col3 = st.columns([2, 2, 1])
        with control_col1:
            # Build agent options from AGENT_CONFIG (display_name -> agent_name)
            agent_options = {
                AGENT_CONFIG[agent_name]["display_name"]: agent_name
                for agent_name in SUPPORTED_AGENTS
            }
            selected_display_name = st.selectbox(
                "Agent Harness",
                options=list(agent_options.keys()),
                index=0,
                key="agent_harness_select",
            )
            selected_agent_name = agent_options[selected_display_name]
        with control_col2:
            # Show API key requirement for selected agent
            agent_config = AGENT_CONFIG[selected_agent_name]
            api_key_env = agent_config["api_key_env"]
            default_model = agent_config["default_model"]
            st.caption(f"**Requires:** `{api_key_env}`")
            st.caption(f"**Model:** `{default_model}`")
        with control_col3:
            start_task_clicked = st.button(
                "🚀 Start Task",
                type="primary",
                key="start_task_btn",
                use_container_width=True,
            )

        # Decode the binary
        row = ds[task["ds_idx"]]
        task_binary = row.get("task_binary", b"")
        files = decode_task_binary(task_binary)

        # Handle Start Task button
        if start_task_clicked:
            # Check API keys
            env_api_key = os.environ.get("DAYTONA_API_KEY")
            if env_api_key:
                st.session_state.daytona_api_key = env_api_key

            # Get the required API key for the selected agent
            required_api_key = AGENT_CONFIG[selected_agent_name]["api_key_env"]

            if not st.session_state.get("daytona_api_key"):
                st.session_state.show_api_key_dialog = True
            elif not os.environ.get(required_api_key):
                st.error(f"{required_api_key} not found in .env (required for {selected_display_name})")
            else:
                st.session_state.running_task = True
                st.session_state.task_binary = task_binary
                st.session_state.task_path = task["path"]
                st.session_state.selected_agent_name = selected_agent_name

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
                            st.session_state.running_task = True
                            st.session_state.task_binary = task_binary
                            st.session_state.task_path = task["path"]
                            st.session_state.selected_agent_name = selected_agent_name
                            st.rerun()
                        else:
                            st.error(
                                "Please enter a valid Daytona API key (starts with dtn_)"
                            )
                with col_cancel:
                    if st.button("Cancel"):
                        st.session_state.show_api_key_dialog = False
                        st.rerun()

        # Run task flow (environment + agent)
        if st.session_state.get("running_task", False):
            st.session_state.running_task = False
            task_binary_to_use = st.session_state.get("task_binary", task_binary)
            agent_name_to_use = st.session_state.get("selected_agent_name", "claude-code")
            agent_display_name = AGENT_CONFIG[agent_name_to_use]["display_name"]

            with st.status(f"Starting task with {agent_display_name}...", expanded=True) as task_status:
                st.write("Extracting task files...")
                task_dir = extract_task_to_tempdir(task_binary_to_use)

                if not task_dir:
                    task_status.update(label="❌ Failed to extract task", state="error")
                    st.error("Could not extract task archive.")
                else:
                    # Read instruction
                    instruction_file = task_dir / "instruction.md"
                    if instruction_file.exists():
                        instruction = instruction_file.read_text()
                    else:
                        instruction = "Complete the task in this environment."

                    st.write(f"**Agent:** {agent_name_to_use}")
                    st.write(f"**Goal:** {instruction[:300]}...")

                    try:
                        import threading
                        import time

                        # Capture values before thread (st.session_state not accessible in thread)
                        daytona_key = st.session_state.daytona_api_key

                        # Collect status messages (can't use st.write from thread)
                        status_log = []
                        task_done = threading.Event()
                        task_result = [None]
                        task_error = [None]

                        def run_task():
                            try:
                                task_result[0] = run_async(
                                    run_agent(
                                        task_dir=task_dir,
                                        instruction=instruction,
                                        daytona_api_key=daytona_key,
                                        agent_name=agent_name_to_use,
                                        status_log=status_log,
                                    )
                                )
                            except Exception as e:
                                task_error[0] = e
                            finally:
                                task_done.set()

                        # Start task in background
                        task_thread = threading.Thread(target=run_task)
                        task_thread.start()

                        # Poll and display progress
                        progress_placeholder = st.empty()
                        displayed_count = 0

                        while not task_done.is_set():
                            # Show new status messages
                            if len(status_log) > displayed_count:
                                with progress_placeholder.container():
                                    for msg in status_log:
                                        st.write(msg)
                                displayed_count = len(status_log)
                            time.sleep(0.3)

                        # Final display of all messages
                        with progress_placeholder.container():
                            for msg in status_log:
                                st.write(msg)

                        task_thread.join()

                        if task_error[0]:
                            raise task_error[0]

                        result = task_result[0]

                        task_status.update(label="✅ Task completed!", state="complete")

                        # Store result for display
                        st.session_state.agent_result = result
                        st.session_state.active_sandbox = result

                    except Exception as e:
                        task_status.update(label="❌ Task failed", state="error")
                        st.error(f"Error: {e}")
                        import traceback

                        st.code(traceback.format_exc(), language=None)

        # Show agent result with trajectory
        if st.session_state.get("agent_result"):
            result = st.session_state.agent_result

            # Show test result prominently
            test_passed = result.get("test_passed")
            if test_passed is True:
                st.success("✅ **Task PASSED** - Agent solution verified!")
            elif test_passed is False:
                st.error("❌ **Task FAILED** - Solution did not pass tests")
            else:
                st.warning("⚠️ **Test status unknown**")

            with st.expander("🤖 Agent Run Result", expanded=True):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"**Task:** `{result['task_name']}`")
                    st.markdown(f"**Agent:** `{result.get('agent_display_name', 'Unknown')}`")
                    st.markdown(f"**Model:** `{result.get('model_name', 'Unknown')}`")
                with col2:
                    ctx = result.get("context", {})
                    st.markdown(f"**Input tokens:** {ctx.get('n_input_tokens') or 0:,}")
                    st.markdown(
                        f"**Output tokens:** {ctx.get('n_output_tokens') or 0:,}"
                    )
                    st.markdown(f"**Sandbox ID:** `{result['sandbox_id']}`")
                with col3:
                    if test_passed is True:
                        st.markdown("**Result:** ✅ PASSED")
                    elif test_passed is False:
                        st.markdown("**Result:** ❌ FAILED")
                    else:
                        st.markdown("**Result:** ⚠️ Unknown")

                st.code(result["ssh_command"], language="bash")

                # Show test output if failed
                test_output = result.get("test_output")
                if test_passed is False and test_output:
                    with st.expander("📋 Test Output", expanded=True):
                        st.code(test_output, language=None)

                # Show trajectory
                trajectory = result.get("trajectory")
                if trajectory:
                    st.markdown("### 📜 Agent Trajectory")
                    # Handle both "steps" (terminus-2) and "events" (claude-code fallback) formats
                    steps = trajectory.get("steps") or trajectory.get("events", [])
                    
                    # Filter and count meaningful events
                    meaningful_events = [
                        s for s in steps
                        if s.get("type") not in ("system", "result")
                    ]
                    st.caption(f"{len(meaningful_events)} turns")

                    for step in steps:
                        event_type = step.get("type", "unknown")
                        
                        # Skip system init and result events
                        if event_type == "system":
                            continue
                        if event_type == "result":
                            continue

                        # Parse message content based on event type
                        raw_message = step.get("message", {})
                        
                        if event_type == "assistant" and isinstance(raw_message, dict):
                            # Assistant message: extract text, thinking, and tool calls from content array
                            content_items = raw_message.get("content", [])
                            if isinstance(content_items, list):
                                for item in content_items:
                                    if isinstance(item, dict):
                                        item_type = item.get("type", "")
                                        
                                        # Handle thinking/reasoning blocks
                                        if item_type in ("thinking", "reasoning", "analysis"):
                                            thinking_text = item.get("text", "")
                                            if thinking_text.strip():
                                                st.markdown('<div class="msg-assistant"><div class="msg-role">🤖 assistant</div></div>', unsafe_allow_html=True)
                                                with st.expander("💭 Thinking", expanded=True):
                                                    st.markdown(thinking_text)
                                        
                                        elif item_type == "text":
                                            text = item.get("text", "")
                                            if text.strip():
                                                st.markdown('<div class="msg-assistant"><div class="msg-role">🤖 assistant</div></div>', unsafe_allow_html=True)
                                                st.markdown(text)
                                        
                                        elif item_type == "tool_use":
                                            tool_name = item.get("name", "unknown")
                                            tool_input = item.get("input", {})
                                            st.markdown('<div class="msg-assistant"><div class="msg-role">🤖 assistant</div></div>', unsafe_allow_html=True)
                                            st.markdown(f"**🔧 Tool:** `{tool_name}`")
                                            if tool_input:
                                                st.json(tool_input)
                        
                        elif event_type == "user" and isinstance(raw_message, dict):
                            # User message: typically tool results
                            content_items = raw_message.get("content", [])
                            if isinstance(content_items, list):
                                for item in content_items:
                                    if isinstance(item, dict) and item.get("type") == "tool_result":
                                        tool_output = item.get("content", "")
                                        is_error = item.get("is_error", False)
                                        if tool_output:
                                            st.markdown('<div class="msg-user"><div class="msg-role">👤 user</div></div>', unsafe_allow_html=True)
                                            label = "❌ Error" if is_error else "📤 Output"
                                            with st.expander(label, expanded=False):
                                                if len(tool_output) > 2000:
                                                    st.code(tool_output[:2000] + "\n... (truncated)", language=None)
                                                else:
                                                    st.code(tool_output, language=None)
                        
                        # Handle terminus-2 format (source/message/tool_calls/observation)
                        elif step.get("source"):
                            source = step.get("source")
                            message = step.get("message", "")
                            tool_calls = step.get("tool_calls", [])
                            
                            # Only render if there's content
                            if message or tool_calls:
                                if source == "agent":
                                    st.markdown('<div class="msg-assistant"><div class="msg-role">🤖 agent</div></div>', unsafe_allow_html=True)
                                elif source == "user":
                                    st.markdown('<div class="msg-user"><div class="msg-role">👤 user</div></div>', unsafe_allow_html=True)
                                else:
                                    st.markdown(f'<div class="msg-system"><div class="msg-role">⚙️ {source}</div></div>', unsafe_allow_html=True)
                            
                            if message:
                                if len(message) > 1000:
                                    with st.expander(f"View message ({len(message):,} chars)"):
                                        st.code(message, language=None)
                                else:
                                    st.code(message, language=None)
                            
                            for tc in tool_calls:
                                fn_name = tc.get("function_name", "unknown")
                                args = tc.get("arguments", {})
                                st.markdown(f"**🔧 Tool:** `{fn_name}`")
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
                    # Show raw output for debugging
                    raw_output = result.get("raw_output")
                    if raw_output:
                        st.markdown("### 📄 Raw Agent Output")
                        with st.expander("View raw output", expanded=True):
                            st.code(
                                raw_output[-5000:]
                                if len(raw_output) > 5000
                                else raw_output,
                                language=None,
                            )

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
