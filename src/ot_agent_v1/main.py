"""
OpenThoughts Agent Datasets Dashboard
"""

import streamlit as st
from datasets import load_dataset

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


@st.cache_data(show_spinner="Loading SFT dataset from HuggingFace...")
def load_sft_data():
    ds = load_dataset("open-thoughts/OpenThoughts-Agent-v1-SFT", split="train")
    return ds


@st.cache_data(show_spinner="Building sorted index...")
def get_sorted_indices(_ds):
    """Return indices sorted by task number ascending."""
    task_nums = []
    for i in range(len(_ds)):
        task = _ds[i].get("task", "task_0")
        # Extract number from "task_XXX"
        try:
            num = int(task.split("_")[1])
        except (IndexError, ValueError):
            num = 0
        task_nums.append((i, num))
    # Sort by task number
    task_nums.sort(key=lambda x: x[1])
    return [idx for idx, _ in task_nums]


def main():
    st.markdown(
        '<div class="main-title">🧠 OpenThoughts-Agent-v1-SFT</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="subtitle">~15,200 conversation traces for supervised fine-tuning</div>',
        unsafe_allow_html=True,
    )

    ds = load_sft_data()

    # Stats row
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"""
        <div class="stat-box">
            <div class="stat-value">{len(ds):,}</div>
            <div class="stat-label">Total Samples</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
        <div class="stat-box">
            <div class="stat-value">{len(ds.features)}</div>
            <div class="stat-label">Fields</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col3:
        sample_size = min(100, len(ds))
        avg_turns = (
            sum(len(ds[i]["conversations"]) for i in range(sample_size)) / sample_size
        )
        st.markdown(
            f"""
        <div class="stat-box">
            <div class="stat-value">{avg_turns:.1f}</div>
            <div class="stat-label">Avg Messages</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Build table data
    page_size = 50
    total_pages = (len(ds) + page_size - 1) // page_size

    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, len(ds))

    # Get sorted indices
    sorted_indices = get_sorted_indices(ds)

    st.caption(f"Showing rows {start_idx + 1} to {end_idx} of {len(ds):,} (sorted by task #)")

    # Create table data for current page
    table_data = []
    for i in range(start_idx, end_idx):
        actual_idx = sorted_indices[i]
        row = ds[actual_idx]
        conv = row.get("conversations", [])
        agent = row.get("agent", "")
        task = row.get("task", "")

        # Extract task number from "task_XXX"
        try:
            task_num = int(task.split("_")[1])
        except (IndexError, ValueError):
            task_num = 0

        # Get task preview - extract after "## Goal" to skip boilerplate
        preview = ""
        for msg in conv:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if "## Goal" in content:
                    goal_idx = content.find("## Goal")
                    goal_text = content[goal_idx + 8 : goal_idx + 150].strip()
                    goal_text = goal_text.replace("\n", " ").strip()
                    preview = goal_text
                else:
                    preview = content[:120].replace("\n", " ").strip()
                break

        table_data.append(
            {
                "idx": actual_idx,
                "task_num": task_num,
                "agent": agent,
                "msgs": len(conv),
                "preview": "User: " + preview + ("..." if len(preview) >= 100 else ""),
            }
        )

    # Display as dataframe with selection
    import pandas as pd

    df = pd.DataFrame(table_data)

    event = st.dataframe(
        df,
        column_config={
            "idx": None,  # hide internal index
            "task_num": st.column_config.NumberColumn("Task #", width=70),
            "agent": st.column_config.TextColumn("Agent", width=80),
            "msgs": st.column_config.NumberColumn("Msgs", width=50),
            "preview": st.column_config.TextColumn("Conversation Preview", width="large"),
        },
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
    )

    # Show selected row details
    if event.selection and event.selection.rows:
        selected_row_idx = event.selection.rows[0]
        actual_idx = table_data[selected_row_idx]["idx"]

        st.divider()
        st.subheader(f"Sample #{actual_idx}")

        sample = ds[actual_idx]

        # Show all fields
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown("**Fields**")
            for key, value in sample.items():
                if key != "conversations":
                    st.code(f"{key}: {value}", language=None)

        with col2:
            st.markdown("**Conversation**")
            conversations = sample.get("conversations", [])

            is_first_user = True
            for msg in conversations:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")

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

                # Split first user message into prefix and task
                if role == "user" and is_first_user and "Task Description:" in content:
                    is_first_user = False
                    split_idx = content.find("Task Description:")
                    prefix = content[:split_idx].strip()
                    task = content[split_idx:].strip()

                    with st.expander("📋 Shared Prefix (boilerplate)", expanded=False):
                        st.code(prefix, language=None)
                    st.markdown("**📌 Actual Task:**")
                    st.code(task, language=None)
                else:
                    if role == "user":
                        is_first_user = False
                    # Show content - truncate if too long
                    if len(content) > 2000:
                        with st.expander(f"View full content ({len(content):,} chars)"):
                            st.code(content, language=None)
                    else:
                        st.code(content, language=None)


if __name__ == "__main__":
    main()
