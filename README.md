# OpenThoughts Agent Dataset Viewer

A Streamlit dashboard for exploring the [OpenThoughts Agent](https://www.openthoughts.ai/blog/agent) datasets and running AI agents on RL tasks in live cloud environments.

![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)

## Overview

This project provides an interactive interface to:

1. **Browse SFT training data** — Explore ~15,200 conversation traces used for supervised fine-tuning
2. **Explore RL task environments** — Inspect 728 dockerized task definitions with seeds, tests, and solutions
3. **Run agents on tasks** — Spin up live cloud environments via Daytona and run multiple AI agents
4. **Visualize agent trajectories** — See step-by-step thinking, tool calls, and outputs from agent runs
5. **Debug failed tasks** — Launch Claude Code to SSH into containers and investigate failures

## Features

### 📝 SFT Dataset Browser

- **~15,200 conversation traces** from agent interactions
- **Two task types**: `nl2bash` (natural language → bash) and `InferredBugs` (bug detection/fixing)
- **Filtering & sorting** by task type, message count, and token count
- **Conversation viewer** with syntax highlighting and collapsible sections
- **Metadata inspection** for each trace

### 🎮 RL Task Runner

- **728 RL tasks** from the nl2bash verified dataset
- **Live environment provisioning** via [Daytona](https://daytona.io) cloud containers
- **One-click task execution** with automatic test verification
- **SSH access** to running containers for manual inspection

### 🤖 Multi-Agent Support

Run tasks with different AI agents via the [Harbor](https://harborframework.com) framework:

| Agent           | Provider       | Trajectory Support | Notes                     |
| --------------- | -------------- | ------------------ | ------------------------- |
| **Claude Code** | Anthropic      | ✅ Full             | Extended thinking traces  |
| **Terminus2**   | Multi-provider | ✅ Full             | External agent            |
| **Codex**       | OpenAI         | ✅ Full             | Reasoning encrypted       |
| **Gemini CLI**  | Google         | ✅ Full             | —                         |
| **SWE-Agent**   | Multi-provider | ✅ Full             | ⚠️ Requires Daytona Tier 3 |
| **OpenHands**   | Multi-provider | ✅ Full             | ⚠️ Requires Daytona Tier 3 |
| **Cline CLI**   | Multi-provider | ⚠️ Limited          | Raw output                |
| **Cursor CLI**  | Multi-provider | ⚠️ Limited          | Raw output                |

### 🧠 Model Selection

Each agent supports different LLM providers:

- **Anthropic**: Claude Opus 4.5, Claude Sonnet 4.5
- **OpenAI**: GPT-5.2, GPT-5.1 Codex Max
- **Google**: Gemini 3 Flash

### 📊 Trajectory Visualization

- **Thinking traces** — See the agent's reasoning process
- **Tool calls** — View every tool invocation with inputs/outputs
- **ATIF format support** — Unified trajectory format across agents
- **Token usage** — Track input/output token counts and costs

### 🔍 Debug Mode

When a task fails, launch Claude Code to investigate:
- SSH into the running container
- Read test scripts and agent logs
- Compare agent output against expected results
- Get a brief explanation of what went wrong

## Quick Start

```bash
# Install dependencies
uv sync

# Set required API keys
export DAYTONA_API_KEY="dtn_..."  # Required for all agents
export ANTHROPIC_API_KEY="sk-ant-..."  # For Claude Code, Terminus2
export OPENAI_API_KEY="sk-..."  # For Codex
export GEMINI_API_KEY="AIza..."  # For Gemini CLI

# Run the dashboard
uv run streamlit run src/ot_agent_v1/main.py
```

Or configure API keys directly in the sidebar after launching.

## Requirements

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** package manager
- **Daytona API key** — [Get one at app.daytona.io](https://app.daytona.io)
- **Agent API keys** — Anthropic, OpenAI, and/or Gemini depending on which agents you use

### Daytona Tier Requirements

Some agents (marked with ⚠️) require **Daytona Tier 3** for network access during installation. The free tier blocks outbound network requests, which prevents `uv`/`pip` from installing dependencies.

## Project Structure

```
src/ot_agent_v1/
├── main.py        # Streamlit dashboard (SFT + RL tabs, UI components)
├── env.py         # Daytona environment management, Harbor agent integration
└── evaluator.py   # Debug agent using Claude Agent SDK
```

## How It Works

### Task Execution Flow

1. **Extract task** — Decode gzipped tar archive from HuggingFace dataset
2. **Create environment** — Spin up Daytona container with task files (Dockerfile, seeds, tests)
3. **Install agent** — Use Harbor's AgentFactory to set up the selected agent
4. **Run agent** — Execute agent with task instruction
5. **Collect trajectory** — Download agent logs and convert to unified format
6. **Verify solution** — Run `test.sh` and check reward file
7. **Display results** — Show pass/fail, trajectory, and debug options

### Harbor Integration

This project uses [Harbor](https://harborframework.com) for:
- **Agent abstraction** — Unified interface across different agent implementations
- **Environment management** — Daytona container lifecycle
- **Task format** — Standard structure for instructions, tests, and solutions
- **Trajectory logging** — ATIF (Agent Trajectory Interchange Format)

## Links

- [OpenThoughts Agent Blog Post](https://www.openthoughts.ai/blog/agent)
- [SFT Dataset on HuggingFace](https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-v1-SFT)
- [RL Dataset on HuggingFace](https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-v1-RL)
- [Harbor Framework](https://harborframework.com/docs/task-format)
- [Daytona](https://daytona.io)
- [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk)

## License

MIT
