# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "streamlit",
#     "pandas",
#     "tiktoken",
#     "plotly",
# ]
# ///

import streamlit as st
import json
import pandas as pd
import glob
import os
import sys
import math
import tiktoken
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from streamlit.web import cli as stcli

# --- Path Setup & Imports ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..", "src")
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from ludic.envs.code_exec.adapters.apps import APPS_SYSTEM_PROMPT
except ImportError as e:
    APPS_SYSTEM_PROMPT = f"Error loading system prompt from codebase: {e}"

import re

# --- Configuration & Tokenizer ---
st.set_page_config(layout="wide", page_title="Neural Observatory", page_icon="🔭")


@st.cache_resource
def get_encoder():
    return tiktoken.get_encoding("cl100k_base")


encoder = get_encoder()

COMMON_ERRORS = [
    "SyntaxError", "IndentationError", "NameError", "TypeError",
    "ValueError", "AttributeError", "IndexError", "KeyError",
    "FileNotFoundError", "ImportError", "ModuleNotFoundError",
    "TimeoutError", "AssertionError", "ZeroDivisionError",
    "RuntimeError", "MemoryError", "RecursionError"
]

def extract_primary_error(steps):
    """Scans execution outputs for common Python errors."""
    if not steps:
        return "No Steps"
    
    # Check the last step first as it's the most likely failure point
    for step in reversed(steps):
        out = step.get("next_obs", "")
        if not out: continue
        
        for err in COMMON_ERRORS:
            if err in out:
                return err
        if "Error:" in out:
            return "Generic Error"
            
    return "None"

def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(encoder.encode(text))


# --- Aesthetic System: Neural Observatory ---
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700&family=Inter:wght@300;400;600&family=JetBrains+Mono:wght@400;700&display=swap');

    :root {
        --bg-base: #050505;
        --bg-elevated: #0f1115;
        --bg-card: #13161c;
        --color-primary: #00f0ff; /* Electric Cyan */
        --color-secondary: #7000ff; /* Deep Violet */
        --color-success: #00ff9d;
        --color-failure: #ff0055;
        --text-primary: #e0e0e0;
        --text-secondary: #949aa5;
        --text-muted: #5c616b;
        --font-display: 'Cinzel', serif;
        --font-body: 'Inter', sans-serif;
        --font-mono: 'JetBrains Mono', monospace;
        --glow-strength: 0px 0px 20px rgba(0, 240, 255, 0.1);
    }

    /* Base Theme Overrides */
    .stApp {
        background-color: var(--bg-base);
        font-family: var(--font-body);
        color: var(--text-primary);
    }
    
    h1, h2, h3 {
        font-family: var(--font-display) !important;
        letter-spacing: 0.05em;
        background: linear-gradient(120deg, var(--text-primary), var(--text-secondary));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Metrics Cards */
    .metric-card {
        background: var(--bg-elevated);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 16px;
        position: relative;
        overflow: hidden;
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; width: 100%; height: 2px;
        background: linear-gradient(90deg, var(--color-primary), transparent);
    }
    .metric-value {
        font-family: var(--font-mono);
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--color-primary);
        text-shadow: 0 0 10px rgba(0, 240, 255, 0.3);
    }
    .metric-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--text-secondary);
        margin-bottom: 4px;
    }

    /* Grid Cards */
    .obs-card {
        background: var(--bg-card);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        margin-bottom: 16px;
        height: 100%;
        display: flex;
        flex-direction: column;
    }
    .obs-card:hover {
        transform: translateY(-4px);
        border-color: var(--color-primary);
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5), var(--glow-strength);
    }
    .obs-card-header {
        padding: 12px 16px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(0,0,0,0.2);
    }
    .obs-card-body {
        padding: 16px;
        flex-grow: 1;
    }
    .obs-card-footer {
        padding: 12px 16px;
        background: rgba(0,0,0,0.3);
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        font-size: 0.75rem;
        color: var(--text-secondary);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    /* Status Indicators */
    .status-dot {
        height: 8px;
        width: 8px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 6px;
        box-shadow: 0 0 8px currentColor;
    }
    .status-pass { color: var(--color-success); }
    .status-fail { color: var(--color-failure); }

    /* Code Preview */
    .code-snippet {
        font-family: var(--font-mono);
        font-size: 0.75rem;
        color: var(--text-secondary);
        background: rgba(0,0,0,0.3);
        padding: 10px;
        border-radius: 6px;
        border-left: 2px solid var(--text-muted);
        margin: 10px 0;
        overflow: hidden;
        display: -webkit-box;
        -webkit-line-clamp: 4;
        -webkit-box-orient: vertical;
        opacity: 0.8;
    }

    /* Token Heatmap/Bar */
    .token-bar-bg {
        width: 100%;
        height: 4px;
        background: rgba(255,255,255,0.1);
        border-radius: 2px;
        margin-top: 8px;
        overflow: hidden;
    }
    .token-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, var(--color-primary), var(--color-secondary));
        opacity: 0.8;
    }

    /* Detail View Timeline */
    .timeline-step {
        border-left: 2px solid rgba(255,255,255,0.1);
        padding-left: 24px;
        margin-left: 12px;
        padding-bottom: 32px;
        position: relative;
    }
    .timeline-step:last-child {
        border-left: 2px solid transparent;
    }
    .timeline-node {
        position: absolute;
        left: -9px;
        top: 0;
        width: 16px;
        height: 16px;
        border-radius: 50%;
        background: var(--bg-base);
        border: 2px solid var(--color-primary);
        box-shadow: 0 0 10px var(--color-primary);
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .animate-enter {
        animation: fadeIn 0.4s ease-out forwards;
    }
    
</style>
""",
    unsafe_allow_html=True,
)

# --- Data Loading & Processing ---


@st.cache_data
def load_and_process_data(file_path):
    data = []
    token_counts = []
    rewards = []
    difficulties = []
    step_counts = []
    error_types = []
    ids = []
    pass_stats = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                # Pre-calculate token counts for efficiency
                steps = item.get("steps", [])
                code = steps[0].get("action", "") if steps else ""
                t_count = count_tokens(code)
                item["_token_count"] = t_count
                
                # Extract other metadata
                diff = item.get("meta", {}).get("difficulty", "Unknown")
                err = extract_primary_error(steps)
                
                # Extract Detailed Test Info
                passed = 0
                total = 0
                pass_rate = 0.0
                compile_failed = False
                exec_ms = 0.0
                
                if steps:
                    last_step = steps[-1]
                    last_info = last_step.get("info", {})
                    passed = last_info.get("passed", 0)
                    total = last_info.get("total", 0)
                    compile_failed = last_info.get("compile_failed", False)
                    # Use provided pass_rate or calculate it
                    pass_rate = last_info.get("pass_rate", 0.0)
                    if total > 0 and "pass_rate" not in last_info:
                        pass_rate = passed / total
                    
                    # Extract Timing
                    timing = last_step.get("timing", {})
                    exec_ms = timing.get("total_execution_ms", 0.0)

                data.append(item)
                token_counts.append(t_count)
                rewards.append(item.get("total_reward", 0.0))
                difficulties.append(diff)
                step_counts.append(len(steps))
                error_types.append(err)
                ids.append(item.get("meta", {}).get("problem_id", "N/A"))
                pass_stats.append({
                    "passed": passed, 
                    "total": total, 
                    "pass_rate": pass_rate,
                    "compile_failed": compile_failed,
                    "exec_ms": exec_ms
                })
            except json.JSONDecodeError:
                continue
    
    # Calculate stats for relative visualization
    df = pd.DataFrame({
        "id": ids,
        "tokens": token_counts,
        "reward": rewards,
        "difficulty": difficulties,
        "steps": step_counts,
        "error_type": error_types,
        "pass_rate": [p["pass_rate"] for p in pass_stats],
        "compile_failed": [p["compile_failed"] for p in pass_stats],
        "exec_ms": [p["exec_ms"] for p in pass_stats]
    })
    
    stats = {
        "max_tokens": max(token_counts) if token_counts else 1,
        "avg_tokens": sum(token_counts) / len(token_counts) if token_counts else 0,
        "avg_steps": sum(step_counts) / len(step_counts) if step_counts else 0,
        "success_rate": sum(1 for r in rewards if r == 1.0) / len(rewards) if rewards else 0,
        "compile_fail_rate": sum(1 for p in pass_stats if p["compile_failed"]) / len(pass_stats) if pass_stats else 0
    }
    
    return data, stats, df


# --- View Components ---


def render_metric(label, value, subtext=None, color="primary"):
    color_var = f"var(--color-{color})"
    st.markdown(
        f"""
    <div class="metric-card animate-enter">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color: {color_var}">{value}</div>
        {f'<div style="font-size: 0.7rem; color: var(--text-muted); margin-top: 4px;">{subtext}</div>' if subtext else ""}
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_overview(data, stats, df):
    st.markdown("## 📡 Mission Control")
    
    # Top Level Metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        render_metric("Total Rollouts", len(data), "Active Datastreams")
    with c2:
        render_metric("Success Rate", f"{stats['success_rate']*100:.1f}%", "Mission Completion", 
                      color="success" if stats['success_rate'] > 0.5 else "failure")
    with c3:
        render_metric("Compile Failures", f"{stats['compile_fail_rate']*100:.1f}%", "Syntax/Build Errors", color="failure")
    with c4:
        render_metric("Avg. Latency", f"{df['exec_ms'].mean()/1000:.1f}s", "Inference + Exec")
    with c5:
        render_metric("Avg. Tokens", f"{int(stats['avg_tokens']):,}", "Computational Load")

    st.markdown("### 📊 Telemetry Analysis")
    
    # Row 1: Difficulty & Error Analysis
    col_charts_1, col_charts_2, col_charts_3 = st.columns(3)
    
    with col_charts_1:
        # Success Rate by Difficulty
        if "difficulty" in df.columns:
            diff_stats = df.groupby("difficulty")["reward"].mean().reset_index()
            diff_stats["success_pct"] = diff_stats["reward"] * 100
            
            fig_diff = px.bar(
                diff_stats, x="difficulty", y="success_pct",
                title="Success Rate by Difficulty (%)",
                color="success_pct",
                color_continuous_scale=["#ff0055", "#00ff9d"],
                template="plotly_dark",
                labels={"success_pct": "Success Rate (%)", "difficulty": "Difficulty Level"}
            )
            fig_diff.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            fig_diff.update_yaxes(range=[0, 100])
            st.plotly_chart(fig_diff, width="stretch")
        else:
            st.info("No difficulty metadata found.")
        
    with col_charts_2:
        # Pass Rate Distribution (Partial Credit)
        fig_pass = px.histogram(
            df, x="pass_rate", nbins=20,
            title="Test Pass Rate Distribution",
            color_discrete_sequence=['#00f0ff'],
            template="plotly_dark",
            labels={"pass_rate": "Pass Rate (0.0 - 1.0)"}
        )
        fig_pass.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_pass, width="stretch")

    with col_charts_3:
        # Latency Distribution (Box Plot)
        fig_lat = px.box(
            df, y="exec_ms",
            title="Execution Latency (ms)",
            template="plotly_dark",
            color_discrete_sequence=['#7000ff']
        )
        fig_lat.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_lat, width="stretch")

    # Row 2: Deep Dive Data Table
    st.markdown("### 📋 Data Matrix")
    st.dataframe(
        df,
        column_config={
            "reward": st.column_config.ProgressColumn(
                "Reward",
                help="Total Reward (0.0 to 1.0)",
                min_value=0.0,
                max_value=1.0,
                format="%.2f",
            ),
            "pass_rate": st.column_config.ProgressColumn(
                "Pass Rate",
                help="Fraction of tests passed",
                min_value=0.0,
                max_value=1.0,
                format="%.2f",
            ),
            "exec_ms": st.column_config.NumberColumn("Latency (ms)", format="%d"),
            "tokens": st.column_config.NumberColumn("Tokens", format="%d"),
            "compile_failed": st.column_config.CheckboxColumn("Compile Fail"),
        },
        width="stretch",
        hide_index=True,
        height=300
    )


def render_grid_view(rollouts, global_stats):
    st.markdown(f"## 🔭 Observed Events ({len(rollouts)})")

    # Pagination
    items_per_page = 24
    if "page" not in st.session_state:
        st.session_state.page = 1
    total_pages = math.ceil(len(rollouts) / items_per_page)

    # Controls
    c_ctrl_1, c_ctrl_2 = st.columns([6, 1])
    with c_ctrl_2:
        c_prev, c_page, c_next = st.columns([1, 2, 1])
        if c_prev.button("◀", width="stretch") and st.session_state.page > 1:
            st.session_state.page -= 1
            st.rerun()
        if c_next.button("▶", width="stretch") and st.session_state.page < total_pages:
            st.session_state.page += 1
            st.rerun()
        c_page.caption(f"Pg {st.session_state.page}/{total_pages}")

    # Render Grid
    start_idx = (st.session_state.page - 1) * items_per_page
    batch = rollouts[start_idx : start_idx + items_per_page]

    cols = st.columns(3)
    for i, item in enumerate(batch):
        col = cols[i % 3]
        with col:
            # Prepare Data
            meta = item.get("meta", {})
            pid = meta.get("problem_id", "Unknown")
            diff = meta.get("difficulty", "N/A")
            reward = item.get("total_reward", 0.0)
            token_count = item.get("_token_count", 0)

            steps = item.get("steps", [])
            code = steps[0].get("action", "") if steps else ""
            preview_code = (
                code[:150] + "..." if len(code) > 150 else code or "// No Code Action"
            )

            # Status styling
            is_pass = reward == 1.0
            status_class = "status-pass" if is_pass else "status-fail"
            status_label = "SUCCESS" if is_pass else "FAILURE"
            border_color = (
                "var(--color-success)" if is_pass else "rgba(255, 255, 255, 0.1)"
            )
            if not is_pass:
                border_color = "rgba(255, 0, 85, 0.3)"

            # Token Meter
            token_pct = min(100, (token_count / global_stats["max_tokens"]) * 100)

            # Card HTML
            card_html = f"""
            <div class="obs-card animate-enter" style="animation-delay: {i * 50}ms; border-left: 3px solid {border_color}">
                <div class="obs-card-header">
                    <div style="font-weight: 600; font-size: 0.9rem;">{pid}</div>
                    <div style="font-size: 0.7rem; opacity: 0.7;">{diff}</div>
                </div>
                <div class="obs-card-body">
                    <div style="display: flex; align-items: center; margin-bottom: 8px;">
                        <span class="status-dot {status_class}"></span>
                        <span style="font-size: 0.75rem; font-weight: 700; letter-spacing: 0.05em;">{status_label}</span>
                    </div>
                    <div class="code-snippet">{preview_code}</div>
                    <div style="margin-top: 12px;">
                        <div style="display: flex; justify-content: space-between; font-size: 0.7rem; color: var(--text-muted);">
                            <span>TOKEN USAGE</span>
                            <span>{token_count:,} T</span>
                        </div>
                        <div class="token-bar-bg">
                            <div class="token-bar-fill" style="width: {token_pct}%;"></div>
                        </div>
                    </div>
                </div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)

            # Invisible button overlay for interaction
            if st.button(f"Analyze {pid}", key=f"btn_{item['id']}", width="stretch"):
                st.session_state.selected_rollout = item
                st.rerun()


def render_detail_view():
    item = st.session_state.selected_rollout
    meta = item.get("meta", {})
    steps = item.get("steps", [])

    # Navigation
    if st.button("← Return to Observatory"):
        st.session_state.selected_rollout = None
        st.rerun()

    # Header
    st.markdown(f"# 🧬 Analysis: {meta.get('problem_id', 'Unknown')}")

    # KPI Strip
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_metric(
            "Reward Signal",
            item.get("total_reward"),
            color="success" if item.get("total_reward") == 1 else "failure",
        )
    with k2:
        render_metric("Total Tokens", f"{item.get('_token_count', 0):,}", "Cost Factor")
    with k3:
        render_metric("Interaction Depth", len(steps), "Steps taken")
    with k4:
        render_metric("Difficulty", meta.get("difficulty", "N/A"))

    st.markdown("---")
    
    # Layout: System Prompt | Interaction Timeline
    col_sys, col_main = st.columns([1, 3])
    
    with col_sys:
        with st.expander("🖥️ System Directive", expanded=False):
            st.markdown("*Initial System Instructions provided to the agent.*")
            st.code(APPS_SYSTEM_PROMPT, language="text")
            
        st.markdown("### Metadata")
        st.json(meta)

        # Test Results Inspection
        if steps:
            last_info = steps[-1].get("info", {})
            
            if last_info.get("compile_failed"):
                st.error("🚨 Compilation Failed")
            
            if "test_results" in last_info:
                st.markdown("### 🧪 Test Results")
                test_results = last_info["test_results"]
                if test_results:
                    tdf = pd.DataFrame(test_results)
                    # Select relevant columns if they exist
                    cols = [c for c in ["test_id", "passed", "failure_reason", "run_duration_ms"] if c in tdf.columns]
                    st.dataframe(
                        tdf[cols], 
                        width="stretch",
                        column_config={
                            "passed": st.column_config.CheckboxColumn("Pass"),
                            "run_duration_ms": st.column_config.NumberColumn("Time (ms)", format="%.1f")
                        }
                    )
            
            if "passed" in last_info:
                p = last_info["passed"]
                t = last_info["total"]
                st.metric("Tests Passed", f"{p}/{t}", f"{(p/t*100) if t else 0:.1f}% Rate")

    with col_main:
        st.markdown("## 🎞️ Interaction Reconstruction")

        for i, step in enumerate(steps):
            # Step Container
            st.markdown(
                f"""
            <div class="timeline-step">
                <div class="timeline-node"></div>
                <h3 style="margin-top: -5px; font-size: 1.1rem; color: var(--color-primary);">Step {i + 1}</h3>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # 1. Observation/Prompt
            with st.container():
                prev_obs = step.get("prev_obs", "")
                obs_tokens = count_tokens(prev_obs)
                st.markdown(
                    f"**📡 Input / Observation** <span style='color:var(--text-muted); font-size:0.8em'>({obs_tokens} tokens)</span>",
                    unsafe_allow_html=True,
                )
                if len(prev_obs) > 300:
                    with st.expander(f"Show Input ({obs_tokens} tokens)"):
                        st.code(prev_obs, language="text")
                else:
                    st.code(prev_obs, language="text")

            # 2. Action (Code)
            with st.container():
                action = step.get("action", "")
                toks = count_tokens(action)
                st.markdown(
                    f"**⚡ Agent Action** <span style='color:var(--text-muted); font-size:0.8em'>({toks} tokens)</span>",
                    unsafe_allow_html=True,
                )
                st.code(action, language="python", line_numbers=True)

            # 3. Output
            with st.container():
                out = step.get("next_obs")
                if out:
                    st.markdown("**⚙️ Execution Output**")
                    if len(out) > 1000:
                        st.code(
                            out[:300] + "\n... [TRUNCATED] ...\n" + out[-300:],
                            language="text",
                        )
                        with st.expander("View Full Output"):
                            st.code(out, language="text")
                    else:
                        st.code(out, language="text")
                else:
                    st.info("No output recorded for this step.")

            st.markdown("<div style='height: 24px'></div>", unsafe_allow_html=True)


# --- Main App Logic ---


def main():
    # Sidebar
    with st.sidebar:
        st.markdown("### 🔭 Neural Observatory")

        # File Selector
        search_paths = ["*.jsonl", "data/*.jsonl", "logs/**/*.jsonl", "**/*.jsonl"]
        files = []
        for p in search_paths:
            files.extend(glob.glob(p, recursive=True))
        files = sorted(list(set(files)), key=os.path.getmtime, reverse=True)

        selected_file = st.selectbox("Select Datastream", files) if files else None

        if not selected_file:
            st.warning("No data found.")
            return

        # Load Data
        with st.spinner("Decoding telemetry..."):
            data, stats, df = load_and_process_data(selected_file)

        st.markdown("---")
        st.markdown("### 🔍 Filters")

        # View Mode
        view_mode = st.radio("View Mode", ["Dashboard", "Grid Inspection"])

        # Filters
        diffs = ["All"] + sorted(
            list(set(d.get("meta", {}).get("difficulty", "unknown") for d in data))
        )
        f_diff = st.selectbox("Difficulty Class", diffs)
        
        # Error Filter
        all_errors = ["All"] + sorted(list(set(df["error_type"].unique())))
        f_error = st.selectbox("Error Type", all_errors)

        f_outcome = st.radio("Outcome State", ["All", "Success", "Failure"])

        f_search = st.text_input("Search (ID or Content)")
        deep_search = st.checkbox("Deep Search (Scan Code/Output)", value=False)

        # Filter Logic
        filtered = data
        if f_diff != "All":
            filtered = [
                d for d in filtered if d.get("meta", {}).get("difficulty") == f_diff
            ]
        if f_error != "All":
            # Match against the pre-calculated error type in df, 
            # but we need to map back to the list of dicts. 
            # Efficient way: filter ids based on df filter
            target_ids = set(df[df["error_type"] == f_error]["id"])
            filtered = [d for d in filtered if d.get("meta", {}).get("problem_id") in target_ids]
            
        if f_outcome == "Success":
            filtered = [d for d in filtered if d.get("total_reward") == 1.0]
        elif f_outcome == "Failure":
            filtered = [d for d in filtered if d.get("total_reward") != 1.0]
            
        if f_search:
            term = f_search.lower()
            if deep_search:
                # Heavy scan
                new_filtered = []
                for d in filtered:
                    # Check ID
                    if term in str(d.get("meta", {}).get("problem_id", "")).lower():
                        new_filtered.append(d)
                        continue
                    # Check content
                    found = False
                    for step in d.get("steps", []):
                        if term in step.get("action", "").lower() or term in step.get("next_obs", "").lower() or term in step.get("prev_obs", "").lower():
                            found = True
                            break
                    if found:
                        new_filtered.append(d)
                filtered = new_filtered
            else:
                # Light scan (ID/Metadata only)
                filtered = [d for d in filtered if term in str(d).lower()]

        st.markdown("---")
        st.caption(f"v2.1.0 | {len(filtered)} records active")

    # Main Router
    if st.session_state.get("selected_rollout"):
        render_detail_view()
    elif view_mode == "Dashboard":
        render_overview(filtered, stats, df)
    else:
        render_grid_view(filtered, stats)


if __name__ == "__main__":
    if st.runtime.exists():
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
