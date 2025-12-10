# Rich Imports for Dashboard
from rich.table import Table
from rich import box


# ---------------------------------------------------------------------------
# Beautiful CLI Dashboard
# ---------------------------------------------------------------------------


def create_dashboard(stats: dict, step: int, total_steps: int) -> Table:
    """Creates a rich table for the current training step."""
    table = Table(
        box=box.ROUNDED,
        title=f"🚀 Ludic Training (Step {step}/{total_steps})",
        width=100,
    )

    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_column("Visual", style="green")

    # 1. Main RL Stats
    reward = stats.get("avg_total_reward", 0.0)
    loss = stats.get("loss", 0.0)

    # Reward Bar (-1 to 1 range usually)
    r_bar_len = int((reward + 1.0) * 10)  # map -1..1 to 0..20
    r_bar = "█" * max(0, r_bar_len)

    table.add_row("💰 Avg Reward", f"{reward:+.4f}", r_bar)
    table.add_row("📉 Loss", f"{loss:.4f}", "")

    table.add_section()

    # 2. Error Rates (The important part for you!)
    syn_err = stats.get("err_syntax", 0.0)
    sem_err = stats.get("err_semantic", 0.0)

    syn_color = "red" if syn_err > 0.1 else "yellow" if syn_err > 0.0 else "dim"
    sem_color = "red" if sem_err > 0.1 else "yellow" if sem_err > 0.0 else "dim"

    table.add_row(
        "🚫 Syntax Errors", f"[{syn_color}]{syn_err:.1%}[/]", "Invalid XML (<move>...)"
    )
    table.add_row(
        "⚠️ Illegal Moves", f"[{sem_color}]{sem_err:.1%}[/]", "Occupied/OOB cell"
    )

    table.add_section()

    # 3. Outcomes
    win = stats.get("rate_win", 0.0)
    loss_rate = stats.get("rate_loss", 0.0)
    draw = stats.get("rate_draw", 0.0)

    table.add_row("🏆 Win Rate", f"{win:.1%}", "")
    table.add_row("💀 Loss Rate", f"{loss_rate:.1%}", "")
    table.add_row("🤝 Draw Rate", f"{draw:.1%}", "")

    # 4. Tech Stats
    items = int(stats.get("batch_items", 0))
    bs = int(stats.get("batch_size", 0))
    table.add_section()
    table.add_row("📦 Batch Size", f"{bs} eps / {items} steps", "")

    return table
