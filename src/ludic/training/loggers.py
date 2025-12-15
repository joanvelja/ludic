from __future__ import annotations

from typing import Any, Dict, Protocol, Sequence

from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from rich.table import Table


class TrainingLogger(Protocol):
    """
    Interface for logging training stats to arbitrary backends.
    """

    def log(self, step: int, stats: Dict[str, float]) -> None:
        ...


class PrintLogger:
    """
    Lightweight console logger (plain stdout).
    """

    def __init__(
        self,
        *,
        prefix: str = "[trainer]",
        keys: Sequence[str] | None = None,
        precision: int = 4,
        max_items_per_line: int = 6,
    ) -> None:
        self.prefix = prefix
        self.keys = list(keys) if keys is not None else None
        self.precision = precision
        self.max_items_per_line = max_items_per_line

    def _fmt_val(self, v: Any) -> str:
        if isinstance(v, float):
            return f"{v:.{self.precision}f}"
        if isinstance(v, int):
            return str(v)
        return str(v)

    def log(self, step: int, stats: Dict[str, float]) -> None:
        if self.keys is not None:
            keys = [k for k in self.keys if k in stats]
        else:
            keys = sorted(stats.keys())
        pairs = [f"{k}={self._fmt_val(stats[k])}" for k in keys]
        if not pairs:
            print(f"{self.prefix} step={step}")
            return

        header = f"{self.prefix} step={step}"
        lines = []
        for i in range(0, len(pairs), self.max_items_per_line):
            lines.append(" ".join(pairs[i : i + self.max_items_per_line]))

        print(f"{header} {lines[0]}")
        indent = " " * len(header)
        for line in lines[1:]:
            print(f"{indent} {line}")


class RichLiveLogger:
    """
    Live-updating console logger using rich.Live with a sparkline.
    """

    def __init__(
        self,
        *,
        keys: Sequence[str] | None = None,
        spark_key: str = "avg_total_reward",
        history: int = 100,
        spark_window: int = 50,
        precision: int = 4,
        console: Console | None = None,
        live: Live | None = None,
    ) -> None:
        self.keys = list(keys) if keys is not None else None
        self.spark_key = spark_key
        self.history = history
        self.spark_window = spark_window
        self.precision = precision
        self.console = console or Console()
        self.live = live or Live(console=self.console, refresh_per_second=4, transient=False)
        self.history_vals: list[float] = []
        self._last_eval_step: int | None = None
        self._own_live = live is None
        self._started = False

    def __enter__(self) -> "RichLiveLogger":
        if self._own_live:
            self.live.__enter__()
            self._started = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._own_live:
            self.live.__exit__(exc_type, exc, tb)

    def _fmt_val(self, v: Any) -> str:
        if isinstance(v, float):
            return f"{v:.{self.precision}f}"
        if isinstance(v, int):
            return str(v)
        return str(v)

    def _sparkline(self, max_len: int | None = None) -> tuple[str, float, float]:
        if not self.history_vals:
            return "", 0.0, 0.0
        vals = self.history_vals[-self.history :]
        # Always keep a moving window of the most recent spark_window points
        if self.spark_window and self.spark_window > 0:
            vals = vals[-self.spark_window :]
        if max_len is not None and max_len > 0:
            vals = vals[-max_len:]
        lo, hi = min(vals), max(vals)
        if hi == lo:
            return "▁" * len(vals), lo, hi
        blocks = "▁▂▃▄▅▆▇█"
        scaled = []
        for v in vals:
            idx = int((v - lo) / (hi - lo) * (len(blocks) - 1))
            scaled.append(blocks[idx])
        return "".join(scaled), lo, hi

    def _render(self, step: int, stats: Dict[str, float]):

        def _make_table(items):
            table = Table(show_header=False, box=None, padding=(0, 1))
            table.add_column("metric", style="cyan", no_wrap=True)
            table.add_column("value", style="magenta", overflow="fold")
            for name, value in items:
                table.add_row(name, self._fmt_val(value))
            return table

        if self.keys is not None:
            allowed_keys = set(self.keys)
        else:
            allowed_keys = None

        train_items = []
        eval_items = []
        for k, v in sorted(stats.items()):
            if k in {"phase", "train_step", "eval_step"}:
                continue
            if allowed_keys is not None and k not in allowed_keys:
                continue
            if k.startswith("eval_"):
                eval_items.append((k.replace("eval_", "", 1), v))
            else:
                train_items.append((k, v))

        train_table = _make_table(train_items) if train_items else Text("no train stats")
        eval_table = _make_table(eval_items) if eval_items else Text("no eval stats")

        train_title = "train"
        train_step = stats.get("train_step")
        if train_step is None:
            train_step = step
        try:
            train_title = f"train (@ step {int(train_step)})"
        except Exception:
            pass

        eval_title = "eval"
        if self._last_eval_step is not None:
            eval_title = f"eval (@ step {self._last_eval_step})"

        # Try to size the sparkline to the available column width to avoid ellipsis
        col_width = max(10, (self.console.width or 80) // 3 - 6)
        spark, lo, hi = self._sparkline(max_len=col_width)
        if spark:
            spark_panel = Panel(
                Text(spark, style="magenta", overflow="crop", no_wrap=True),
                title=f"{self.spark_key}",
                subtitle=f"step {step} | {lo:.{self.precision}f} – {hi:.{self.precision}f}",
                padding=(0, 1),
            )
        else:
            spark_panel = Panel(
                Text("-", style="magenta"),
                title=self.spark_key,
                subtitle=f"step {step}",
                padding=(0, 1),
            )

        root = Layout()
        top = Layout(name="top", ratio=3)
        top.split_row(
            Layout(
                Panel(train_table, title=train_title, padding=(0, 0), expand=True),
                name="train",
                ratio=1,
            ),
            Layout(
                Panel(eval_table, title=eval_title, padding=(0, 0), expand=True),
                name="eval",
                ratio=1,
            ),
        )
        bottom = Layout(
            Panel(spark_panel, padding=(0, 0), expand=True),
            name="spark",
            ratio=1,
        )
        root.split_column(top, bottom)
        return root

    def log(self, step: int, stats: Dict[str, float]) -> None:
        if "eval_step" in stats:
            try:
                self._last_eval_step = int(stats["eval_step"])
            except Exception:
                self._last_eval_step = step

        spark_val = stats.get(self.spark_key)
        if isinstance(spark_val, (int, float)):
            self.history_vals.append(float(spark_val))
            if len(self.history_vals) > self.history:
                self.history_vals = self.history_vals[-self.history :]

        if self._own_live and not self._started:
            self.live.start()
            self._started = True

        table = self._render(step, stats)
        self.live.update(table)
class WandbLogger:
    """
    Minimal Weights & Biases logger. Lazily imports wandb.
    """

    def __init__(self, *, run: Any | None = None, init_kwargs: Dict[str, Any] | None = None) -> None:
        try:
            import wandb
        except ImportError as exc:  # pragma: no cover - soft dependency
            raise ImportError("WandbLogger requires the 'wandb' package installed.") from exc

        self._wandb = wandb
        self._run = run or wandb.init(**(init_kwargs or {}))

    def log(self, step: int, stats: Dict[str, float]) -> None:
        self._wandb.log(stats, step=step)
