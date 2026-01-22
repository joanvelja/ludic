#!/usr/bin/env python3
"""Analyze and visualize RM experiment results.

This script parses experiment logs and generates:
1. Summary tables comparing all experiments
2. Training curves (loss, accuracy, margin over steps)
3. Comparative plots across configurations
4. Identifies best checkpoints and trends

Usage:
    # Analyze all experiments in logs directory
    uv run python examples/reward_model/analyze_rm_experiments.py --logs-dir logs

    # Generate plots (requires matplotlib)
    uv run python examples/reward_model/analyze_rm_experiments.py --logs-dir logs --plot

    # Export to CSV for further analysis
    uv run python examples/reward_model/analyze_rm_experiments.py --logs-dir logs --csv results.csv
"""

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class EvalPoint:
    """Single evaluation checkpoint."""
    step: int
    loss: float
    accuracy: float
    margin: float
    ece: float
    auroc: float


@dataclass
class TrainPoint:
    """Single training step metrics."""
    step: int
    loss: float
    accuracy: float
    margin: float
    grad_norm: float
    chosen_score: float
    rejected_score: float


@dataclass
class DiagnosticCriteria:
    """Pass/fail criteria for diagnostic experiments."""
    name: str
    min_acc: Optional[float] = None
    max_acc: Optional[float] = None
    expected_loss: Optional[float] = None
    loss_tolerance: float = 0.1
    baseline_comparison: bool = False  # If True, compare to C1 baseline


# V5 Diagnostic pass/fail criteria
DIAGNOSTIC_CRITERIA = {
    "D1": DiagnosticCriteria("label_shuffle", min_acc=48.0, max_acc=52.0, expected_loss=0.693),
    "D2": DiagnosticCriteria("prompt_only", min_acc=48.0, max_acc=52.0),
    "D3": DiagnosticCriteria("random_negatives", min_acc=80.0, max_acc=95.0),
    "D4": DiagnosticCriteria("length_matched", baseline_comparison=True),  # Within 3% of C1
    "C1": DiagnosticCriteria("baseline", min_acc=73.0, max_acc=75.0),
}


@dataclass
class ExperimentResult:
    """Full results for one experiment."""
    exp_id: int
    name: str
    config: str
    train_points: List[TrainPoint] = field(default_factory=list)
    eval_points: List[EvalPoint] = field(default_factory=list)
    best_step: Optional[int] = None
    best_accuracy: Optional[float] = None
    final_accuracy: Optional[float] = None
    chat_template_verified: bool = False
    error: Optional[str] = None
    diagnostic_mode: Optional[str] = None
    delta_std: Optional[float] = None
    reward_abs_max: Optional[float] = None


def parse_log_file(log_path: Path) -> ExperimentResult:
    """Parse a single experiment log file."""
    # Extract experiment ID from filename: slurm_JOBID_EXPID_rm_v3.log or slurm_JOBID_EXPID_rm_v5.log
    match = re.search(r'slurm_\d+_(\d+)_rm', log_path.name)
    exp_id = int(match.group(1)) if match else -1

    result = ExperimentResult(
        exp_id=exp_id,
        name=f"Exp {exp_id}",
        config="Unknown",
    )

    content = log_path.read_text()
    lines = content.split('\n')

    # Extract experiment config name (supports V3, V4, V5 formats)
    for line in lines[:100]:
        # V5 format: "Running: D1 - Label shuffle (50% flip)"
        v5_match = re.search(r'Running: ([\w-]+)', line)
        if v5_match:
            result.name = v5_match.group(1)
            result.config = line.strip()
            break
        # V3/V4 format
        config_match = re.match(r'^([A-G]\d+:.+)$', line)
        if config_match:
            result.config = config_match.group(1).strip()
            result.name = result.config.split(':')[0]
            break

    # Check for diagnostic mode
    diag_match = re.search(r'DIAGNOSTIC MODE: (\w+)', content)
    if diag_match:
        result.diagnostic_mode = diag_match.group(1)

    # Check for chat template verification
    result.chat_template_verified = 'chat_template=True' in content

    # Check for errors
    if 'Error' in content or 'Exception' in content:
        error_match = re.search(r'(Error|Exception).*', content)
        if error_match:
            result.error = error_match.group(0)[:100]

    # Parse eval points
    eval_pattern = r'Step (\d+) Eval \| Loss: ([\d.]+) \| Acc: ([\d.]+)% \| Margin: ([-\d.]+) \| ECE: ([\d.]+) \| AUROC: ([\d.]+)'
    for match in re.finditer(eval_pattern, content):
        result.eval_points.append(EvalPoint(
            step=int(match.group(1)),
            loss=float(match.group(2)),
            accuracy=float(match.group(3)),
            margin=float(match.group(4)),
            ece=float(match.group(5)),
            auroc=float(match.group(6)),
        ))

    # Parse train points
    train_pattern = r'step=(\d+).*?train/accuracy=([\d.]+).*?train/loss=([\d.]+).*?train/margin=([-\d.]+).*?train/.*?grad_norm=([\d.]+).*?chosen_score=([-\d.]+).*?rejected_score=([-\d.]+)'
    for match in re.finditer(train_pattern, content, re.DOTALL):
        result.train_points.append(TrainPoint(
            step=int(match.group(1)),
            accuracy=float(match.group(2)),
            loss=float(match.group(3)),
            margin=float(match.group(4)),
            grad_norm=float(match.group(5)),
            chosen_score=float(match.group(6)),
            rejected_score=float(match.group(7)),
        ))

    # Parse best model info
    best_match = re.search(r'Best model: step (\d+) with eval acc ([\d.]+)%', content)
    if best_match:
        result.best_step = int(best_match.group(1))
        result.best_accuracy = float(best_match.group(2))

    # Get final accuracy
    if result.eval_points:
        result.final_accuracy = result.eval_points[-1].accuracy

    # Parse additional V5 metrics from RM_DIAG JSON logs
    diag_pattern = r'RM_DIAG ({.*})'
    diag_matches = list(re.finditer(diag_pattern, content))
    if diag_matches:
        try:
            # Get the last diagnostic log entry
            last_diag = json.loads(diag_matches[-1].group(1))
            if "delta_stats" in last_diag:
                result.delta_std = last_diag["delta_stats"].get("std")
            if "reward_stats" in last_diag:
                result.reward_abs_max = last_diag["reward_stats"].get("abs_max")
        except (json.JSONDecodeError, KeyError):
            pass

    return result


def check_diagnostic_pass(result: ExperimentResult, baseline_acc: Optional[float] = None) -> Tuple[bool, str]:
    """Check if a diagnostic experiment passes its criteria.

    Returns:
        (passed, reason) tuple where passed is bool and reason explains the result.
    """
    # Determine diagnostic type from name (D1, D2, D3, D4, C1)
    diag_type = None
    for dtype in DIAGNOSTIC_CRITERIA:
        if result.name.startswith(dtype):
            diag_type = dtype
            break

    if diag_type is None:
        return True, "Not a diagnostic experiment"

    criteria = DIAGNOSTIC_CRITERIA[diag_type]
    acc = result.best_accuracy

    if acc is None:
        return False, "No accuracy data"

    # Check min/max accuracy bounds
    if criteria.min_acc is not None and acc < criteria.min_acc:
        return False, f"Acc {acc:.1f}% < min {criteria.min_acc}%"

    if criteria.max_acc is not None and acc > criteria.max_acc:
        return False, f"Acc {acc:.1f}% > max {criteria.max_acc}%"

    # Check expected loss (for label shuffle)
    if criteria.expected_loss is not None and result.eval_points:
        final_loss = result.eval_points[-1].loss
        if abs(final_loss - criteria.expected_loss) > criteria.loss_tolerance:
            return False, f"Loss {final_loss:.3f} != expected {criteria.expected_loss:.3f}"

    # Check baseline comparison (for length_matched)
    if criteria.baseline_comparison and baseline_acc is not None:
        delta = abs(acc - baseline_acc)
        if delta > 3.0:  # More than 3% difference
            return False, f"Acc {acc:.1f}% differs from baseline {baseline_acc:.1f}% by {delta:.1f}%"

    return True, f"PASS: {acc:.1f}%"


def load_all_experiments(logs_dir: Path, pattern: str = "slurm_*_rm_v3.log") -> List[ExperimentResult]:
    """Load all experiment results from logs directory."""
    results = []
    for log_path in sorted(logs_dir.glob(pattern)):
        try:
            result = parse_log_file(log_path)
            results.append(result)
        except Exception as e:
            print(f"Warning: Failed to parse {log_path}: {e}")
    return sorted(results, key=lambda r: r.exp_id)


def print_summary_table(results: List[ExperimentResult], show_diagnostics: bool = False):
    """Print a formatted summary table."""
    # Detect version from first result
    version = "V5" if any(r.name.startswith(("D1", "D2", "D3", "D4", "C1", "M1", "S1", "S2", "S3", "A1", "A2")) for r in results) else "V3/V4"

    print("\n" + "=" * 120)
    print(f"RM {version} EXPERIMENT SUMMARY")
    print("=" * 120)

    # Find baseline accuracy for D4 comparison
    baseline_acc = None
    for r in results:
        if r.name.startswith("C1"):
            baseline_acc = r.best_accuracy
            break

    if show_diagnostics:
        # V5 diagnostic-aware header
        print(f"{'Exp':<20} {'Config':<35} {'Best Acc':>10} {'Final Acc':>10} {'Best Step':>10} {'Status':>15}")
        print("-" * 120)

        # Sort by experiment ID for V5 (preserves phase order)
        sorted_results = sorted(results, key=lambda r: r.exp_id)

        for r in sorted_results:
            best_acc = f"{r.best_accuracy:.2f}%" if r.best_accuracy else "N/A"
            final_acc = f"{r.final_accuracy:.2f}%" if r.final_accuracy else "N/A"
            best_step = str(r.best_step) if r.best_step else "N/A"
            config = r.config[:33] + ".." if len(r.config) > 35 else r.config

            # Check diagnostic pass/fail
            passed, reason = check_diagnostic_pass(r, baseline_acc)
            status = reason if len(reason) <= 15 else reason[:12] + "..."

            print(f"{r.name:<20} {config:<35} {best_acc:>10} {final_acc:>10} {best_step:>10} {status:>15}")
    else:
        # Original V3/V4 format
        print(f"{'Exp':<5} {'Config':<45} {'Best Acc':>10} {'Final Acc':>10} {'Best Step':>10} {'Chat':>6}")
        print("-" * 100)

        # Sort by best accuracy descending
        sorted_results = sorted(results, key=lambda r: r.best_accuracy or 0, reverse=True)

        for r in sorted_results:
            chat = "Yes" if r.chat_template_verified else "No"
            best_acc = f"{r.best_accuracy:.2f}%" if r.best_accuracy else "N/A"
            final_acc = f"{r.final_accuracy:.2f}%" if r.final_accuracy else "N/A"
            best_step = str(r.best_step) if r.best_step else "N/A"
            config = r.config[:43] + ".." if len(r.config) > 45 else r.config

            print(f"{r.name:<5} {config:<45} {best_acc:>10} {final_acc:>10} {best_step:>10} {chat:>6}")

    print("=" * 120)


def print_trajectory_analysis(results: List[ExperimentResult]):
    """Analyze and print trajectory trends for each experiment."""
    print("\n" + "=" * 100)
    print("TRAJECTORY ANALYSIS")
    print("=" * 100)

    for r in results:
        if not r.eval_points:
            continue

        print(f"\n{r.name}: {r.config}")
        print("-" * 60)

        # Check trend
        accs = [p.accuracy for p in r.eval_points]
        if len(accs) >= 3:
            early_avg = sum(accs[:3]) / 3
            late_avg = sum(accs[-3:]) / 3
            mid_idx = len(accs) // 2
            peak_idx = accs.index(max(accs))
            peak_step = r.eval_points[peak_idx].step

            trend = "improving" if late_avg > early_avg else "degrading" if late_avg < early_avg - 1 else "stable"

            print(f"  Trend: {trend}")
            print(f"  Peak: {max(accs):.2f}% at step {peak_step}")
            print(f"  Early avg (first 3): {early_avg:.2f}%")
            print(f"  Late avg (last 3): {late_avg:.2f}%")

            # Identify overfitting
            if peak_idx < len(accs) - 2 and max(accs) - accs[-1] > 2:
                print(f"  WARNING: Possible overfitting (peak at step {peak_step}, final {accs[-1]:.2f}%)")

        # Show trajectory
        print("  Steps: ", end="")
        for p in r.eval_points:
            print(f"{p.step}({p.accuracy:.1f}%) ", end="")
        print()


def print_v5_diagnostic_summary(results: List[ExperimentResult]):
    """Print V5 diagnostic pass/fail summary."""
    print("\n" + "=" * 80)
    print("V5 DIAGNOSTIC PHASE SUMMARY")
    print("=" * 80)

    # Find baseline
    baseline_acc = None
    for r in results:
        if r.name.startswith("C1"):
            baseline_acc = r.best_accuracy
            break

    # Categorize results by phase
    phase1 = [r for r in results if r.name.startswith(("D1", "D2", "D3", "D4", "C1"))]
    phase2 = [r for r in results if r.name.startswith(("M1", "S1"))]
    phase3 = [r for r in results if r.name.startswith(("S2", "S3"))]
    phase4 = [r for r in results if r.name.startswith(("A1", "A2"))]

    def print_phase(name: str, exps: List[ExperimentResult]):
        if not exps:
            return
        print(f"\n{name}:")
        print("-" * 60)
        all_passed = True
        for r in sorted(exps, key=lambda x: x.exp_id):
            passed, reason = check_diagnostic_pass(r, baseline_acc)
            status = "[PASS]" if passed else "[FAIL]"
            all_passed = all_passed and passed
            acc = f"{r.best_accuracy:.2f}%" if r.best_accuracy else "N/A"
            print(f"  {status} {r.name:<20} Acc: {acc:<10} {reason}")

        if name == "Phase 1: Diagnostics":
            if all_passed:
                print("\n  >>> ALL DIAGNOSTICS PASSED - Safe to proceed to Phase 2 <<<")
            else:
                print("\n  >>> SOME DIAGNOSTICS FAILED - Debug pipeline before continuing! <<<")

    print_phase("Phase 1: Diagnostics", phase1)
    print_phase("Phase 2: Regularization", phase2)
    print_phase("Phase 3: Temperature & Early Stopping", phase3)
    print_phase("Phase 4: Ablations", phase4)

    # Print additional V5 metrics if available
    print("\n" + "-" * 60)
    print("Additional V5 Metrics:")
    for r in sorted(results, key=lambda x: x.exp_id):
        metrics = []
        if r.delta_std is not None:
            metrics.append(f"delta_std={r.delta_std:.3f}")
        if r.reward_abs_max is not None:
            metrics.append(f"abs_max={r.reward_abs_max:.2f}")
        if metrics:
            print(f"  {r.name:<20} {', '.join(metrics)}")


def print_group_comparison(results: List[ExperimentResult]):
    """Compare experiments by group (A, B, C, etc.)."""
    print("\n" + "=" * 100)
    print("GROUP COMPARISON")
    print("=" * 100)

    groups: Dict[str, List[ExperimentResult]] = defaultdict(list)
    for r in results:
        group = r.name[0] if r.name and r.name[0].isalpha() else "?"
        groups[group].append(r)

    for group in sorted(groups.keys()):
        exps = groups[group]
        print(f"\nGroup {group}:")
        print("-" * 40)

        for r in sorted(exps, key=lambda x: x.best_accuracy or 0, reverse=True):
            best = f"{r.best_accuracy:.2f}%" if r.best_accuracy else "N/A"
            print(f"  {r.name}: {best} - {r.config[:50]}")

        # Group summary
        valid_accs = [r.best_accuracy for r in exps if r.best_accuracy]
        if valid_accs:
            print(f"  -> Best in group: {max(valid_accs):.2f}%")
            print(f"  -> Group avg: {sum(valid_accs)/len(valid_accs):.2f}%")


def generate_plots(results: List[ExperimentResult], output_dir: Path):
    """Generate matplotlib plots for experiment analysis."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
    except ImportError:
        print("matplotlib not available. Install with: pip install matplotlib")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Accuracy trajectories for all experiments
    fig, ax = plt.subplots(figsize=(14, 8))

    for r in results:
        if r.eval_points and r.best_accuracy and r.best_accuracy > 55:  # Only plot learning experiments
            steps = [p.step for p in r.eval_points]
            accs = [p.accuracy for p in r.eval_points]
            ax.plot(steps, accs, label=f"{r.name} ({r.best_accuracy:.1f}%)", marker='o', markersize=3)

    ax.set_xlabel('Step')
    ax.set_ylabel('Eval Accuracy (%)')
    ax.set_title('RM V3 Experiments - Accuracy Trajectories')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='Random chance')
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_trajectories.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'accuracy_trajectories.png'}")

    # Plot 2: Loss trajectories
    fig, ax = plt.subplots(figsize=(14, 8))

    for r in results:
        if r.eval_points and r.best_accuracy and r.best_accuracy > 55:
            steps = [p.step for p in r.eval_points]
            losses = [p.loss for p in r.eval_points]
            ax.plot(steps, losses, label=f"{r.name}", marker='o', markersize=3)

    ax.set_xlabel('Step')
    ax.set_ylabel('Eval Loss')
    ax.set_title('RM V3 Experiments - Loss Trajectories')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.693, color='r', linestyle='--', alpha=0.5, label='ln(2) = random')
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_trajectories.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'loss_trajectories.png'}")

    # Plot 3: Bar chart of best accuracies
    fig, ax = plt.subplots(figsize=(12, 6))

    valid_results = [r for r in results if r.best_accuracy]
    valid_results = sorted(valid_results, key=lambda r: r.best_accuracy, reverse=True)

    names = [r.name for r in valid_results]
    accs = [r.best_accuracy for r in valid_results]
    colors = ['green' if a > 70 else 'blue' if a > 60 else 'orange' if a > 55 else 'red' for a in accs]

    bars = ax.bar(names, accs, color=colors)
    ax.axhline(y=50, color='r', linestyle='--', alpha=0.5)
    ax.axhline(y=67, color='g', linestyle='--', alpha=0.5, label='Baseline')
    ax.set_xlabel('Experiment')
    ax.set_ylabel('Best Accuracy (%)')
    ax.set_title('RM V3 Experiments - Best Accuracy Comparison')
    ax.set_ylim(45, 80)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'accuracy_comparison.png'}")

    # Plot 4: Margin trajectories
    fig, ax = plt.subplots(figsize=(14, 8))

    for r in results:
        if r.eval_points and r.best_accuracy and r.best_accuracy > 55:
            steps = [p.step for p in r.eval_points]
            margins = [p.margin for p in r.eval_points]
            ax.plot(steps, margins, label=f"{r.name}", marker='o', markersize=3)

    ax.set_xlabel('Step')
    ax.set_ylabel('Margin (chosen - rejected score)')
    ax.set_title('RM V3 Experiments - Margin Trajectories')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / 'margin_trajectories.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'margin_trajectories.png'}")

    print(f"\nAll plots saved to {output_dir}/")


def export_csv(results: List[ExperimentResult], output_path: Path):
    """Export results to CSV for external analysis."""
    import csv

    # Export summary
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['exp_id', 'name', 'config', 'best_accuracy', 'final_accuracy', 'best_step', 'chat_template', 'error'])
        for r in results:
            writer.writerow([
                r.exp_id, r.name, r.config, r.best_accuracy, r.final_accuracy,
                r.best_step, r.chat_template_verified, r.error or ''
            ])
    print(f"Saved summary: {output_path}")

    # Export detailed trajectories
    traj_path = output_path.with_name(output_path.stem + '_trajectories.csv')
    with open(traj_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['exp_id', 'name', 'step', 'eval_loss', 'eval_accuracy', 'eval_margin', 'eval_ece', 'eval_auroc'])
        for r in results:
            for p in r.eval_points:
                writer.writerow([r.exp_id, r.name, p.step, p.loss, p.accuracy, p.margin, p.ece, p.auroc])
    print(f"Saved trajectories: {traj_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze RM experiment results")
    parser.add_argument('--logs-dir', type=Path, default=Path('logs'), help='Directory containing log files')
    parser.add_argument('--pattern', default='slurm_*_rm_v3.log', help='Log file pattern (use slurm_*_rm_v5.log for V5)')
    parser.add_argument('--plot', action='store_true', help='Generate matplotlib plots')
    parser.add_argument('--plot-dir', type=Path, default=Path('plots'), help='Output directory for plots')
    parser.add_argument('--csv', type=Path, help='Export results to CSV')
    parser.add_argument('--json', type=Path, help='Export results to JSON')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed trajectory analysis')
    parser.add_argument('--diagnostics', '-d', action='store_true', help='Show V5 diagnostic pass/fail summary')

    args = parser.parse_args()

    print(f"Loading experiments from {args.logs_dir}...")
    results = load_all_experiments(args.logs_dir, args.pattern)

    if not results:
        print("No experiment logs found!")
        return

    print(f"Found {len(results)} experiments")

    # Detect if this is V5 (diagnostic experiments)
    is_v5 = any(r.name.startswith(("D1", "D2", "D3", "D4", "C1", "M1", "S1", "S2", "S3", "A1", "A2")) for r in results)

    # Always print summary
    print_summary_table(results, show_diagnostics=is_v5 or args.diagnostics)

    # For V5, show diagnostic summary
    if is_v5 or args.diagnostics:
        print_v5_diagnostic_summary(results)
    else:
        print_group_comparison(results)

    if args.verbose:
        print_trajectory_analysis(results)

    if args.plot:
        generate_plots(results, args.plot_dir)

    if args.csv:
        export_csv(results, args.csv)

    if args.json:
        # Export to JSON
        data = []
        for r in results:
            data.append({
                'exp_id': r.exp_id,
                'name': r.name,
                'config': r.config,
                'best_accuracy': r.best_accuracy,
                'final_accuracy': r.final_accuracy,
                'best_step': r.best_step,
                'chat_template_verified': r.chat_template_verified,
                'error': r.error,
                'diagnostic_mode': r.diagnostic_mode,
                'delta_std': r.delta_std,
                'reward_abs_max': r.reward_abs_max,
                'eval_points': [
                    {'step': p.step, 'loss': p.loss, 'accuracy': p.accuracy,
                     'margin': p.margin, 'ece': p.ece, 'auroc': p.auroc}
                    for p in r.eval_points
                ]
            })
        with open(args.json, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved JSON: {args.json}")


if __name__ == '__main__':
    main()
