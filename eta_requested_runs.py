#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
import time
from pathlib import Path


DEFAULT_REPO = Path("/home/sunayana/Documents/Concept_LoRA")


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(errors="ignore")


def _pick_runner_log(repo: Path) -> Path:
    log_dir = repo / "unified_logs_requested"
    preferred = [
        log_dir / "tmux_align_both.out",
        log_dir / "tmux_runner.out",
        log_dir / "runner.out",
    ]
    existing = [p for p in preferred if p.exists()]
    if existing:
        return max(existing, key=lambda p: p.stat().st_mtime)

    all_out = sorted(log_dir.glob("*.out"), key=lambda p: p.stat().st_mtime, reverse=True)
    if all_out:
        return all_out[0]
    return log_dir / "runner.out"


def _parse_run_plan(run_script: Path) -> tuple[int, int]:
    text = _read_text(run_script)
    if not text:
        return 104, 1600

    task_entries = len(re.findall(r'^\s*".*\|.*\|.*\|.*\|.*"\s*$', text, flags=re.M))

    methods_match = re.search(r"for METHOD in (.+?); do", text)
    method_count = 2
    if methods_match:
        method_count = len([m for m in methods_match.group(1).split() if m.strip()])

    shots = 16
    n_iters = 100
    m = re.search(r"^\s*SHOTS=(\d+)\s*$", text, flags=re.M)
    if m:
        shots = int(m.group(1))
    m = re.search(r"^\s*N_ITERS=(\d+)\s*$", text, flags=re.M)
    if m:
        n_iters = int(m.group(1))

    total_runs = task_entries * method_count if task_entries else 104
    total_iters = shots * n_iters
    return total_runs, total_iters


def _parse_start_time(runner_text: str) -> dt.datetime | None:
    for pat in (r"^Run started:\s+(.+)$", r"^ALIGN continuation run started:\s+(.+)$"):
        m = re.search(pat, runner_text, flags=re.M)
        if not m:
            continue
        raw = m.group(1).strip()
        try:
            return dt.datetime.strptime(raw, "%a %b %d %I:%M:%S %p %Z %Y")
        except ValueError:
            continue
    return None


def _parse_summary_counts(summary_path: Path) -> dict[str, int]:
    counts = {"OK": 0, "FAIL": 0, "SKIP": 0}
    if not summary_path.exists():
        return counts

    with summary_path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            s = (row.get("status") or "").strip().upper()
            if s in counts:
                counts[s] += 1
    return counts


def _parse_runner_progress(runner_text: str) -> tuple[int, str | None, int]:
    runs_started = len(re.findall(r"^RUN\s+(\S+)", runner_text, flags=re.M))
    current_run = None
    run_matches = re.findall(r"^RUN\s+(\S+)", runner_text, flags=re.M)
    if run_matches:
        current_run = run_matches[-1]

    iter_val = 0
    for m in re.finditer(r"(?:CLIP|DINO|ALIGN)/(?:lora|dora) iter (\d+)", runner_text):
        iter_val = int(m.group(1))

    return runs_started, current_run, iter_val


def _fmt_td(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def render(repo: Path) -> str:
    run_script = repo / "run_requested_lora_dora_matrix.sh"
    runner_out = _pick_runner_log(repo)
    legacy_runner_out = repo / "unified_logs_requested" / "runner.out"
    summary_tsv = repo / "results" / "requested_lora_dora_summary.tsv"

    total_runs, total_iters = _parse_run_plan(run_script)
    runner_text = _read_text(runner_out)
    start_time = _parse_start_time(runner_text)
    if start_time is None and legacy_runner_out.exists():
        start_time = _parse_start_time(_read_text(legacy_runner_out))
    if start_time is None and runner_out.exists():
        start_time = dt.datetime.fromtimestamp(runner_out.stat().st_mtime)
    started_runs, current_run, iter_val = _parse_runner_progress(runner_text)
    counts = _parse_summary_counts(summary_tsv)

    done_runs = counts["OK"] + counts["FAIL"] + counts["SKIP"]
    current_frac = 0.0
    if started_runs > done_runs and total_iters > 0:
        current_frac = min(max(iter_val / total_iters, 0.0), 1.0)

    equivalent_done = done_runs + current_frac
    equivalent_done = min(equivalent_done, float(total_runs))
    remaining_equiv = max(0.0, total_runs - equivalent_done)

    now = dt.datetime.now()
    elapsed_s = None
    avg_run_s = None
    eta_s = None
    eta_time = None
    if start_time is not None:
        elapsed_s = max(0.0, (now - start_time).total_seconds())
        if equivalent_done > 0:
            avg_run_s = elapsed_s / equivalent_done
            eta_s = remaining_equiv * avg_run_s
            eta_time = now + dt.timedelta(seconds=eta_s)

    pct = (equivalent_done / total_runs * 100.0) if total_runs else 0.0
    lines = []
    lines.append(f"Now:                {now.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Runner log:         {runner_out.name}")
    lines.append(f"Total planned runs: {total_runs}")
    lines.append(f"Completed runs:     {done_runs} (OK={counts['OK']}, FAIL={counts['FAIL']}, SKIP={counts['SKIP']})")
    lines.append(f"Active run:         {current_run or 'N/A'}")
    lines.append(f"Active iter:        {iter_val}/{total_iters}")
    lines.append(f"Overall progress:   {pct:.2f}% (equiv {equivalent_done:.2f}/{total_runs})")
    if elapsed_s is not None:
        lines.append(f"Elapsed:            {_fmt_td(elapsed_s)}")
    if avg_run_s is not None:
        lines.append(f"Avg per run:        {_fmt_td(avg_run_s)}")
    if eta_s is not None and eta_time is not None:
        lines.append(f"ETA remaining:      {_fmt_td(eta_s)}")
        lines.append(f"ETA finish:         {eta_time.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        lines.append("ETA:                collecting data...")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Live ETA for requested LoRA/DoRA run sweep")
    parser.add_argument("--repo", default=str(DEFAULT_REPO), help="Concept_LoRA repo path")
    parser.add_argument("--watch", type=int, default=0, help="Refresh interval in seconds")
    args = parser.parse_args()

    repo = Path(args.repo).expanduser().resolve()

    if args.watch and args.watch > 0:
        while True:
            print("\033[2J\033[H", end="")
            print(render(repo))
            time.sleep(args.watch)
    else:
        print(render(repo))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
