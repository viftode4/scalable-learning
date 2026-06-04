"""Extract decision metrics from FederatedScope supplement logs.

The existing summarizer is optimized for final result tables. This script is
for improvement decisions while a run is still in progress: server curves,
client train/eval rows, fairness spread, and RoLoRA phase markers.
"""

from __future__ import annotations

import argparse
import ast
import csv
import re
import statistics
from collections.abc import Iterable
from pathlib import Path

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
DICT_RE = re.compile(r"\{.*\}")
PHASE_RE = re.compile(r"\[sls-rolora\]\s+RoLoRA round\s+(\d+):\s+train\s+([A-Za-z_]+)")
HEADER_RE = re.compile(r"^#\s+([^:]+):\s*(.*)$")


def strip_ansi(line: str) -> str:
    return ANSI_RE.sub("", line)


def parse_payload(line: str) -> dict | None:
    match = DICT_RE.search(strip_ansi(line))
    if not match:
        return None
    try:
        payload = ast.literal_eval(match.group(0))
    except (SyntaxError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def int_round(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_log(path: Path) -> dict[str, list[dict]]:
    rows: dict[str, list[dict]] = {
        "server": [],
        "client_eval": [],
        "client_train": [],
        "phase": [],
        "monitor": [],
        "agg_monitor": [],
        "meta": [],
    }
    for lineno, raw_line in enumerate(path.read_text(errors="replace").splitlines(), start=1):
        line = strip_ansi(raw_line)
        header_match = HEADER_RE.match(line)
        if header_match:
            rows["meta"].append(
                {
                    "line": lineno,
                    "key": header_match.group(1).strip(),
                    "value": header_match.group(2).strip(),
                    "log": str(path),
                }
            )
        phase_match = PHASE_RE.search(line)
        if phase_match:
            rows["phase"].append(
                {
                    "line": lineno,
                    "round": int(phase_match.group(1)),
                    "phase": phase_match.group(2),
                    "log": str(path),
                }
            )

        payload = parse_payload(line)
        if payload is None:
            continue
        if "[sls-monitor]" in line:
            row = {"line": lineno, "log": str(path)}
            row.update(payload)
            rows["monitor"].append(row)
            continue
        if "[sls-agg]" in line:
            row = {"line": lineno, "log": str(path)}
            row.update(payload)
            rows["agg_monitor"].append(row)
            continue
        role = str(payload.get("Role", ""))
        round_id = int_round(payload.get("Round"))
        if round_id is None:
            continue

        if role.startswith("Server") and "Results_weighted_avg" in payload:
            weighted = payload.get("Results_weighted_avg", {})
            fairness = payload.get("Results_fairness", {})
            row = {
                "line": lineno,
                "round": round_id,
                "test_acc": weighted.get("test_acc"),
                "val_acc": weighted.get("val_acc"),
                "test_loss": weighted.get("test_loss"),
                "val_loss": weighted.get("val_loss"),
                "test_acc_std": fairness.get("test_acc_std"),
                "test_acc_min": fairness.get("test_acc_min"),
                "test_acc_max": fairness.get("test_acc_max"),
                "val_acc_std": fairness.get("val_acc_std"),
                "val_acc_min": fairness.get("val_acc_min"),
                "val_acc_max": fairness.get("val_acc_max"),
                "log": str(path),
            }
            rows["server"].append(row)
            continue

        if role.startswith("Client") and "Results_raw" in payload:
            metrics = payload["Results_raw"]
            client_id = role.replace("Client #", "")
            if "train_acc" in metrics:
                rows["client_train"].append(
                    {
                        "line": lineno,
                        "round": round_id,
                        "client": client_id,
                        "train_acc": metrics.get("train_acc"),
                        "train_loss": metrics.get("train_loss"),
                        "train_avg_loss": metrics.get("train_avg_loss"),
                        "train_total": metrics.get("train_total"),
                        "log": str(path),
                    }
                )
            if "test_acc" in metrics or "val_acc" in metrics:
                rows["client_eval"].append(
                    {
                        "line": lineno,
                        "round": round_id,
                        "client": client_id,
                        "test_acc": metrics.get("test_acc"),
                        "val_acc": metrics.get("val_acc"),
                        "test_avg_loss": metrics.get("test_avg_loss"),
                        "val_avg_loss": metrics.get("val_avg_loss"),
                        "test_total": metrics.get("test_total"),
                        "val_total": metrics.get("val_total"),
                        "log": str(path),
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def latest_by_round(rows: Iterable[dict]) -> list[dict]:
    by_round: dict[int, dict] = {}
    for row in rows:
        by_round[int(row["round"])] = row
    return [by_round[key] for key in sorted(by_round)]


def phase_by_round(rows: Iterable[dict]) -> dict[int, str]:
    """Collapse one-per-client phase prints into one phase label per round."""
    phases: dict[int, str] = {}
    for row in rows:
        phases[int(row["round"])] = str(row["phase"])
    return phases


def headers_from_rows(rows: Iterable[dict]) -> dict[str, str]:
    return {str(row["key"]): str(row["value"]) for row in rows}


def phase_from_headers(round_id: int, headers: dict[str, str]) -> str:
    """Infer phase for live logs whose stdout markers have not flushed yet."""
    mode = headers.get("mode", "")
    if mode != "rolora":
        return ""
    pattern = headers.get("sls_phase_pattern", "")
    if pattern and pattern != "default":
        compact = "".join(pattern.replace(",", " ").replace(";", " ").split()).upper()
        if compact and set(compact) <= {"A", "B"}:
            return compact[round_id % len(compact)]
    warmup = headers.get("sls_b_warmup_rounds", "")
    if warmup and warmup != "default":
        try:
            warmup_rounds = int(warmup)
        except ValueError:
            warmup_rounds = 0
        if warmup_rounds > 0:
            if round_id < warmup_rounds:
                return "B"
            return "A" if (round_id - warmup_rounds) % 2 == 0 else "B"
    return "B" if round_id % 2 == 0 else "A"


def numeric(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def aggregate_by_round(rows: Iterable[dict], metric: str) -> dict[int, dict]:
    grouped: dict[int, list[float]] = {}
    for row in rows:
        value = numeric(row.get(metric))
        if value is None:
            continue
        grouped.setdefault(int(row["round"]), []).append(value)

    aggregate: dict[int, dict] = {}
    for round_id, values in grouped.items():
        aggregate[round_id] = {
            f"{metric}_n": len(values),
            f"{metric}_mean": statistics.fmean(values),
            f"{metric}_std": statistics.pstdev(values) if len(values) > 1 else 0.0,
            f"{metric}_min": min(values),
            f"{metric}_max": max(values),
        }
    return aggregate


def rows_by_kind(rows: Iterable[dict], kind: str) -> list[dict]:
    return [row for row in rows if row.get("kind") == kind]


def metric_mean(aggregates: dict[int, dict], round_id: int, metric: str) -> object:
    return aggregates.get(round_id, {}).get(f"{metric}_mean")


def write_summary(path: Path, rows: dict[str, list[dict]]) -> None:
    server_rows = latest_by_round(rows["server"])
    phases = phase_by_round(rows["phase"])
    headers = headers_from_rows(rows["meta"])
    train_acc = aggregate_by_round(rows["client_train"], "train_acc")
    eval_test_acc = aggregate_by_round(rows["client_eval"], "test_acc")
    lines = ["# Supplement metrics summary", ""]
    lines.append(f"- server rows: {len(server_rows)}")
    lines.append(f"- client eval rows: {len(rows['client_eval'])}")
    lines.append(f"- client train rows: {len(rows['client_train'])}")
    lines.append(
        f"- phase markers: {len(rows['phase'])} raw, {len(phases)} unique rounds"
    )
    lines.append(f"- internal monitor rows: {len(rows['monitor'])}")
    lines.append(f"- aggregation monitor rows: {len(rows['agg_monitor'])}")
    lines.append("")
    if server_rows:
        lines.extend(
            [
                "| round | phase | test_acc | Δtest | val_acc | test_acc_std | val_acc_std | client_test_mean | train_acc_mean | train_n |",
                "|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        previous_test_acc: float | None = None
        for row in server_rows:
            round_id = int(row["round"])
            test_acc = numeric(row.get("test_acc"))
            delta = None if previous_test_acc is None or test_acc is None else test_acc - previous_test_acc
            if test_acc is not None:
                previous_test_acc = test_acc
            train = train_acc.get(round_id, {})
            eval_agg = eval_test_acc.get(round_id, {})
            lines.append(
                f"| {round_id} | {phases.get(round_id) or phase_from_headers(round_id, headers)} | "
                f"{fmt(row.get('test_acc'))} | {fmt_signed(delta)} | "
                f"{fmt(row.get('val_acc'))} | {fmt(row.get('test_acc_std'))} | "
                f"{fmt(row.get('val_acc_std'))} | "
                f"{fmt(eval_agg.get('test_acc_mean'))} | "
                f"{fmt(train.get('train_acc_mean'))} | "
                f"{train.get('train_acc_n', '')} |"
            )
    rounds_without_server = (set(phases) | set(train_acc) | set(eval_test_acc)) - {
        int(row["round"]) for row in server_rows
    }
    pending_rounds = sorted(rounds_without_server)
    if pending_rounds:
        lines.extend(["", "## Rounds seen without server aggregate yet", ""])
        lines.extend(
            [
                "| round | phase | client_test_mean | train_acc_mean | train_n |",
                "|---:|:---:|---:|---:|---:|",
            ]
        )
        for round_id in pending_rounds:
            train = train_acc.get(round_id, {})
            eval_agg = eval_test_acc.get(round_id, {})
            lines.append(
                f"| {round_id} | {phases.get(round_id) or phase_from_headers(round_id, headers)} | "
                f"{fmt(eval_agg.get('test_acc_mean'))} | "
                f"{fmt(train.get('train_acc_mean'))} | "
                f"{train.get('train_acc_n', '')} |"
            )

    fit_end_rows = rows_by_kind(rows["monitor"], "fit_end")
    client_post_rows = rows_by_kind(rows["monitor"], "client_post_train")
    drift_rows = rows_by_kind(rows["agg_monitor"], "aggregate_client_drift")
    result_rows = rows_by_kind(rows["agg_monitor"], "aggregate_result")
    monitor_source = client_post_rows or fit_end_rows
    if monitor_source or drift_rows or result_rows:
        product_norm = aggregate_by_round(monitor_source, "lora_product_norm")
        update_a = aggregate_by_round(monitor_source, "update_lora_A_norm")
        update_b = aggregate_by_round(monitor_source, "update_lora_B_norm")
        update_classifier = aggregate_by_round(monitor_source, "update_classifier_norm")
        drift_a = aggregate_by_round(drift_rows, "client_delta_lora_A_mean")
        drift_b = aggregate_by_round(drift_rows, "client_delta_lora_B_mean")
        agg_a = aggregate_by_round(result_rows, "aggregate_update_lora_A_norm")
        agg_b = aggregate_by_round(result_rows, "aggregate_update_lora_B_norm")
        monitor_rounds = sorted(
            set(product_norm)
            | set(update_a)
            | set(update_b)
            | set(update_classifier)
            | set(drift_a)
            | set(drift_b)
            | set(agg_a)
            | set(agg_b)
        )
        source_label = "client_post_train" if client_post_rows else "fit_end"
        lines.extend(
            [
                "",
                "## Internal monitor summary",
                "",
                f"Local-update source: `{source_label}` rows.",
                "",
                "| round | phase | local_n | mean_||BA|| | mean_ΔA | mean_ΔB | mean_Δclassifier | mean_client_drift_ΔA | mean_client_drift_ΔB | aggregate_ΔA | aggregate_ΔB |",
                "|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for round_id in monitor_rounds:
            local_n = (
                product_norm.get(round_id, {}).get("lora_product_norm_n")
                or update_a.get(round_id, {}).get("update_lora_A_norm_n")
                or update_b.get(round_id, {}).get("update_lora_B_norm_n")
                or ""
            )
            lines.append(
                f"| {round_id} | {phases.get(round_id) or phase_from_headers(round_id, headers)} | "
                f"{local_n} | "
                f"{fmt(metric_mean(product_norm, round_id, 'lora_product_norm'))} | "
                f"{fmt(metric_mean(update_a, round_id, 'update_lora_A_norm'))} | "
                f"{fmt(metric_mean(update_b, round_id, 'update_lora_B_norm'))} | "
                f"{fmt(metric_mean(update_classifier, round_id, 'update_classifier_norm'))} | "
                f"{fmt(metric_mean(drift_a, round_id, 'client_delta_lora_A_mean'))} | "
                f"{fmt(metric_mean(drift_b, round_id, 'client_delta_lora_B_mean'))} | "
                f"{fmt(metric_mean(agg_a, round_id, 'aggregate_update_lora_A_norm'))} | "
                f"{fmt(metric_mean(agg_b, round_id, 'aggregate_update_lora_B_norm'))} |"
            )
    path.write_text("\n".join(lines) + "\n")


def fmt(value: object) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def fmt_signed(value: object) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):+.6f}"
    except (TypeError, ValueError):
        return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    rows = parse_log(args.log)
    args.outdir.mkdir(parents=True, exist_ok=True)
    write_csv(args.outdir / "server_metrics.csv", latest_by_round(rows["server"]))
    write_csv(args.outdir / "client_eval_metrics.csv", rows["client_eval"])
    write_csv(args.outdir / "client_train_metrics.csv", rows["client_train"])
    write_csv(args.outdir / "phase_markers.csv", rows["phase"])
    write_csv(args.outdir / "internal_monitor.csv", rows["monitor"])
    write_csv(args.outdir / "aggregation_monitor.csv", rows["agg_monitor"])
    write_csv(args.outdir / "run_metadata.csv", rows["meta"])
    write_summary(args.outdir / "metrics_summary.md", rows)
    print(f"wrote metrics to {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
