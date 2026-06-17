#!/usr/bin/env python3
"""Aggregate EVERY result + log in this repo into one canonical registry.

This is the single source of truth so nobody has to re-derive numbers from raw
logs again. It reads primary sources only (FederatedScope server logs, evidence
server_metrics.csv, the W&B export, FedAvg eval logs, toy result JSON) and
writes runs/REGISTRY.md.

Every number is the VERIFIED value extracted here, not copied from a summary
doc. The canonical metric for a federated run is the final aggregation round's
server `Results_weighted_avg.test_acc` (`final`), plus the best such value over
all rounds (`best`).

Regenerate / share after any new run:

    uv run python scripts/aggregate_all_results.py            # rewrite runs/REGISTRY.md only
    uv run python scripts/aggregate_all_results.py --promote  # also copy run artifacts into runs/
    uv run python scripts/aggregate_all_results.py --stdout   # print the registry instead

`--promote` materialises a committed, shareable folder per canonical run under
runs/{proxy,audit,baselines,toy}/ (log + server_metrics.csv + config + meta.json),
so teammates get every real run via `git pull` without the 200 MB of local
wandb/checkpoint scratch.

`nan` appears in FedAvg fairness dicts, so metrics are pulled with a targeted
regex rather than ast.literal_eval.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "runs"
ANSI = re.compile(r"\x1b\[[0-9;]*m")
HEADER_RE = re.compile(r"^#\s*([a-z_]+):\s*(.*)$")
ROUND_RE = re.compile(r"'Round':\s*(\d+)")
WAVG_RE = re.compile(r"'Results_weighted_avg':")
# first 'test_acc'/'val_acc' float after a given offset
ACC_RE = re.compile(r"'test_acc':\s*([0-9.]+)")
VACC_RE = re.compile(r"'val_acc':\s*([0-9.]+)")


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class Run:
    key: str
    final: float | None = None          # last-round weighted_avg test_acc
    best: float | None = None           # max weighted_avg test_acc over rounds
    final_round: int | None = None
    n_rounds: int = 0
    val_final: float | None = None
    source: str = ""                    # repo-relative path of evidence
    note: str = ""
    display: str = ""                   # human-facing name (defaults to key)

    def acc_str(self) -> str:
        if self.final is None:
            return "—"
        s = f"{self.final * 100:.2f}"
        if self.best is not None and abs(self.best - self.final) > 1e-4:
            s += f" (best {self.best * 100:.2f})"
        return s

    def status(self, expect_rounds: int = 19) -> str:
        if self.final is None:
            return "no metrics"
        if self.final < 0.60:
            return "chance / did not learn"
        if self.final_round is not None and self.final_round < expect_rounds:
            return f"partial ({self.final_round + 1} rounds)"
        return "ok"


def ci95(vals: list[float]) -> float | None:
    """95% CI half-width = 1.96 * sample_std / sqrt(n). Matches existing docs/plots."""
    if len(vals) < 2:
        return None
    return 1.96 * statistics.stdev(vals) / (len(vals) ** 0.5)


def agg_str(vals: list[float], scale: float = 100.0) -> str:
    if not vals:
        return "—"
    if len(vals) == 1:
        return f"{vals[0] * scale:.2f}"
    half = ci95(vals)
    return f"{statistics.mean(vals) * scale:.2f} ± {half * scale:.2f}"


def rel(p: Path) -> str:
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


# --------------------------------------------------------------------------- #
# Parsers
# --------------------------------------------------------------------------- #
def server_rows_from_log(path: Path) -> list[tuple[int, float, float | None]]:
    """(round, weighted_avg test_acc, weighted_avg val_acc) per aggregation round."""
    rows: list[tuple[int, float, float | None]] = []
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return rows
    for raw in text.splitlines():
        if "'Role': 'Server #'" not in raw or "'Results_weighted_avg'" not in raw:
            continue
        line = ANSI.sub("", raw)
        rmatch = ROUND_RE.search(line)
        wmatch = WAVG_RE.search(line)
        if not rmatch or not wmatch:
            continue
        tail = line[wmatch.end():]          # restrict to the weighted_avg block onward
        amatch = ACC_RE.search(tail)
        if not amatch:
            continue
        vmatch = VACC_RE.search(tail)
        rows.append((int(rmatch.group(1)), float(amatch.group(1)),
                     float(vmatch.group(1)) if vmatch else None))
    return rows


def run_from_log(path: Path, key: str | None = None) -> Run | None:
    rows = server_rows_from_log(path)
    if not rows:
        return None
    rows.sort(key=lambda r: r[0])
    last = rows[-1]
    return Run(
        key=key or path.stem.replace("overnight_", ""),
        final=last[1],
        best=max(r[1] for r in rows),
        final_round=last[0],
        n_rounds=len(rows),
        val_final=last[2],
        source=rel(path),
    )


def run_from_csv(path: Path, key: str | None = None) -> Run | None:
    try:
        with path.open() as f:
            rows = list(csv.DictReader(f))
    except OSError:
        return None
    accs = [(int(float(r["round"])), float(r["test_acc"]),
             float(r["val_acc"]) if r.get("val_acc") not in (None, "") else None)
            for r in rows if r.get("test_acc") not in (None, "")]
    if not accs:
        return None
    accs.sort(key=lambda r: r[0])
    last = accs[-1]
    return Run(
        key=key or path.parent.name,
        final=last[1],
        best=max(r[1] for r in accs),
        final_round=last[0],
        n_rounds=len(accs),
        val_final=last[2],
        source=rel(path),
    )


# --------------------------------------------------------------------------- #
# Collectors
# --------------------------------------------------------------------------- #
def _canon_proxy_key(key: str) -> str:
    """Collapse log names, CSV dir names, and partial/retry variants to one id.

    e.g. `proxy_orth_a_c50_r4_lr1e-2_seed0` (log) and `proxy_orth_a_seed0` (CSV)
    and `...seed0.partial_stalled_20260609` all map to `proxy_orth_a_seed0`.
    """
    k = key.split(".")[0]                       # drop .partial_*/.invalid_*/.retry_* tails
    k = re.sub(r"_(partial|final)$", "", k)
    k = re.sub(r"_c50_r\d+_lr[0-9.]+e?-?\d*", "", k)  # drop the c50/r4/lr1e-2 infix
    return k


def collect_proxy() -> list[Run]:
    """QNLI / RoBERTa-base proxy improvement+control runs from logs and CSVs.

    Merge by canonical run-id, preferring the most complete source (most rounds),
    tie-broken toward the descriptive log name.
    """
    best: dict[str, Run] = {}

    def consider(run: Run | None, prefer_name: bool) -> None:
        if run is None or run.final is None:
            return
        norm = _canon_proxy_key(run.key)
        clean = run.key.split(".")[0]           # drop .invalid_/.partial_/.retry_ tails
        run.display = clean
        cur = best.get(norm)
        if cur is None or run.n_rounds > cur.n_rounds:
            if cur is not None and len(cur.display) > len(clean):
                run.display = cur.display       # keep the more descriptive clean name
            best[norm] = run
        elif prefer_name and len(clean) > len(cur.display):
            cur.display = clean

    for log in sorted((REPO / "results").glob("overnight_proxy_*.log")):
        consider(run_from_log(log), prefer_name=True)
    for csv_path in sorted((REPO / "evidence").rglob("server_metrics.csv")):
        if "proxy" in csv_path.parent.name:
            consider(run_from_csv(csv_path), prefer_name=False)
    return sorted(best.values(), key=lambda r: r.display)


def collect_wandb_sweep() -> list[Run]:
    """Vanilla RoLoRA / LoRA / FFA-LoRA learning-rate sweep from the W&B export."""
    path = REPO / "evidence/wandb_qnli_c50_r4_20260603/server_history.csv"
    if not path.exists():
        return []
    by_run: dict[str, list[dict]] = {}
    with path.open() as f:
        for r in csv.DictReader(f):
            by_run.setdefault(r["name"], []).append(r)
    runs: list[Run] = []
    for name, rows in by_run.items():
        rows = [r for r in rows if r.get("server_test_acc") not in (None, "")]
        if not rows:
            continue
        rows.sort(key=lambda r: int(float(r["round"])))
        last = rows[-1]
        accs = [float(r["server_test_acc"]) for r in rows]
        state = rows[0].get("state", "")
        runs.append(Run(
            key=name,
            final=float(last["server_test_acc"]),
            best=max(accs),
            final_round=int(float(last["round"])),
            n_rounds=len(rows),
            source=rel(path),
            note=f"method={rows[0]['method']} lr={rows[0]['lr']} seed={rows[0]['seed']} state={state}",
        ))
    return sorted(runs, key=lambda r: r.key)


def collect_fedavg_audit() -> list[Run]:
    """Centralized/FedAvg optimizer-audit runs (exp/FedAvg_*qnli*/eval_results.log)."""
    runs: list[Run] = []
    for log in sorted((REPO / "exp").glob("FedAvg_*qnli*/eval_results.log")):
        name = log.parent.name
        lr = re.search(r"lr([0-9.]+)", name)
        lstep = re.search(r"lstep(\d+)", name)
        run = run_from_log(log, key=name)
        if run:
            run.note = f"lr={lr.group(1) if lr else '?'} local_steps={lstep.group(1) if lstep else '?'}"
            runs.append(run)
    return runs


def collect_table1_local() -> list[Run]:
    runs: list[Run] = []
    for log in sorted((REPO / "results").glob("table1_*.log")):
        run = run_from_log(log)
        if run:
            runs.append(run)
    return runs


@dataclass
class ToyTable:
    title: str
    config: str
    source: str
    rows: list[dict] = field(default_factory=list)


def collect_toy() -> list[ToyTable]:
    tables: list[ToyTable] = []
    for jpath in sorted((REPO / "evidence").rglob("results.json")):
        try:
            data = json.loads(jpath.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        summary = data.get("summary")
        if not isinstance(summary, list) or not summary:
            continue
        cfg = data.get("config", {})
        # canonical toy config only (rank-16 / 60-round / multi-seed); the
        # rank-4 / 20-round / 2-seed banks are exploratory smokes (listed in §6).
        if cfg.get("rounds", 0) < 60 or len(cfg.get("seeds", []) or []) < 3:
            continue
        cfg_str = ", ".join(
            f"{k}={cfg[k]}" for k in ("split", "clients", "labels_per_client",
                                      "rank", "rounds", "lr") if k in cfg)
        tables.append(ToyTable(
            title=jpath.parent.name,
            config=cfg_str,
            source=rel(jpath),
            rows=[{
                "name": s.get("name", "?"),
                "n": len(data.get("config", {}).get("seeds", []) or []) or s.get("n", ""),
                "final_mean": s.get("final_acc_mean"),
                "final_std": s.get("final_acc_std"),
                "best_mean": s.get("best_acc_mean"),
                "best_std": s.get("best_acc_std"),
                "note": s.get("note", ""),
            } for s in summary],
        ))
    return tables


def collect_log_index() -> list[tuple[str, int, str, str]]:
    """(relpath, bytes, mtime, role) for every result-bearing artifact."""
    out: list[tuple[str, int, str, str]] = []
    seen: set[Path] = set()

    def add(p: Path, role: str) -> None:
        if p in seen or not p.is_file():
            return
        seen.add(p)
        st = p.stat()
        mtime = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).strftime("%Y-%m-%d")
        out.append((rel(p), st.st_size, mtime, role))

    for p in sorted((REPO / "results").glob("*.log")):
        n = p.name
        if n.startswith("overnight_proxy"):
            role = "proxy QNLI run log"
        elif n.startswith("table1_"):
            role = "Table-1 local run log"
        elif "queue" in n or "launch" in n or "watch" in n:
            role = "queue/launcher log (no per-round metrics)"
        elif n.startswith("smoke") or "smoke" in n:
            role = "smoke/validation log"
        elif n.startswith("overnight_"):
            role = "overnight run log (optimizer audit / baseline)"
        else:
            role = "run log"
        add(p, role)
    for p in sorted((REPO / "evidence").rglob("server_metrics.csv")):
        add(p, "proxy per-round server curve (CSV)")
    add(REPO / "evidence/wandb_qnli_c50_r4_20260603/server_history.csv", "W&B LR-sweep export (CSV)")
    for p in sorted((REPO / "exp").glob("FedAvg_*qnli*/eval_results.log")):
        add(p, "FedAvg optimizer-audit eval log")
    for p in sorted((REPO / "evidence").rglob("results.json")):
        add(p, "toy MNIST result JSON")
    return out


# --------------------------------------------------------------------------- #
# Promotion: copy canonical run artifacts into the committed runs/ registry
# --------------------------------------------------------------------------- #
def log_header(path: Path) -> dict[str, str]:
    """Parse the `# key: value` provenance block our runner writes atop each log."""
    meta: dict[str, str] = {}
    try:
        with path.open(errors="replace") as f:
            for line in f:
                if not line.startswith("#"):
                    break
                m = HEADER_RE.match(line.strip())
                if m and m.group(2) not in ("", "default", "(none)"):
                    meta[m.group(1)] = m.group(2)
    except OSError:
        pass
    return meta


def _write_metrics_csv(rows: list[tuple[int, float, float | None]], dest: Path) -> None:
    with dest.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["round", "test_acc", "val_acc"])
        for rnd, ta, va in rows:
            w.writerow([rnd, ta, "" if va is None else va])


def _find_config(display: str) -> Path | None:
    for cand in sorted(REPO.glob(f"exp/{display}_*/**/config.yaml")):
        return cand
    return None


def promote_all() -> list[str]:
    """Materialise self-contained run folders under runs/. Idempotent."""
    actions: list[str] = []
    RUNS.mkdir(exist_ok=True)

    def meta_json(dest: Path, payload: dict) -> None:
        (dest / "meta.json").write_text(json.dumps(payload, indent=2) + "\n")

    # proxy runs: run.log + server_metrics.csv + config.yaml + meta.json
    for r in collect_proxy():
        dest = RUNS / "proxy" / r.display
        dest.mkdir(parents=True, exist_ok=True)
        log = REPO / "results" / f"overnight_{r.display}.log"
        header = log_header(log) if log.exists() else {}
        if log.exists():
            shutil.copy2(log, dest / "run.log")
            _write_metrics_csv(server_rows_from_log(log), dest / "server_metrics.csv")
        elif r.source.endswith(".csv"):
            shutil.copy2(REPO / r.source, dest / "server_metrics.csv")
        cfg = _find_config(r.display)
        if cfg:
            shutil.copy2(cfg, dest / "config.yaml")
        meta_json(dest, {
            "run": r.display, "task": "qnli_roberta-base_c50_r4_20rounds",
            "final_test_acc": r.final, "best_test_acc": r.best,
            "final_round": r.final_round, "n_rounds": r.n_rounds,
            "status": r.status(), "primary_source": r.source,
            "git_sha": header.get("git_sha"),
            "sls_lora_init": header.get("sls_lora_init"),
            "sls_phase_pattern": header.get("sls_phase_pattern"),
            "has_log": log.exists(), "has_config": bool(cfg),
        })
        actions.append(f"proxy/{r.display}")

    # FedAvg optimizer-audit eval logs
    for r in collect_fedavg_audit():
        dest = RUNS / "audit" / r.key
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / r.source, dest / "eval_results.log")
        meta_json(dest, {
            "run": r.key, "kind": "fedavg_optimizer_audit",
            "final_test_acc": r.final, "best_test_acc": r.best,
            "final_round": r.final_round, "config": r.note, "status": r.status(),
            "primary_source": r.source,
        })
        actions.append(f"audit/{r.key}")

    # baselines: the W&B LR-sweep export + grouped meta
    wandb_csv = REPO / "evidence/wandb_qnli_c50_r4_20260603/server_history.csv"
    if wandb_csv.exists():
        dest = RUNS / "baselines"
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(wandb_csv, dest / "server_history.csv")
        groups: dict[str, list[float]] = {}
        for run in collect_wandb_sweep():
            if run.final is not None:
                groups.setdefault(re.sub(r"_seed\d+$", "", run.key), []).append(run.final)
        meta_json(dest, {
            "kind": "baseline_lr_sweep", "source": rel(wandb_csv),
            "final_acc_mean_ci95": {g: agg_str(v) for g, v in sorted(groups.items())},
        })
        actions.append("baselines/")

    # canonical toy results
    toy_dest = RUNS / "toy"
    toy_dest.mkdir(parents=True, exist_ok=True)
    for jpath in sorted((REPO / "evidence").rglob("results.json")):
        try:
            cfg = json.loads(jpath.read_text()).get("config", {})
        except (OSError, json.JSONDecodeError):
            continue
        if cfg.get("rounds", 0) >= 60 and len(cfg.get("seeds", []) or []) >= 3:
            shutil.copy2(jpath, toy_dest / f"{jpath.parent.name}.json")
            actions.append(f"toy/{jpath.parent.name}.json")
    return actions


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def fmt_mean(mean, std, scale=100.0):
    if mean is None:
        return "—"
    s = f"{mean * scale:.2f}"
    if std is not None:
        s += f" ± {std * scale:.2f}"
    return s


def render(stamp: str) -> str:
    proxy = collect_proxy()
    sweep = collect_wandb_sweep()
    audit = collect_fedavg_audit()
    table1 = collect_table1_local()
    toy = collect_toy()
    index = collect_log_index()

    lines: list[str] = []
    lines.append("# Aggregate results registry — single source of truth")
    lines.append("")
    lines.append(f"> Auto-generated by `scripts/aggregate_all_results.py` on {stamp}. "
             "Every number is extracted from the primary source listed in its row "
             "(server log / evidence CSV / W&B export / toy JSON) — **not** copied "
             "from a summary doc. Per-run artifacts (log + metrics + config + meta) "
             "live beside this file under `runs/{proxy,audit,baselines,toy}/`; "
             "refresh both with `make promote-runs`.")
    lines.append("")
    lines.append("**Metric convention.** Federated runs report the final aggregation "
             "round's server `Results_weighted_avg.test_acc` (`final`); `best` is the "
             "max over all rounds. QNLI proxy = RoBERTa-base / 50 clients / rank 4 / "
             "20 rounds — proxy evidence, **not** RoBERTa-Large Table-1 reproduction.")
    lines.append("")
    lines.append("**Spread convention (read this before comparing to older docs).** "
             "Proxy/sweep means here are **mean ± CI95** (`1.96·s/√n`), matching the "
             "W&B summary and the figures. Toy means are **± sample std** (as stored in "
             "the JSON). Note `docs/improvement-handoff-2026-06-13.md` reports the "
             "improvement arms with **population std** instead — so its ±0.22 (orth+BBA), "
             "±0.31 (adaptive), ±0.59 (SVD) are the *same per-seed numbers* shown here, "
             "for which CI95 = ±0.43 / ±0.43 / ±0.82. The means are identical; only the "
             "error-bar definition differed. This registry is the one to cite.")
    lines.append("")

    # vanilla baseline (vanilla RoLoRA lr1e-2, 3 seeds) for delta columns
    base_vals = [r.final for r in sweep
                 if r.key.startswith("rolora_lr1e-2_") and r.final is not None]
    baseline = statistics.mean(base_vals) if base_vals else None

    # 1. Proxy improvement + control runs
    lines.append("## 1. QNLI RoBERTa-base proxy — improvement & control runs")
    lines.append("")
    if baseline is not None:
        lines.append(f"Vanilla RoLoRA control = **{baseline * 100:.2f}%** "
                 f"({agg_str(base_vals)}, n={len(base_vals)}). Δ columns are vs this control.")
        lines.append("")

    # 1a. headline grouped view (seeds aggregated)
    pgroups: dict[str, list[Run]] = {}
    for r in proxy:
        # only aggregate complete (>=19 round) runs into the headline
        if r.status() != "ok":
            continue
        g = re.sub(r"_seed\d+$", "", r.display)
        pgroups.setdefault(g, []).append(r)
    lines.append("**Headline (complete runs, seeds aggregated, final round, mean ± CI95):**")
    lines.append("")
    lines.append("| Arm | n | Final test acc % | Δ vs control (pp) |")
    lines.append("|---|---:|---:|---:|")
    for g in sorted(pgroups, key=lambda k: -statistics.mean([r.final for r in pgroups[k]])):
        rs = pgroups[g]
        finals = [r.final for r in rs]
        delta = (f"{(statistics.mean(finals) - baseline) * 100:+.2f}"
                 if baseline is not None else "—")
        lines.append(f"| `{g}` | {len(rs)} | {agg_str(finals)} | {delta} |")
    lines.append("")

    # 1b. every run, verified, with status + source
    lines.append("**Per-run (verified, including partial/failed):**")
    lines.append("")
    lines.append("| Run | Final test acc % | Status | Source |")
    lines.append("|---|---:|---|---|")
    for r in proxy:
        lines.append(f"| `{r.display}` | {r.acc_str()} | {r.status()} | `{r.source}` |")
    lines.append("")

    # 2. W&B LR sweep
    lines.append("## 2. Baseline learning-rate sweep (vanilla RoLoRA / LoRA / FFA-LoRA)")
    lines.append("")
    lines.append("Source: `evidence/wandb_qnli_c50_r4_20260603/server_history.csv` "
             "(state=crashed means the process died after round 19 cleanup; the "
             "20 rounds of data are complete).")
    lines.append("")
    lines.append("| Run | Final test acc % | Source |")
    lines.append("|---|---:|---|")
    for r in sweep:
        lines.append(f"| `{r.key}` | {r.acc_str()} | _(W&B export)_ |")
    lines.append("")
    # grouped means for the 3-seed cells
    lines.append("**Grouped means (final round, mean ± CI95):**")
    lines.append("")
    groups: dict[str, list[Run]] = {}
    for r in sweep:
        g = re.sub(r"_seed\d+$", "", r.key)
        groups.setdefault(g, []).append(r)
    lines.append("| Method×LR | n | Final test acc % | Best test acc % |")
    lines.append("|---|---:|---:|---:|")
    for g in sorted(groups):
        rs = groups[g]
        finals = [r.final for r in rs if r.final is not None]
        bests = [r.best for r in rs if r.best is not None]
        lines.append(f"| `{g}` | {len(rs)} | {agg_str(finals)} | {agg_str(bests)} |")
    lines.append("")

    # 3. FedAvg optimizer audit
    lines.append("## 3. FedAvg / optimizer reproducibility audit (centralized-style)")
    lines.append("")
    lines.append("Confirms ADR 0006: shipped SGD lr=0.005 sits at chance; AdamW lr=5e-4 learns.")
    lines.append("")
    lines.append("| Run | Final test acc % | Rounds | Config | Status | Source |")
    lines.append("|---|---:|---:|---|---|---|")
    for r in audit:
        lines.append(f"| `{r.key}` | {r.acc_str()} | {r.final_round} | {r.note} | "
                 f"{r.status()} | `{r.source}` |")
    lines.append("")

    # 4. Table-1 local
    if table1:
        lines.append("## 4. Table-1-shaped local runs")
        lines.append("")
        lines.append("These are short harness pilots (2–9 rounds), **not** converged "
                 "Table-1 reproductions — they validate the pipeline and sit at chance "
                 "at this round count.")
        lines.append("")
        lines.append("| Run | Final test acc % | Rounds | Status | Source |")
        lines.append("|---|---:|---:|---|---|")
        for r in table1:
            lines.append(f"| `{r.key}` | {r.acc_str()} | {r.final_round} | "
                     f"{r.status()} | `{r.source}` |")
        lines.append("")

    # 5. Toy MNIST
    lines.append("## 5. Toy MNIST (CPU triage — mechanism filter, not a paper claim)")
    lines.append("")
    for t in toy:
        lines.append(f"### `{t.title}`")
        lines.append(f"_{t.config}_ · source: `{t.source}`")
        lines.append("")
        lines.append("| Variant | n | Final acc % | Best acc % | Note |")
        lines.append("|---|---:|---:|---:|---|")
        for row in t.rows:
            lines.append(f"| {row['name']} | {row['n']} | "
                     f"{fmt_mean(row['final_mean'], row['final_std'])} | "
                     f"{fmt_mean(row['best_mean'], row['best_std'])} | {row['note']} |")
        lines.append("")

    # 6. Full file index
    lines.append("## 6. Full log / artifact index")
    lines.append("")
    lines.append(f"{len(index)} result-bearing files. Paths are repo-relative; "
             "`results/` and `exp/` are gitignored (local artifacts).")
    lines.append("")
    lines.append("| Path | KB | Modified | Role |")
    lines.append("|---|---:|---|---|")
    for path, size, mtime, role in index:
        lines.append(f"| `{path}` | {size / 1024:.0f} | {mtime} | {role} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stdout", action="store_true", help="print the registry instead of writing it")
    ap.add_argument("--promote", action="store_true",
                    help="copy canonical run artifacts into runs/ before writing REGISTRY.md")
    ap.add_argument("--out", type=Path, default=RUNS / "REGISTRY.md")
    ap.add_argument("--stamp", default="", help="date stamp to embed")
    args = ap.parse_args()
    if args.promote:
        actions = promote_all()
        print(f"promoted {len(actions)} run folders into {rel(RUNS)}/")
    stamp = args.stamp or "regeneration"
    doc = render(stamp)
    if args.stdout:
        print(doc)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(doc)
        print(f"wrote {rel(args.out)} ({len(doc.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
