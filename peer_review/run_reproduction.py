"""Prepare a peer's frozen evaluation; execute only after human review in isolation.

This helper DOES NOT create a security sandbox. A venv is not a sandbox. Default
mode never launches peer code, installs packages, downloads data, or submits an
Issue. Use --execute only in a credential-free disposable isolation boundary.

Config JSON fields: submission (positive integer), claimed_accuracy (0..100),
code_commit (full Git SHA), peer_root, evaluator_dir, python, method, checkpoint
(path or null), output_dir. All paths must be absolute. The method and checkpoint
must be inside peer_root. evaluator_dir contains unchanged official predict.py,
score_predictions.py and common.py, with the already verified official data.

Human review JSON fields: reviewer_type="human", reviewed_by (name), plan_sha256
(from prepared_plan.json), isolated_environment_confirmed=true,
network_disabled=true, official_data_verified=true. Preparation hashes all peer
files (except .git, .venv, __pycache__) and official evaluation Python sources.
Review must cover the exact current plan and those source/asset hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def absolute(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"An absolute path is required: {value}")
    return path.resolve(strict=True)


def within(path: Path, root: Path) -> None:
    if not path.is_relative_to(root):
        raise ValueError(f"Path must remain inside peer_root: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--human-review", type=Path)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8-sig"))
    if type(cfg["submission"]) is not int or cfg["submission"] < 1:
        raise ValueError("submission must be a positive integer")
    if not isinstance(cfg["claimed_accuracy"], (int, float)) or not 0 <= cfg["claimed_accuracy"] <= 100:
        raise ValueError("claimed_accuracy must be an official percentage in 0..100")
    commit = cfg["code_commit"]
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("code_commit must be a full lowercase Git commit SHA")
    peer = absolute(cfg["peer_root"])
    evaluator = absolute(cfg["evaluator_dir"])
    python = absolute(cfg["python"])
    method = absolute(cfg["method"])
    within(method, peer)
    checkpoint = absolute(cfg["checkpoint"]) if cfg.get("checkpoint") else None
    if checkpoint is not None:
        within(checkpoint, peer)
    out = Path(cfg["output_dir"])
    if not out.is_absolute():
        raise ValueError("output_dir must be absolute")
    out = out.resolve()
    if out.is_relative_to(peer) or out.is_relative_to(evaluator):
        raise ValueError("output_dir must be outside peer and evaluator directories")
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for path in sorted(peer.rglob("*")):
        relative = path.relative_to(peer)
        if any(part in {".git", ".venv", "__pycache__"} for part in relative.parts):
            continue
        if path.is_symlink():
            raise ValueError(f"Review and replace symlink before execution: {path}")
        if path.is_file():
            files[relative.as_posix()] = {"sha256": digest(path), "bytes": path.stat().st_size}
    official = {name: digest(evaluator / name) for name in ("predict.py", "score_predictions.py", "common.py")}
    pred = [str(python), str(evaluator / "predict.py"), "--method", str(method), "--split", "test", "--output", str(out / "predictions.json")]
    if checkpoint is not None:
        pred += ["--checkpoint", str(checkpoint)]
    score = [str(python), str(evaluator / "score_predictions.py"), "--split", "test", "--predictions", str(out / "predictions.json"), "--output", str(out / "score.json")]
    plan = {"config": cfg, "peer_files": files, "official_sources": official, "cwd": str(evaluator), "commands": [pred, score], "threads_each_pool": 1}
    plan_hash = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    (out / "prepared_plan.json").write_text(json.dumps({"plan_sha256": plan_hash, "plan": plan}, indent=2), encoding="utf-8")
    if not args.execute:
        print(json.dumps({"status": "prepared_only", "plan_sha256": plan_hash, "output_dir": str(out), "peer_code_executed": False}))
        return 0
    if args.human_review is None:
        raise ValueError("--execute requires --human-review; no peer code executed")
    review = json.loads(args.human_review.read_text(encoding="utf-8-sig"))
    required = (review.get("reviewer_type") == "human", bool(str(review.get("reviewed_by", "")).strip()), review.get("plan_sha256") == plan_hash)
    if not all(required) or not all(review.get(k) is True for k in ("isolated_environment_confirmed", "network_disabled", "official_data_verified")):
        raise ValueError("Human review must attest the exact plan, isolation, offline operation and verified official data")
    if any((out / name).exists() for name in ("predictions.json", "score.json", "execution.json")):
        raise ValueError("Use a new empty output_dir for every actual reproduction; logs are never overwritten")
    # This allowlist reduces accidental env credential inheritance. It does not
    # remove credentials/files accessible to the OS user; external isolation is required.
    env = {key: os.environ[key] for key in ("SYSTEMROOT", "WINDIR", "COMSPEC", "PATH", "PATHEXT", "TEMP", "TMP", "LANG", "LC_ALL") if key in os.environ}
    env.update({key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")})
    env["PYTHONNOUSERSITE"] = "1"
    record = {"status": "running", "plan_sha256": plan_hash, "human_review": review, "started_at": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(), "controller_python": sys.version, "processes": []}
    start = time.perf_counter()
    try:
        for label, command in zip(("predict", "score"), (pred, score)):
            t = time.perf_counter()
            with (out / f"{label}.stdout.log").open("w", encoding="utf-8") as stdout, (out / f"{label}.stderr.log").open("w", encoding="utf-8") as stderr:
                proc = subprocess.run(command, cwd=evaluator, env=env, stdout=stdout, stderr=stderr, timeout=300, shell=False)
            record["processes"].append({"name": label, "command": command, "returncode": proc.returncode, "wall_seconds": time.perf_counter() - t})
            if proc.returncode:
                raise RuntimeError(f"{label} failed with exit code {proc.returncode}; no valid reproduced score")
        result = json.loads((out / "score.json").read_text(encoding="utf-8"))
        accuracy = result.get("accuracy")
        if not isinstance(accuracy, (int, float)) or not 0 <= accuracy <= 100:
            raise ValueError("Official score lacks a valid accuracy percentage")
        record.update(status="reproduced", official_score=result, score_sha256=digest(out / "score.json"), difference_percentage_points=accuracy-cfg["claimed_accuracy"])
    except Exception as error:
        record.update(status="blocked", error=f"{type(error).__name__}: {error}")
    finally:
        record["total_wall_seconds"] = time.perf_counter() - start
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        (out / "execution.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({"status": record["status"], "record": str(out / "execution.json")}))
    return 0 if record["status"] == "reproduced" else 1


if __name__ == "__main__":
    raise SystemExit(main())
