"""Independent startup integrity check; no training or data/reference parsing."""
import hashlib
import argparse
import json
import pathlib
import subprocess
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument("--output", default="results/research_v3/audit_startup_preservation.json")
args = parser.parse_args()
out = (ROOT / args.output).resolve()
if not out.is_relative_to(ROOT / "results/research_v3"):
    raise SystemExit("Audit output must stay inside results/research_v3")
if out.exists():
    raise SystemExit("Refusing to overwrite audit evidence; choose a new --output")

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

checks = []
for version, rel in [("v1", "submission/PRE_TEST_FREEZE.json"), ("v2", "results/research_v2/V2_FREEZE.json")]:
    freeze = json.loads((ROOT / rel).read_text(encoding="utf-8"))
    for category in ("source_files", "assets"):
        for path, expected in freeze[category].items():
            actual = sha(ROOT / path)
            checks.append(dict(version=version, category=category, path=path, expected=expected, actual=actual, matches=expected == actual))
old = json.loads((ROOT / "results/research_v2/rules_preservation_check.json").read_text(encoding="utf-8"))
old_checks = []
for row in old["checks"]:
    actual = sha(ROOT / row["path"])
    old_checks.append(dict(path=row["path"], expected=row["expected"], actual=actual, matches=actual == row["expected"]))

# Record old tracked bytes for end-of-campaign preservation verification.
# Historical result files are only hashed, never parsed or sent to development.
# Official train/validation/test data files are not read here.
tracked = git("ls-files").splitlines()
inventory = {}
for rel in tracked:
    if rel.startswith(("artifacts/", "results/", "reports/", "submission/", "docs/", "code/")) and not rel.startswith("code/data/"):
        path = ROOT / rel
        if path.is_file():
            inventory[rel] = {"sha256": sha(path), "bytes": path.stat().st_size}
untracked_prior = {}
for rel in ("results/leaderboard-review-2026-10-29.json", "results/leaderboard-review-2026-10-30.json"):
    path = ROOT / rel
    untracked_prior[rel] = {"exists": path.is_file(), "sha256": sha(path) if path.is_file() else None, "bytes": path.stat().st_size if path.is_file() else None}
prompt = pathlib.Path(r"C:\Users\Liuye\Downloads\DASE7506_MP2_Autonomous_Research_Prompt.txt")
saved = ROOT / "docs/research_v3/AUTONOMOUS_RESEARCH_SPEC.txt"
result = {
    "schema": "mp2-v3-startup-preservation-v1",
    "checked_at_utc": datetime.now(timezone.utc).isoformat(),
    "branch": git("branch", "--show-current"), "head": git("rev-parse", "HEAD"),
    "tracked_diff_paths": git("diff", "--name-only").splitlines(),
    "frozen_checks": checks, "all_frozen_hashes_match": all(x["matches"] for x in checks),
    "prior_v1_92_checks": old_checks, "all_prior_v1_hashes_match": all(x["matches"] for x in old_checks),
    "historical_inventory": inventory, "historical_inventory_count": len(inventory),
    "preexisting_untracked_leaderboard_logs": untracked_prior,
    "user_prompt": {"source": str(prompt), "sha256": sha(prompt), "line_count": len(prompt.read_text(encoding="utf-8-sig").splitlines()), "archived_copy": str(saved.relative_to(ROOT)), "archived_sha256": sha(saved), "matches": sha(prompt) == sha(saved)},
    "data_manifest_sha256": sha(ROOT / "code/data/manifest.json"),
    "test_data_read": False, "historical_result_hashes_only": True,
}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps({k: result[k] for k in ("checked_at_utc", "branch", "head", "all_frozen_hashes_match", "all_prior_v1_hashes_match", "historical_inventory_count", "user_prompt", "test_data_read")}, indent=2))
if not result["all_frozen_hashes_match"] or not result["all_prior_v1_hashes_match"] or not result["user_prompt"]["matches"]:
    raise SystemExit(1)
