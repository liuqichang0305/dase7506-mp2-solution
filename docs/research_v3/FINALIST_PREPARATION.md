# Prepare explicit finalists, then copy the uniquely confirmed method

`scripts/prepare_finalists_v3.py` has three narrow commands. It does not close research, rank or automatically select candidates, train a model, predict, score, access the network, or open official validation/test inputs. Root chooses explicit IDs from completed train-only evidence separately. Production commands have not been executed during implementation.

The helper imports the existing confirmation/freezer validators without editing them. Its CLI takes their shared cooperative resource lock, rejects an active scheduler/trainer, and installs a permanent process-local train/manifest-only data guard. Post-confirmation copying reads saved validation metadata and hashes saved outputs, not official validation inputs/references. Ranking/background inputs and noncanonical paths are rejected.

## 1. Register at most two explicit single-model candidates

Only after root has genuinely written `SEARCH_CLOSED.json`, run from the repository root (replace the example ID and rationale with the actual deliberate decision):

```powershell
.venv/Scripts/python.exe scripts/prepare_finalists_v3.py select --experiment chosen_experiment_s0 --analysis results/research_v3/analysis/final_search/analysis.json --rationale "Document the actual train-OOF evidence and tradeoff for retaining this recipe."
```

Repeat `--experiment second_experiment_s0` for a second candidate. Zero, more than two, duplicates, incomplete CV and any base seed other than 3036900957 are rejected. There is no fusion support or candidate-display-name override. Candidate name equals its experiment ID; its full-train output is `artifacts/research_v3/final_<experiment_id>`.

`FINALISTS_SELECTION.json` uses the existing `mp2-finalists-selection-v3` contract: created time, deployment seed, exact allowed final-fit ID union, one/two finalists with component ID and structured selector. Additional fields preserve the chosen config file/hash, planned checkpoint, actual train-only aggregate metrics, explicit rationale and analysis path/hash, complete source/data/CV/score/asset evidence mapping, helper producer path/hash and exact command argument arrays.

Validation includes the complete disjoint 4,755-question/three-fold/nested-stopping partition and equality to the preserved V2 common folds; exact eleven training sources/four train-manifest files; queue/config/registration/summary provenance; all three completed 1,585-question fold reports and deterministic seed offsets; actual CV score/OOF and whole numeric bundle hashes; actual analysis counts/fold means. No OOF arrays or raw labels are decoded. A new selection is refused if the specified final fits/states/assets already exist. Matching repeat invocations verify and return the original bytes; changed requested selection/rationale never replaces them.

The completed scheduler receipt must also name the exact canonical summary path and match its SHA256, OOF SHA256 and saved full/hard accuracy. This prevents a refreshed secondary analysis from silently replacing an earlier completed-job receipt. Parameter and numeric-byte summaries agree with the saved fold reports. Full question-level OOF reanalysis remains the separate analyzer's responsibility; this helper verifies its saved provenance and aggregates instead of repeating it.

The helper prints `final_fit_commands` as argument arrays. Execute those exact commands using the existing trainer, for example:

```powershell
.venv/Scripts/python.exe code/train_v3.py --config code/configs/research_v3/chosen_experiment_s0.json --final-output artifacts/research_v3/final_chosen_experiment_s0
```

The trainer already requires this selection before final fitting and supports its own trusted epoch recovery. This helper does not modify that trainer or initiate it.

## 2. Lock the actual completed full-train bundles

```powershell
.venv/Scripts/python.exe scripts/prepare_finalists_v3.py lock
```

`FINALISTS_LOCK.json` uses `mp2-finalists-lock-v3`, the exact selection/closure/manifest/fallback hashes, required inference-source closure, whole candidate asset union, and actual components with final-fit/complete-CV hashes. It additionally preserves the exact allowed ID union, full-train report/state hashes and producer hash. The inference source mapping remains under `code/`; the preparation helper is provenance metadata, not an inference dependency or a change to the eleven training sources.

Full-train report configuration/provenance/checkpoint must match the selected CV recipe. Actual exported architecture, preprocessing and decoder must match the configuration. Neural models require the fixed full-train state path/hash, `final-all-train` context, median inner-best epoch (ceil rule), and the same live/EMA policy. Every inference file is audited and the entire bundle must fit 16 MiB.

Before publishing the immutable lock, the unchanged `confirm_validation_v3.validate_lock()` runs against the proposed lock. Only its read/hash of that single not-yet-published lock path is temporarily supplied in memory; all other sources, data hashes, numeric assets, trusted final state and fallback evidence are read from the actual files. These two function substitutions are restored even on failure. A failed prevalidation writes no final lock. After exclusive creation, the validator is called normally on the real lock again. Existing matching locks are verified, never rewritten; a changed lock is refused.

After locking, root invokes the existing confirmation command separately. This helper never writes `VALIDATION_OPENED.json` or enables validation by itself.

## 3. Copy only the saved unique post-validation selection

After the actual `validation_summary.json` is complete:

```powershell
.venv/Scripts/python.exe scripts/prepare_finalists_v3.py copy-selected
```

This command checks the unchanged lock and the summary's already recorded selection against the existing fixed confirmation rule. It copies that exact original bundle to `artifacts/research_v3/final_selected`, byte for byte. A new single-model finalist retains its locked component/selector; a historical V2 fallback gets its original fitted selector, exact V2 freeze component and `results/research_v2/<ID>` OOF directory. No alternate model, seed, epoch or fusion is fitted or considered.

The fixed `submission/v3/SELECTED_MODEL.json` schema is `mp2-selected-model-v3`, including name/method/fixed checkpoint, origin checkpoint, historical-fallback flag, deployment seed, structured selector, exact components, matching OOF directory, validation-summary hash, original asset map and helper producer hash. The existing `freeze_v3.validate_selected` validates the proposed record against the actual completed copy before publication, then validates the real saved record normally.

`SELECTED_COPY_OPENED.json` is persisted before copying. A partial marker, partial directory or unmatched preexisting output is preserved and refused automatically; this command never fills over or deletes it. Matching complete selected records/copies can be verified without rewriting. No copy operation is allowed after V3 freeze/test opening.

All records retain explicit timestamps and producer SHA256. This helper does not require an additional permission, approval email or inference-source inclusion. It does not write or weaken the existing trainer, confirmation, freeze, inference or official scoring source.

## Synthetic validation

`code/tests/test_prepare_finalists_v3.py` creates temporary synthetic train metadata, opaque score/state bytes and valid tiny numeric inference bundles. It exercises the actual unchanged confirmation and selected-copy validators without any validation/test bank files. Initial 12/12 passed in 14.755 seconds (`results/research_v3/finalist-preparation-synthetic-tests.log`); subsequent two-candidate and whole-asset guards have separate receipts.

Final expanded suite: 15/15 passed in 17.754 seconds (`results/research_v3/finalist-preparation-complete-synthetic-tests.log`). It includes an actual two-candidate strict lock, missing fold, wrong deployment seed, third/duplicate candidate, stale scheduler receipt, changed sources/scores, extra assets, full-train state/epoch mismatch, immutable repetition, strict-prevalidation failure without publishing a lock, byte-identical new/fallback copies, inconsistent chosen metadata and preserved partial copies. No production closure, selection, training, validation, test or copy command was run.
